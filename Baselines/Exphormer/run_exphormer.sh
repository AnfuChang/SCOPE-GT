#!/usr/bin/env bash
set -euo pipefail

# BILGT-compatible launch script for the Exphormer baseline.
# Usage:
#   DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets DEVICE=cuda:0 bash run_exphormer_baseline.sh flickr
#   DATA_DIR=/path/to/one/dataset bash run_exphormer_baseline.sh pubmed cuda:1
#
# Output per run and final mean/std fields:
#   Test Acc(%)    Tepoch(s)    Mem.(MB)
#   (Test ROC_AUC on questions / minesweeper / tolokers: METRIC=auto selects, reports and averages ROC-AUC)

DATASET=${1:-pubmed}
DEVICE=${2:-${DEVICE:-cuda:1}}
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}
DATA_DIR=${DATA_DIR:-}
RUNS=${RUNS:-5}
SEED=${SEED:-42}
METRIC=${METRIC:-auto}

# Common comparison defaults used when the Exphormer paper does not provide a
# dataset-specific configuration for a BILGT dataset.
USER_EPOCHS=${EPOCHS:-}
USER_PATIENCE=${PATIENCE:-}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_LAYERS=${NUM_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-4}
FFN_DIM=${FFN_DIM:-256}
DROPOUT=${DROPOUT:-0.5}
ATTN_DROPOUT=${ATTN_DROPOUT:-0.5}
LR=${LR:-0.01}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
EPOCHS=${EPOCHS:-1000}
PATIENCE=${PATIENCE:-100}
EVAL_EVERY=${EVAL_EVERY:-5}
EXPANDER_DEGREE=${EXPANDER_DEGREE:-6}
NUM_VIRTUAL_NODES=${NUM_VIRTUAL_NODES:-0}
NORM=${NORM:-batch}
FEATURE_NORM=${FEATURE_NORM:-none}
OPTIMIZER=${OPTIMIZER:-AdamW}
SCHEDULER=${SCHEDULER:-cosine}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
NUM_THREADS=${NUM_THREADS:-4}

# Exphormer (ICML'23) configuration: layers hidden heads dropout attn_dropout lr.
# Shared by all transductive configs (configs/Exphormer/{arxiv,computer,photo,cs,physics}.yaml):
# GCN+Exphormer layers, expander degree 6 (exp_deg 3), 0 virtual nodes, AdamW, weight decay 1e-3,
# cosine schedule with 5 warm-up epochs, batch norm, FFN = 2 x hidden. Paper Table 9 gives layers /
# hidden / heads / dropout / lr / epochs; the attention dropout comes from the yaml files.
# *_OVERRIDE variables replace single values.
exph_cfg() {
  NUM_LAYERS=${NUM_LAYERS_OVERRIDE:-$1}
  HIDDEN_DIM=${HIDDEN_DIM_OVERRIDE:-$2}
  NUM_HEADS=${NUM_HEADS_OVERRIDE:-$3}
  FFN_DIM=${FFN_DIM_OVERRIDE:-$((2 * HIDDEN_DIM))}
  DROPOUT=${DROPOUT_OVERRIDE:-$4}
  ATTN_DROPOUT=${ATTN_DROPOUT_OVERRIDE:-$5}
  LR=${LR_OVERRIDE:-$6}
  WEIGHT_DECAY=${WEIGHT_DECAY_OVERRIDE:-${WEIGHT_DECAY}}   # default 0.001 = official
}

# Epochs / patience always follow the unified comparison protocol (official: 600 / 150 / 100 / 70
# epochs for arxiv / computer / photo / cs+physics).
case "${DATASET,,}" in
  ogbn_arxiv|ogbn-arxiv|arxiv)
    # official: paper Table 9 (3 layers, 96 hidden, 2 heads, dropout 0.3, lr 0.01, expander degree 6,
    # 0 virtual nodes); attention dropout 0.5 from configs/Exphormer/arxiv.yaml. (The yaml itself
    # lists 4 layers / 80 hidden / lr 1e-3; the paper values are used.)
    exph_cfg 3 96 2 0.3 0.5 0.01; SRC="official:paper-Table9" ;;
  computer|computers|amazon_computer|amazon_computers|amazon-computer|amazon-computers)
    exph_cfg 4 80 2 0.4 0.8 0.001; SRC="official:paper-Table9+computer.yaml" ;;
  photo|amazon_photo|amazon-photo)
    exph_cfg 4 64 2 0.4 0.8 0.001; SRC="official:paper-Table9+photo.yaml" ;;
  cs|coauthor_cs|coauthor-cs)
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="official:paper-Table9+cs.yaml" ;;
  physics|coauthor_physics|coauthor-physics)
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="official:paper-Table9+physics.yaml" ;;
  # ---- datasets without an Exphormer configuration: closest official configuration
  pubmed)
    # homophilous citation graph of Coauthor-CS size (19.7k vs 18.3k nodes) -> CS configuration
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="fallback:cs" ;;
  corafull|dblp)
    # citation graphs of PubMed's size (19.8k / 17.7k nodes) -> the pubmed setting (CS configuration)
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="fallback:cs" ;;
  facebook_large|facebook|lastfm_asia)
    # social networks with binary node attributes (people as nodes, as in Coauthor) -> CS configuration
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="fallback:cs" ;;
  github)
    # social network of Physics' size and density (37.7k nodes, mean degree 15.3 vs 34.5k, 14.4) -> Physics
    exph_cfg 4 72 2 0.4 0.8 0.001; SRC="fallback:physics" ;;
  wikics)
    # graph of Computer's size and density (11.7k nodes, mean degree 36.9 vs 13.8k, 35.8) -> Computer
    exph_cfg 4 80 2 0.4 0.8 0.001; SRC="fallback:computer" ;;
  squirrel)
    # 5.2k-node graph -> configuration of the closest-size graph (Photo, 7.6k nodes)
    exph_cfg 4 64 2 0.4 0.8 0.001; SRC="fallback:photo" ;;
  roman_empire|roman-empire|amazon_ratings|amazon-ratings|minesweeper|tolokers|questions)
    # heterophilous 10k-49k-node graphs -> configuration of the largest medium graph (Computer)
    exph_cfg 4 80 2 0.4 0.8 0.001; SRC="fallback:computer" ;;
  arxiv_year|arxiv-year)
    # same graph as ogbn-arxiv (year labels) -> ogbn-arxiv configuration
    exph_cfg 3 96 2 0.3 0.5 0.01; SRC="fallback:ogbn-arxiv" ;;
  ogbn_products|ogbn-products|products|pokec)
    # only large-graph configuration of the paper; the adapter trains full-batch, so these
    # 1.6M-2.4M-node graphs are expected to run out of GPU memory
    exph_cfg 3 96 2 0.3 0.5 0.01; SRC="fallback:ogbn-arxiv" ;;
  *)
    # BILGT comparison setting for datasets outside the comparison (e.g. cora/citeseer/flickr);
    # every main-table dataset has a branch above.
    SRC="generic-defaults"
    ;;
esac
EPOCHS="${EPOCHS_OVERRIDE:-${USER_EPOCHS:-1000}}"
PATIENCE="${PATIENCE_OVERRIDE:-${USER_PATIENCE:-100}}"
echo "[Exphormer] dataset=${DATASET} hyperparameters=${SRC} metric=${METRIC}"

CMD=("${PYTHON:-python}" exphormer_npz.py
  --out_dir "${OUT_DIR:-results}"
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --device "${DEVICE}"
  --runs "${RUNS}"
  --seed "${SEED}"
  --metric "${METRIC}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_layers "${NUM_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --ffn_dim "${FFN_DIM}"
  --dropout "${DROPOUT}"
  --attn_dropout "${ATTN_DROPOUT}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --optimizer "${OPTIMIZER}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --eval_every "${EVAL_EVERY}"
  --expander_degree "${EXPANDER_DEGREE}"
  --num_virtual_nodes "${NUM_VIRTUAL_NODES}"
  --norm "${NORM}"
  --feature_norm "${FEATURE_NORM}"
  --scheduler "${SCHEDULER}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --num_threads "${NUM_THREADS}"
)

if [[ -n "${DATA_DIR}" ]]; then
  CMD+=(--data_dir "${DATA_DIR}")
fi

EXTRA_ARGS=${EXTRA_ARGS:-}
echo "[Launch] ${CMD[*]} ${EXTRA_ARGS}"
# shellcheck disable=SC2086
"${CMD[@]}" ${EXTRA_ARGS}
