#!/usr/bin/env bash
# BILGT-compatible launcher for GraphGPS baseline.
# Usage examples:
#   DATA_ROOT=/mnt/sdb/changaf25/AURA_Former/data DEVICE=cuda:0 bash run_graphgps_baseline.sh pubmed
#   DATA_DIR=/path/to/flickr DEVICE=cuda:1 RUNS=5 bash run_graphgps_baseline.sh flickr
#
# Expected input files in ${DATA_ROOT}/${DATASET} or ${DATA_DIR}:
#   feat.npy / features.npy / x.npy
#   labels.npy / y.npy
#   adj.npz or edge_index.npy / edges.npy
#   train.npy, val.npy, test.npy  (or *_idx.npy / *_mask.npy)

set -euo pipefail

DATASET="${1:-${DATASET:-pubmed}}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}"
DATA_DIR="${DATA_DIR:-}"
DEVICE="${DEVICE:-cuda:1}"
RUNS="${RUNS:-5}"
SEED="${SEED:-42}"
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

# ---------------------------------------------------------------------------------------------
# Presets for the 14 datasets of the ACE_GTv6 experiments (': "${VAR:=v}"' -> environment wins).
# GraphGPS (NeurIPS'22) has no transductive node-classification experiment on these datasets; the
# repo contains exactly one such config, configs/GPS/wn-squirrel-GPS.yaml (PyG WikipediaNetwork
# squirrel, i.e. the 5,201-node graph used here):
#   GCN+Transformer, 2 layers, 4 heads, hidden 64, dropout 0.2, attention dropout 0, no batch/layer
#   norm in the GPS layers, LapPE (4 eigenvectors, DeepSet), AdamW lr 5e-4, weight decay 1e-5,
#   cosine schedule (official: 200 epochs with 10 warm-up epochs; epochs follow the protocol here),
#   raw node features (the official loader applies no feature normalisation).
#   [ACE fix] FEATURE_NORM=row (L1 rows) used to be applied here as well: it divides every row by its
#   L1 norm (amazon_computers: 267 non-zeros per row -> entries 1/267, amazon_ratings: /11), so the
#   input projection sees almost identical nodes (signal ~6% of its bias) and full-batch training sits
#   on the majority class for >100 epochs; PATIENCE then stops it there (amazon_ratings 36.80 +- 0.00 =
#   class prior in all runs, amazon_computers 37.43 in 4 of 5 runs).
#   squirrel                               : official (wn-squirrel-GPS.yaml)
#   amazon_photo, minesweeper, tolokers,
#   wikics, lastfm_asia                    : fallback = squirrel config (N <= 12k: dense Transformer)
#   amazon_computers, coauthor_cs, pubmed,
#   roman_empire, amazon_ratings, questions,
#   corafull, facebook_large, github, dblp : fallback = squirrel config, global attention = Performer
#                                            (GraphGPS's linear global attention; N > 12k)
#   ogbn_arxiv, arxiv_year, ogbn_products,
#   pokec                                  : fallback = squirrel config + Performer; log-degree encoding
#                                            instead of LapPE (eigsh on >= 169k nodes is impractical)
# ---------------------------------------------------------------------------------------------
gps_squirrel_recipe() {
  : "${HIDDEN:=64}"; : "${LAYERS:=2}"; : "${HEADS:=4}"; : "${DROPOUT:=0.2}"; : "${ATTN_DROPOUT:=0.0}"
  : "${OPTIMIZER:=adamw}"; : "${LR:=0.0005}"; : "${WEIGHT_DECAY:=1e-5}"; : "${SCHEDULER:=cosine}"; : "${NORM:=none}"
  : "${FEATURE_NORM:=none}"
}
PRESET=1
case "${KEY}" in
  squirrel)
    gps_squirrel_recipe; : "${GLOBAL_MODEL:=transformer}"; : "${LAP_PE_K:=4}"; : "${USE_DEGREE_PE:=0}"
    SRC="official:configs/GPS/wn-squirrel-GPS.yaml" ;;
  # every main-table dataset uses the squirrel recipe; the global attention follows the size rule above
  # (dense Transformer for N <= 12k: wikics 11.7k, lastfm_asia 7.6k; Performer for corafull, facebook_large,
  # github, dblp). These graphs used to get the generic defaults, including the L1 row normalisation above.
  amazon_photo|photo|minesweeper|tolokers|lastfm_asia|wikics)
    gps_squirrel_recipe; : "${GLOBAL_MODEL:=transformer}"; : "${LAP_PE_K:=4}"; : "${USE_DEGREE_PE:=0}"
    SRC="fallback:wn-squirrel(dense-Transformer)" ;;
  amazon_computers|computers|computer|coauthor_cs|cs|pubmed|roman_empire|amazon_ratings|questions|dblp|corafull|facebook_large|facebook|github)
    gps_squirrel_recipe; : "${GLOBAL_MODEL:=performer}"; : "${LAP_PE_K:=4}"; : "${USE_DEGREE_PE:=0}"
    SRC="fallback:wn-squirrel+Performer" ;;
  ogbn_arxiv|arxiv|arxiv_year|ogbn_products|products|pokec)
    gps_squirrel_recipe; : "${GLOBAL_MODEL:=performer}"; : "${LAP_PE_K:=0}"; : "${USE_DEGREE_PE:=1}"
    SRC="fallback:wn-squirrel+Performer+degree-encoding" ;;
  *)
    PRESET=0; SRC="generic-defaults" ;;
esac

# Default fair-comparison settings used by the user's other GT baselines.
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
EVAL_EVERY="${EVAL_EVERY:-5}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.5}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
LR="${LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SCHEDULER="${SCHEDULER:-none}"
NORM="${NORM:-batch}"
GLOBAL_MODEL="${GLOBAL_MODEL:-auto}"
FEATURE_NORM="${FEATURE_NORM:-row}"
USE_DEGREE_PE="${USE_DEGREE_PE:-1}"
LAP_PE_K="${LAP_PE_K:-0}"
# auto: ROC-AUC (model selection, per-run output and mean/std) on questions/minesweeper/tolokers, accuracy otherwise
METRIC="${METRIC:-auto}"
DENSE_LIMIT="${DENSE_LIMIT:-12000}"
NUM_THREADS="${NUM_THREADS:-4}"

# Dataset-aware defaults for datasets outside the list above. Paper GraphGPS is a modular recipe; for
# large single transductive graphs we use the linear global-attention branch to avoid N^2 OOM.
if [[ "${PRESET}" == "0" ]]; then
  case "${DATASET,,}" in
    cora|citeseer)
      GLOBAL_MODEL="${GLOBAL_MODEL/auto/transformer}"
      LR="${LR:-0.01}"
      WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
      LAP_PE_K="${LAP_PE_K:-4}"
      ;;
    corafull|flickr|github)
      GLOBAL_MODEL="${GLOBAL_MODEL/auto/performer}"
      # Large sparse graphs are usually more stable with a smaller LR for GPS.
      if [[ "${LR}" == "0.01" ]]; then LR="0.001"; fi
      if [[ "${WEIGHT_DECAY}" == "5e-4" ]]; then WEIGHT_DECAY="1e-4"; fi
      ;;
    *)
      GLOBAL_MODEL="${GLOBAL_MODEL/auto/performer}"
      ;;
  esac
fi

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --runs "${RUNS}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --eval_period "${EVAL_EVERY}"
  --hidden "${HIDDEN}"
  --layers "${LAYERS}"
  --heads "${HEADS}"
  --dropout "${DROPOUT}"
  --attn_dropout "${ATTN_DROPOUT}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --scheduler "${SCHEDULER}"
  --norm "${NORM}"
  --global_model "${GLOBAL_MODEL}"
  --feature_norm "${FEATURE_NORM}"
  --lap_pe_k "${LAP_PE_K}"
  --metric "${METRIC}"
  --dense_attention_limit "${DENSE_LIMIT}"
  --num_threads "${NUM_THREADS}"
)

if [[ -n "${DATA_DIR}" ]]; then
  COMMON_ARGS+=(--data_dir "${DATA_DIR}")
fi
if [[ "${USE_DEGREE_PE}" == "1" || "${USE_DEGREE_PE,,}" == "true" ]]; then
  COMMON_ARGS+=(--use_degree_pe)
else
  COMMON_ARGS+=(--no-use_degree_pe)
fi

cd "${PROJECT_DIR}"
echo "[GraphGPS-BILGT] dataset=${DATASET} data_root=${DATA_ROOT} data_dir=${DATA_DIR:-<auto>} device=${DEVICE} runs=${RUNS} seed=${SEED}"
echo "[GraphGPS-BILGT] hyperparameters=${SRC} metric=${METRIC}"
echo "[GraphGPS-BILGT] hidden=${HIDDEN} layers=${LAYERS} heads=${HEADS} global=${GLOBAL_MODEL} lr=${LR} wd=${WEIGHT_DECAY} dropout=${DROPOUT} norm=${NORM} feature_norm=${FEATURE_NORM} lap_pe_k=${LAP_PE_K} degree_pe=${USE_DEGREE_PE}"
"${PYTHON}" npz_adapter/graphgps_node_baseline.py "${COMMON_ARGS[@]}"
