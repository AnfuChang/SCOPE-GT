#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_vcr_baseline.sh <dataset> [data_dir] [device]
# Example:
#   bash run_vcr_baseline.sh pubmed /mnt/sdb/changaf25/sdust/datasets/pubmed 0
#   bash run_vcr_graphormer.sh roman_empire          (data_dir = ${DATA_ROOT}/roman_empire, device = ${DEVICE:-1})
# Optional environment overrides:
#   NUM_RUNS=5 SEED_START=42 EPOCHS=1000 PATIENCE=100 BATCH_SIZE=2000 bash run_vcr_baseline.sh flickr /path/to/flickr 0
# DATA_ROOT may point either at the folder holding all datasets or (as in the previous version of this
# script) directly at one dataset folder.
# Metric: auto -> ROC-AUC (model selection, output and mean/std) on questions/minesweeper/tolokers.

DATASET=${1:?"dataset name required"}
if [[ -n "${2:-}" ]]; then
  DATA_DIR="$2"
elif [[ -n "${DATA_DIR:-}" ]]; then
  :
elif [[ -n "${DATA_ROOT:-}" && -f "${DATA_ROOT}/adj.npz" ]]; then
  DATA_DIR="${DATA_ROOT}"
else
  DATA_DIR="${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}/${DATASET}"
fi
DEVICE=${3:-${DEVICE:-1}}

NUM_RUNS=${NUM_RUNS:-5}
SEED_START=${SEED_START:-42}
EPOCHS=${EPOCHS:-1000}
PATIENCE=${PATIENCE:-100}
EVAL_EVERY=${EVAL_EVERY:-5}
BATCH_SIZE=${BATCH_SIZE:-2000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$BATCH_SIZE}
CACHE_DIR=${CACHE_DIR:-"${DATA_DIR}/VCRGraphormer_cache"}
PARTITION_METHOD=${PARTITION_METHOD:-auto}
CONTENT_LABEL_SOURCE=${CONTENT_LABEL_SOURCE:-train}
LOAD_TOKENS_TO_RAM=${LOAD_TOKENS_TO_RAM:-auto}
RAM_TOKEN_LIMIT_GB=${RAM_TOKEN_LIMIT_GB:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
TO_UNDIRECTED=${TO_UNDIRECTED:-1}
METRIC=${METRIC:-auto}

# VCR paper/demo hyperparameters when available (demo.txt of the repo: hops / structure super-nodes /
# structure tokens / content tokens / hidden / heads / layers, batch 2000; train.py defaults:
# dropout 0.1, attention dropout 0.1, peak lr 1e-3 -> 1e-4, warm-up 400, weight decay 1e-5, FFN 64).
# For datasets not in the VCR paper, the closest demo.txt command is used.
case "${DATASET,,}" in
  pubmed)
    SRC="official:demo.txt-pubmed"
    HOPS=${HOPS:-7};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-16}; CONTENT_TOK=${CONTENT_TOK:-13}; HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  corafull|cora_full)
    SRC="official:demo.txt-corafull"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-20};  STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  computer|computers|amazon_computer|amazon_computers|amazon-computers)
    SRC="official:demo.txt-computer"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-11}; CONTENT_TOK=${CONTENT_TOK:-16}; HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  photo|amazon_photo|amazon-photo)
    SRC="official:demo.txt-photo"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  cs|coauthor_cs|coauthor-cs)
    SRC="official:demo.txt-cs"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-20};  STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-3} ;;
  physics)
    SRC="official:demo.txt-physics"
    HOPS=${HOPS:-4};  STRUCT_SUPER=${STRUCT_SUPER:-50};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-20}; HIDDEN=${HIDDEN:-128}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  reddit)
    SRC="official:demo.txt-reddit"
    HOPS=${HOPS:-6};  STRUCT_SUPER=${STRUCT_SUPER:-400}; STRUCT_TOK=${STRUCT_TOK:-20}; CONTENT_TOK=${CONTENT_TOK:-20}; HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  aminer)
    SRC="official:demo.txt-aminer"
    HOPS=${HOPS:-16}; STRUCT_SUPER=${STRUCT_SUPER:-100}; STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-9};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-16}; LAYERS=${LAYERS:-3} ;;
  amazon2m|ogbn-products|ogbn_products|products)
    # Amazon2M = the ogbn-products graph
    SRC="official:demo.txt-Amazon2M"
    HOPS=${HOPS:-10}; STRUCT_SUPER=${STRUCT_SUPER:-300}; STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-1};  LAYERS=${LAYERS:-1} ;;
  squirrel)
    # demo.txt squirrel = the original 5,201-node graph used here
    SRC="official:demo.txt-squirrel"
    HOPS=${HOPS:-5};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-1};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  actor)
    SRC="official:demo.txt-actor"
    HOPS=${HOPS:-2};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-1};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-1};  LAYERS=${LAYERS:-1} ;;
  texas)
    SRC="official:demo.txt-texas"
    HOPS=${HOPS:-2};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  arxiv-year|arxiv_year)
    # demo.txt: train_hetero.py --dataset arxiv-year (LINKX 50/25/25 split)
    SRC="official:demo.txt-arxiv-year"
    HOPS=${HOPS:-8};  STRUCT_SUPER=${STRUCT_SUPER:-500}; STRUCT_TOK=${STRUCT_TOK:-16}; CONTENT_TOK=${CONTENT_TOK:-6};  HIDDEN=${HIDDEN:-128}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  # ---- datasets without a demo.txt command
  ogbn-arxiv|ogbn_arxiv|arxiv)
    # same graph as arxiv-year -> arxiv-year command
    SRC="fallback:demo.txt-arxiv-year"
    HOPS=${HOPS:-8};  STRUCT_SUPER=${STRUCT_SUPER:-500}; STRUCT_TOK=${STRUCT_TOK:-16}; CONTENT_TOK=${CONTENT_TOK:-6};  HIDDEN=${HIDDEN:-128}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  pokec)
    # large heterophilous LINKX graph -> arxiv-year command (VCR's large heterophilous setting)
    SRC="fallback:demo.txt-arxiv-year"
    HOPS=${HOPS:-8};  STRUCT_SUPER=${STRUCT_SUPER:-500}; STRUCT_TOK=${STRUCT_TOK:-16}; CONTENT_TOK=${CONTENT_TOK:-6};  HIDDEN=${HIDDEN:-128}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  roman_empire|roman-empire|amazon_ratings|amazon-ratings|minesweeper|tolokers|questions)
    # heterophilous medium graphs -> squirrel command (the heterophilous command of a graph of this size)
    SRC="fallback:demo.txt-squirrel"
    HOPS=${HOPS:-5};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-1};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  wikics)
    # graph of Computer's size and density (11.7k nodes, mean degree 36.9 vs 13.8k, 35.8) -> computer command
    SRC="fallback:demo.txt-computer"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-11}; CONTENT_TOK=${CONTENT_TOK:-16}; HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  facebook_large|facebook|lastfm_asia)
    # social networks with binary node attributes (people as nodes, as in Coauthor) -> cs command
    SRC="fallback:demo.txt-cs"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-20};  STRUCT_TOK=${STRUCT_TOK:-5};  CONTENT_TOK=${CONTENT_TOK:-5};  HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-3} ;;
  dblp)
    # citation graph of PubMed's size (17.7k nodes) -> pubmed command
    SRC="fallback:demo.txt-pubmed"
    HOPS=${HOPS:-7};  STRUCT_SUPER=${STRUCT_SUPER:-10};  STRUCT_TOK=${STRUCT_TOK:-16}; CONTENT_TOK=${CONTENT_TOK:-13}; HIDDEN=${HIDDEN:-512}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  flickr)
    SRC="self-set"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-80};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-10}; HIDDEN=${HIDDEN:-256}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  github)
    SRC="self-set"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-80};  STRUCT_TOK=${STRUCT_TOK:-10}; CONTENT_TOK=${CONTENT_TOK:-10}; HIDDEN=${HIDDEN:-256}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
  *)
    SRC="generic-defaults"
    HOPS=${HOPS:-3};  STRUCT_SUPER=${STRUCT_SUPER:-50};  STRUCT_TOK=${STRUCT_TOK:-8};  CONTENT_TOK=${CONTENT_TOK:-8};  HIDDEN=${HIDDEN:-256}; HEADS=${HEADS:-8};  LAYERS=${LAYERS:-1} ;;
esac
echo "[VCR-Graphormer] dataset=${DATASET} data_dir=${DATA_DIR} device=${DEVICE} hyperparameters=${SRC} metric=${METRIC}"
echo "[VCR-Graphormer] hops=${HOPS} structure_supernodes=${STRUCT_SUPER} structure_tokens=${STRUCT_TOK} content_tokens=${CONTENT_TOK} hidden=${HIDDEN} heads=${HEADS} layers=${LAYERS}"

"${PYTHON:-python}" vcr_bilgt_baseline.py \
  --dataset "$DATASET" \
  --output_dir "${OUT_DIR:-results}" \
  --data_dir "$DATA_DIR" \
  --device "$DEVICE" \
  --metric "$METRIC" \
  --num_runs "$NUM_RUNS" \
  --seed_start "$SEED_START" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --cache_dir "$CACHE_DIR" \
  --eval_every "$EVAL_EVERY" \
  --hops "$HOPS" \
  --num_structure_supernodes "$STRUCT_SUPER" \
  --num_structure_tokens "$STRUCT_TOK" \
  --num_content_tokens "$CONTENT_TOK" \
  --hidden_dim "$HIDDEN" \
  --n_heads "$HEADS" \
  --n_layers "$LAYERS" \
  --partition_method "$PARTITION_METHOD" \
  --content_label_source "$CONTENT_LABEL_SOURCE" \
  --load_tokens_to_ram "$LOAD_TOKENS_TO_RAM" \
  --ram_token_limit_gb "$RAM_TOKEN_LIMIT_GB" \
  --num_workers "$NUM_WORKERS" \
  --to_undirected "$TO_UNDIRECTED" \
  ${EXTRA_ARGS:-}
