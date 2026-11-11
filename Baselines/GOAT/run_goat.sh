#!/usr/bin/env bash
# GOAT (ICML'23, github.com/devnkong/GOAT) under the shared comparison protocol.
#   bash run_goat.sh <dataset>                         e.g. bash run_goat.sh roman_empire
#   DEVICE=cuda:0 bash run_goat.sh ogbn_products
#   BATCH_SIZE=512 bash run_goat.sh tolokers           (any hyperparameter below can be overridden by name)
# Needs ../common/protocol.py, numpy/scipy/sklearn/torch; no torch_sparse, torch_geometric or einops.
# Code: goat_npz.py, goat_model.py, goat_sampler.py, node2vec_pe.py; official/ = unmodified repository files.
set -euo pipefail

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Fixed by project comparison protocol (identical in every run_*.sh, see ../common/protocol.py)
RUNS="${RUNS:-5}"                      # seeds BASE_SEED .. BASE_SEED+RUNS-1
BASE_SEED="${BASE_SEED:-42}"
EPOCHS="${EPOCHS:-1000}"               # upper bound on epochs (official: 500 epochs, best validation epoch)
PATIENCE="${PATIENCE:-30}"             # validation checks without improvement
EVAL_EVERY="${EVAL_EVERY:-5}"          # one validation check every EVAL_EVERY epochs (+ epoch 1 and the last)
SPLIT_SUFFIXES="${SPLIT_SUFFIXES:-}"   # empty -> train.npy/val.npy/test.npy; run r uses suffix r % count
METRIC="${METRIC:-}"                   # empty -> ROC-AUC for questions/minesweeper/tolokers, accuracy otherwise

DATASET=${1:-roman_empire}
# Data layout: ${DATA_ROOT}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets/${DATASET}}
# DATA_ROOT may also be the folder that holds all datasets
if [[ ! -f "${DATA_ROOT}/adj.npz" && -f "${DATA_ROOT}/${DATASET}/adj.npz" ]]; then DATA_ROOT="${DATA_ROOT}/${DATASET}"; fi
DEVICE="${DEVICE:-cuda:1}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
PE_CACHE="${PE_CACHE:-${SCRIPT_DIR}/cache}"   # node2vec encodings, one file per graph and setting (reused by all runs)
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

TUNABLE=(LR BATCH_SIZE TEST_BATCH_SIZE HIDDEN_DIM NUM_HEADS NUM_CENTROIDS GLOBAL_DIM FF_DROPOUT ATTN_DROPOUT
         SIZES TEST_SIZES NUM_WORKERS PE_SETTING CONV_TYPE)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- GOAT hyperparameters: the README commands of the repository (ogbn-arxiv, ogbn-products, arxiv-year,
#      snap-patents); arguments the commands do not set keep the defaults of arxiv_ERM_ns.py: conv full
#      (set by every command), global_dim 64, 1 layer, attn_dropout 0, ff_dropout 0.5, skip 0,
#      dist_count_norm 1, sizes = test_sizes = [20, 10, 5], batch_norm, Adam without weight decay.
#      --test_freq is replaced by EVAL_EVERY. Positional encodings: node2vec of pos_enc/arxiv_node2vec.py
#      (PE_SETTING=arxiv) or pos_enc/products_node2vec.py = pokec_node2vec.py (PE_SETTING=products), width 64.
#      Datasets outside the paper use the closest command by domain, then size and mean degree (marked fallback):
#        the 13 graphs with 7.6k-37.7k nodes -> ogbn-arxiv (closest in size; citation graphs as pubmed, dblp,
#        corafull); arxiv_year is in the paper (same command as ogbn-arxiv, test_freq 5);
#        pokec -> ogbn-products (1.6M nodes / mean degree 27 vs 2.4M / 50; snap-patents: 2.9M / 4.8 directed,
#        batch 8192). pokec_node2vec.py is official (same settings as products_node2vec.py).
#      wikics has 580 training nodes (< batch 1024): one batch per epoch (the official script asserts
#      batch_size <= #train).
CONV_TYPE=full; GLOBAL_DIM=64; FF_DROPOUT=0.5; ATTN_DROPOUT=0.0; SIZES="20 10 5"; TEST_SIZES="20 10 5"
NUM_WORKERS=4; LR=1e-3
arxiv_cmd() {     # README: ogbn-arxiv (arxiv-year: same with --test_freq 5)
  BATCH_SIZE=1024; TEST_BATCH_SIZE=256; HIDDEN_DIM=128; NUM_HEADS=4; NUM_CENTROIDS=4096; PE_SETTING=arxiv; }
products_cmd() {  # README: ogbn-products
  BATCH_SIZE=512; TEST_BATCH_SIZE=256; HIDDEN_DIM=256; NUM_HEADS=2; NUM_CENTROIDS=4096; PE_SETTING=products; }
case "${KEY}" in
  ogbn_arxiv|arxiv)          arxiv_cmd; SRC="official:README ogbn-arxiv" ;;
  arxiv_year)                arxiv_cmd; SRC="official:README arxiv-year" ;;
  ogbn_products|products)    products_cmd; SRC="official:README ogbn-products" ;;
  pokec)                     products_cmd; SRC="fallback:ogbn-products (PE: official pokec_node2vec.py)" ;;
  lastfm_asia|minesweeper|wikics|tolokers|amazon_computers|computers|dblp|coauthor_cs|cs|pubmed|corafull|\
  facebook_large|facebook|roman_empire|amazon_ratings|github|cora|citeseer|amazon_photo|photo|questions)
                             arxiv_cmd; SRC="fallback:ogbn-arxiv" ;;
  *)
    echo "note: '${DATASET}' has no setting of its own, using the ogbn-arxiv command" >&2
    arxiv_cmd; SRC="fallback:ogbn-arxiv" ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done

echo "==== GOAT | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/goat_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --lr "${LR}" --batch-size "${BATCH_SIZE}" --test-batch-size "${TEST_BATCH_SIZE}"
  --sizes ${SIZES} --test-sizes ${TEST_SIZES} --num-workers "${NUM_WORKERS}" --conv-type "${CONV_TYPE}"
  --hidden-dim "${HIDDEN_DIM}" --global-dim "${GLOBAL_DIM}" --num-heads "${NUM_HEADS}"
  --attn-dropout "${ATTN_DROPOUT}" --ff-dropout "${FF_DROPOUT}" --num-centroids "${NUM_CENTROIDS}"
  --pe-setting "${PE_SETTING}" --pe-cache "${PE_CACHE}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
