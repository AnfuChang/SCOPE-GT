#!/usr/bin/env bash
# Gophormer (Zhao et al., arXiv:2110.13094) under the shared comparison protocol.
#   bash run_gophormer.sh <dataset>                     e.g. bash run_gophormer.sh cora
#   RUNS=10 SPLIT_SUFFIXES="_0 _1 _2 _3 _4 _5 _6 _7 _8 _9" bash run_gophormer.sh roman_empire   (10 official splits)
#   LR=1e-4 LAYERS=4 bash run_gophormer.sh pubmed       (any hyperparameter below can be overridden by name)
# This dense NPZ adapter is a paper-guided reimplementation, separate from the upstream DGL code.
set -euo pipefail

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Fixed by project comparison protocol (identical in every run_*.sh, see ../common/protocol.py)
RUNS="${RUNS:-5}"                      # seeds BASE_SEED .. BASE_SEED+RUNS-1 (previous version of this script: 2)
BASE_SEED="${BASE_SEED:-42}"
EPOCHS="${EPOCHS:-1000}"               # upper bound on epochs
PATIENCE="${PATIENCE:-30}"             # validation checks without improvement
EVAL_EVERY="${EVAL_EVERY:-5}"          # one validation check every EVAL_EVERY epochs (+ epoch 1 and the last)
SPLIT_SUFFIXES="${SPLIT_SUFFIXES:-}"   # empty -> train.npy/val.npy/test.npy; run r uses suffix r % count
METRIC="${METRIC:-}"                   # empty -> ROC-AUC for questions/minesweeper/tolokers, accuracy otherwise

DATASET=${1:-cora}
# Data layout: ${DATA_ROOT}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
DATA_ROOT=${DATA_ROOT:-/mnt/sde/changaf25/sdust/datasets/${DATASET}}
# DATA_ROOT may also be the folder that holds all datasets
if [[ ! -f "${DATA_ROOT}/adj.npz" && -f "${DATA_ROOT}/${DATASET}/adj.npz" ]]; then DATA_ROOT="${DATA_ROOT}/${DATASET}"; fi
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

TUNABLE=(HID_DIM NUM_HEADS DROPOUT WEIGHT_DECAY END_LR FANOUTS LAYERS LR BATCH_SIZE GLOBAL_NODES PROX_ORDER CR_WEIGHT
         DIM_FEEDFORWARD TRAIN_SAMPLES EVAL_SAMPLES CR_TEMPERATURE EVAL_BATCH_SIZE GRAD_CLIP WARMUP_EPOCHS LR_DECAY
         LR_PATIENCE)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- Gophormer hyperparameters. The paper (Sec. 4.1.4) fixes dropout 0.5, weight decay 1e-5, depth D = 2,
#      hidden 64, 8 heads and end lr 1e-9, and grid-searches per dataset: lr {1e-4, 2e-4}, batch {32, 64, 128},
#      layers {1..5}, global nodes {0..4}, proximity views M {2, 3, 4} (M = PROX_ORDER + 2) and lambda
#      {0.5, 0.75, 1}. The selected values are not published , so every dataset uses the fixed
#      values plus one point of that grid: lr 2e-4, 2 global nodes, M = 3, lambda 0.5, and the depth the paper
#      reports as best (Sec. 4.5.2: 2-3 layers on small graphs such as Cora, 3-4 on larger ones). Batch size 128
#      (inside the grid) on the graphs with > 40k nodes keeps the epoch time manageable. Not given in the paper:
#      neighbours per hop (7, 7), S = 4 training ego-graphs, S' = 5 inference samples (Fig. 5 saturates at 5-7),
#      FFN 4 x hidden, sharpening temperature 0.5, warm-up 10 epochs, ReduceLROnPlateau factor 0.5 / patience 5.
#      A validation-only search over the paper's grid is available:
#        bash ../tune_baseline.sh gophormer <dataset> "LR=1e-4,2e-4" "LAYERS=2,3,4" "GLOBAL_NODES=0,2,4"
HID_DIM=64; NUM_HEADS=8; DROPOUT=0.5; WEIGHT_DECAY=1e-5; END_LR=1e-9; FANOUTS="7,7"
LR=0.0002; BATCH_SIZE=32; GLOBAL_NODES=2; PROX_ORDER=1; CR_WEIGHT=0.5
DIM_FEEDFORWARD=256; TRAIN_SAMPLES=4; EVAL_SAMPLES=5; CR_TEMPERATURE=0.5; EVAL_BATCH_SIZE=64; GRAD_CLIP=1.0
WARMUP_EPOCHS=10; LR_DECAY=0.5; LR_PATIENCE=5
case "${KEY}" in
  # small graph (paper dataset): 2 layers
  cora|citeseer)
    LAYERS=2; SRC="paper:fixed+grid(L=2,small-graph)" ;;
  # paper dataset / medium graphs (< 40k nodes): 3 layers
  pubmed|amazon_photo|photo|amazon_computers|computers|coauthor_cs|cs|wikics|corafull|facebook_large|facebook \
  |roman_empire|amazon_ratings|minesweeper|tolokers|squirrel|chameleon)
    LAYERS=3; SRC="paper:fixed+grid(L=3)" ;;
  questions)
    LAYERS=3; BATCH_SIZE=128; SRC="paper:fixed+grid(L=3,batch128)" ;;
  # large graphs: 4 layers, batch 128. The ego-graph sampler and the proximity matrices run in Python on the CPU
  # (as in the paper's DGL implementation), so one epoch over ~10^5-10^6 training nodes takes hours.
  ogbn_arxiv|arxiv|arxiv_year|ogbn_products|products|pokec)
    LAYERS=4; BATCH_SIZE=128; EVAL_BATCH_SIZE=256; SRC="paper:fixed+grid(L=4,batch128)" ;;
  *)
    echo "note: '${DATASET}' has no setting of its own, using the medium-graph one" >&2
    LAYERS=3; SRC="paper:fixed+grid(L=3)" ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done

echo "==== Gophormer | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/gophormer_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --hid-dim "${HID_DIM}" --num-heads "${NUM_HEADS}" --dropout "${DROPOUT}" --weight-decay "${WEIGHT_DECAY}"
  --end-lr "${END_LR}" --fanouts "${FANOUTS}" --num-layers "${LAYERS}" --lr "${LR}" --batch-size "${BATCH_SIZE}"
  --global-nodes "${GLOBAL_NODES}" --prox-order "${PROX_ORDER}" --cr-weight "${CR_WEIGHT}"
  --dim-feedforward "${DIM_FEEDFORWARD}" --train-samples "${TRAIN_SAMPLES}" --eval-samples "${EVAL_SAMPLES}"
  --cr-temperature "${CR_TEMPERATURE}" --eval-batch-size "${EVAL_BATCH_SIZE}" --grad-clip "${GRAD_CLIP}"
  --warmup-epochs "${WARMUP_EPOCHS}" --lr-decay "${LR_DECAY}" --lr-patience "${LR_PATIENCE}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
