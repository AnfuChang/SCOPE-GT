#!/usr/bin/env bash
set -euo pipefail

# NLGT official-style launcher adapted to unified npz/npy datasets.
# Usage:
#   bash run_nlgt.sh /path/to/dataset_root
# Dataset root must contain: adj.npz feat.npy label.npy train.npy val.npy test.npy
DATASET=${1:-citeseer}

# Data layout:
#   ${DATA_ROOT}/${DATASET}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}
if [[ "${DATASET}" == "custom" ]]; then
  if [[ -z "${DATA_DIR:-}" ]]; then
    echo "[Error] DATASET=custom requires DATA_DIR=/path/to/dataset" >&2
    exit 1
  fi
else
  DATA_DIR=${DATA_DIR:-${DATA_ROOT}/${DATASET}}
fi
DATA_ROOT_EFFECTIVE="${DATA_DIR}"

DEVICE="${DEVICE:-cuda:0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/nlgt_npz.py}"

DATASET_NAME="${DATASET_PRESET:-${DATASET}}"
if [[ "${DATASET_NAME}" == "custom" ]]; then
  DATASET_NAME="$(basename "${DATA_DIR}")"
fi

# -----------------------------
# Fixed comparison protocol required by your paper
# -----------------------------
RUNS="${RUNS:-5}"
PATIENCE="${PATIENCE:-50}"
BASE_SEED="${BASE_SEED:-42}"
# auto: ROC-AUC (model selection, per-run output and mean/std) on questions/minesweeper/tolokers, accuracy otherwise
METRIC="${METRIC:-auto}"

# -----------------------------
# Dataset presets for the 14 datasets of the ACE_GTv6 experiments (': "${VAR:=v}"', so values
# given in the environment still win).
# NLGT (AAAI'25) uses ONE shared configuration for all of its datasets (Cora, Citeseer, PubMed,
# ogbn-arxiv): "Implementation details" of the paper + train.py of the repo --
#   fanouts [10,5,2] (max 168 nodes), embedding 128, 4 layers, 2 heads, FFN 512, dropout 0.1,
#   SGD (momentum 0.9, weight decay 5e-5), batch 64, lr 1e-3 with cosine decay to lrf=0.01,
#   10-subgraph voting at inference (the paper trains at most 10 epochs; the epoch budget here is
#   the shared protocol).
#   pubmed, ogbn_arxiv : official (paper datasets, shared configuration)
#   other 12 datasets  : fallback = the same shared configuration (no other configuration exists)
# The large graphs only get a larger *inference* batch (no effect on results); NLGT samples one
# subgraph per node in Python and votes over 10 subgraphs, so ogbn_products / pokec are very slow.
# -----------------------------
DATASET_KEY="$(echo "${DATASET_NAME}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
NLGT_SHARED() {
  : "${FANOUT_1:=10}"; : "${FANOUT_2:=5}"; : "${FANOUT_3:=2}"; : "${MAX_NODES:=168}"
  : "${HID_DIM:=128}"; : "${NUM_LAYERS:=4}"; : "${NUM_HEADS:=2}"; : "${DIM_FEEDFORWARD:=512}"; : "${DROPOUT:=0.1}"
  : "${BATCH_SIZE:=64}"; : "${EVAL_VOTES:=10}"; : "${OPTIMIZER:=sgd}"; : "${LR:=1e-3}"; : "${LRF:=0.01}"
  : "${WEIGHT_DECAY:=5e-5}"; : "${MOMENTUM:=0.9}"
}
case "${DATASET_KEY}" in
  ogbn_arxiv|arxiv)
    # *_ARXIV variables of the previous version of this script are still honoured
    : "${NUM_LAYERS:=${NUM_LAYERS_ARXIV:-4}}"; : "${NUM_HEADS:=${NUM_HEADS_ARXIV:-2}}"
    NLGT_SHARED; : "${EVAL_BATCH_SIZE:=512}"; PRESET_SRC="official:shared-config(paper+train.py)" ;;
  pubmed)
    NLGT_SHARED; PRESET_SRC="official:shared-config(paper+train.py)" ;;
  roman_empire|amazon_ratings|minesweeper|tolokers|questions|squirrel|amazon_photo|photo|amazon_computers|computers|computer|coauthor_cs|cs|lastfm_asia|dblp \
  |wikics|corafull|facebook_large|facebook|github)
    # (wikics, corafull, facebook_large and github used to get the generic defaults below: 2 layers / 4 heads
    # instead of the paper's 4 layers / 2 heads)
    NLGT_SHARED; PRESET_SRC="fallback:shared-config" ;;
  arxiv_year|ogbn_products|products|pokec)
    NLGT_SHARED; : "${EVAL_BATCH_SIZE:=1024}"; PRESET_SRC="fallback:shared-config" ;;
  *)
    PRESET_SRC="generic-defaults" ;;
esac

# -----------------------------
# Generic defaults (datasets outside the list above keep the values of the previous version;
# note that the paper/repo use 4 layers and 2 heads)
# -----------------------------
FANOUT_1="${FANOUT_1:-10}"
FANOUT_2="${FANOUT_2:-5}"
FANOUT_3="${FANOUT_3:-2}"
MAX_NODES="${MAX_NODES:-168}"

HID_DIM="${HID_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-512}"
DROPOUT="${DROPOUT:-0.1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_VOTES="${EVAL_VOTES:-10}"
EVAL_EVERY="${EVAL_EVERY:-5}"

OPTIMIZER="${OPTIMIZER:-sgd}"
LR="${LR:-1e-3}"
LRF="${LRF:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
MOMENTUM="${MOMENTUM:-0.9}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
EPOCHS="${EPOCHS:-1000}"

# Fair baseline protocol for this project: no method may use labels as input features.
# Therefore, NLGT's label-enhanced neighbor-token input is disabled by default.
# The ground-truth center labels are still used only as supervised training targets.
STRICT_NO_NEIGHBOR_LABELS="${STRICT_NO_NEIGHBOR_LABELS:-0}"


echo "==== NLGT official-style adaptation ===="
echo "DATA_ROOT=${DATA_ROOT_EFFECTIVE}"
echo "DATASET_NAME=${DATASET_NAME} hyperparameters=${PRESET_SRC} metric=${METRIC}"
echo "DEVICE=${DEVICE}"
echo "RUNS=${RUNS}"
echo "PATIENCE=${PATIENCE}"
echo "FANOUTS=${FANOUT_1},${FANOUT_2},${FANOUT_3}"
echo "MAX_NODES=${MAX_NODES}"
echo "HID_DIM=${HID_DIM}"
echo "NUM_LAYERS=${NUM_LAYERS}"
echo "NUM_HEADS=${NUM_HEADS}"
echo "DIM_FEEDFORWARD=${DIM_FEEDFORWARD}"
echo "DROPOUT=${DROPOUT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "EVAL_VOTES=${EVAL_VOTES}"
echo "OPTIMIZER=${OPTIMIZER}"
echo "LR=${LR}"
echo "LRF=${LRF}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "MOMENTUM=${MOMENTUM}"
echo "EPOCHS=${EPOCHS}"
echo "STRICT_NO_NEIGHBOR_LABELS=${STRICT_NO_NEIGHBOR_LABELS}"

CMD=(
  "${PYTHON:-python}" "${PY_SCRIPT}"
  --data-root "${DATA_ROOT_EFFECTIVE}"
  --dataset "${DATASET_NAME}"
  --metric "${METRIC}"
  --device "${DEVICE}"
  --runs "${RUNS}"
  --base-seed "${BASE_SEED}"
  --fanouts "${FANOUT_1}" "${FANOUT_2}" "${FANOUT_3}"
  --max-nodes "${MAX_NODES}"
  --hid-dim "${HID_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --eval-votes "${EVAL_VOTES}"
  --eval-every "${EVAL_EVERY}"
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --lrf "${LRF}"
  --weight-decay "${WEIGHT_DECAY}"
  --momentum "${MOMENTUM}"
  --label-smoothing "${LABEL_SMOOTHING}"
  --grad-clip "${GRAD_CLIP}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
)

if [[ "${STRICT_NO_NEIGHBOR_LABELS}" == "1" ]]; then
  CMD+=(--strict-no-neighbor-labels)
fi

printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
