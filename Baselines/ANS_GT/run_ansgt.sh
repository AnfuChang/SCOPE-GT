#!/usr/bin/env bash
set -euo pipefail
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

DEVICE="${DEVICE:-cuda:1}"
RUNS="${RUNS:-5}"
BASE_SEED="${BASE_SEED:-42}"
# Classification OOT: one training epoch exceeds twelve hours; not a whole-run limit.
EPOCH_TIMEOUT_SECONDS="${EPOCH_TIMEOUT_SECONDS:-43200}"
# auto: ROC-AUC (model selection, per-run output and mean/std) on questions/minesweeper/tolokers, accuracy otherwise
METRIC="${METRIC:-auto}"

DATASET_NAME="${DATASET_PRESET:-${DATASET}}"
if [[ "${DATASET_NAME}" == "custom" ]]; then
  DATASET_NAME="$(basename "${DATA_DIR}")"
fi
DATASET_NAME="$(echo "${DATASET_NAME}" | tr '[:upper:]' '[:lower:]')"

# ---------------------------------------------------------------------------------------------
# Dataset presets for the 14 datasets of the ACE_GTv6 experiments. Values are assigned with
# ': "${VAR:=value}"', so any variable given in the environment still wins.
#
# ANS-GT (NeurIPS'22) publishes no per-dataset hyperparameters: the paper (Sec. 6.1 "Parameter
# Settings") gives one default setting -- dropout 0.5, hidden 128, 8 heads, peak lr 2e-4 (100-epoch
# warm-up, linear decay), 1000 epochs, sampler update period T=100, 20 sampled nodes, coarsening rate
# 0.01 -- and tunes batch size {8,16,32}, #layers {2..6}, #sampled nodes {10..25}, #super-nodes
# {0,3,6,9} and #global nodes {1,2,3} by grid search without releasing the chosen values; the repo
# (start.sh) only has a Cora command (4 layers, FFN 128, batch 32, dropout 0.1).
#   pubmed, squirrel : in the paper (60/20/20 splits) -> paper default setting, start.sh layers/FFN/batch
#   other 12 datasets: not in the paper -> the same default setting (fallback); on the four large
#                      graphs the batch is raised to 512 (outside the paper grid) and the kNN search
#                      is approximate, otherwise one epoch takes hours with the per-node sampler
# ---------------------------------------------------------------------------------------------
PAPER_DEFAULT() {
  : "${HID_DIM:=128}"; : "${NUM_HEADS:=8}"; : "${DROPOUT:=0.5}"; : "${LR:=2e-4}"
  : "${SAMPLER_UPDATE_PERIOD:=100}"; : "${COARSEN_RATE:=0.01}"; : "${NUM_SAMPLED_NODES:=20}"
  : "${NUM_LAYERS:=4}"; : "${DIM_FEEDFORWARD:=128}"; : "${NUM_SAMPLED_SUPERNODES:=3}"; : "${NUM_GLOBAL_NODES:=1}"
}
case "$DATASET_NAME" in
  pubmed|squirrel)
    PAPER_DEFAULT; : "${BATCH_SIZE:=32}"
    PRESET_SRC="official:paper-default-setting(ANS-GT Sec.6.1; per-dataset grid values not released)" ;;
  roman_empire|roman-empire|amazon_ratings|amazon-ratings|minesweeper|tolokers|questions|amazon_photo|photo|amazon_computers|computers|computer|coauthor_cs|cs)
    PAPER_DEFAULT; : "${BATCH_SIZE:=32}"
    PRESET_SRC="fallback:paper-default-setting" ;;
  wikics|corafull|facebook_large|facebook|github|lastfm_asia|dblp)
    # other homophilous / social graphs of the main table: the same paper default setting (these used to fall
    # into the generic defaults below, whose only difference is dropout 0.1 instead of the paper's 0.5)
    PAPER_DEFAULT; : "${BATCH_SIZE:=32}"
    PRESET_SRC="fallback:paper-default-setting" ;;
  ogbn_arxiv|ogbn-arxiv|arxiv)
    # *_ARXIV variables of the previous version of this script are still honoured
    : "${COARSEN_RATE:=${COARSEN_RATE_ARXIV:-0.01}}"
    PAPER_DEFAULT
    : "${BATCH_SIZE:=${BATCH_SIZE_ARXIV:-512}}"; : "${REWARD_EVAL_BATCHES:=${REWARD_EVAL_BATCHES_ARXIV:-32}}"
    : "${KNN_EXACT_MAX_NODES:=${KNN_EXACT_MAX_NODES_ARXIV:-0}}"
    PRESET_SRC="fallback:paper-default-setting+large-graph-batch" ;;
  arxiv_year|arxiv-year)
    PAPER_DEFAULT; : "${BATCH_SIZE:=512}"; : "${REWARD_EVAL_BATCHES:=32}"; : "${KNN_EXACT_MAX_NODES:=0}"
    PRESET_SRC="fallback:paper-default-setting+large-graph-batch" ;;
  ogbn_products|ogbn-products|products|pokec)
    PAPER_DEFAULT; : "${BATCH_SIZE:=512}"; : "${REWARD_EVAL_BATCHES:=8}"; : "${KNN_EXACT_MAX_NODES:=0}"
    : "${KNN_CANDIDATE_POOL:=2048}"
    PRESET_SRC="fallback:paper-default-setting+large-graph-batch" ;;
  *)
    # other datasets keep the generic defaults below (the values the previous version of this
    # script effectively used: its per-dataset branches were assigned after the defaults and had
    # no effect)
    PRESET_SRC="generic-defaults" ;;
esac

# Generic defaults (start.sh / main_adaptive.py), used for anything not set above.
HID_DIM="${HID_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
EVAL_EVERY="${EVAL_EVERY:-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-5.0}"

# ANS-GT sampler / hierarchy defaults from the paper, with light size-aware runtime tuning.
NUM_SAMPLED_NODES="${NUM_SAMPLED_NODES:-20}"
NUM_GLOBAL_NODES="${NUM_GLOBAL_NODES:-1}"
COARSEN_RATE="${COARSEN_RATE:-0.01}"
SAMPLER_UPDATE_PERIOD="${SAMPLER_UPDATE_PERIOD:-100}"
SAMPLER_PMIN="${SAMPLER_PMIN:-0.05}"
SAMPLER_ETA="${SAMPLER_ETA:-1.0}"
REWARD_EVAL_BATCHES="${REWARD_EVAL_BATCHES:-16}"
PPR_ALPHA="${PPR_ALPHA:-0.15}"
PPR_STEPS="${PPR_STEPS:-8}"
PPR_TOPK="${PPR_TOPK:-128}"
KNN_TOPK="${KNN_TOPK:-64}"
KNN_EXACT_MAX_NODES="${KNN_EXACT_MAX_NODES:-50000}"
KNN_CANDIDATE_POOL="${KNN_CANDIDATE_POOL:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLED_SUPERNODES="${NUM_SAMPLED_SUPERNODES:-3}"

echo "==== ANS-GT official-style launch ===="
echo "DATA_ROOT=${DATA_ROOT_EFFECTIVE}"
echo "DATASET_NAME=${DATASET_NAME} hyperparameters=${PRESET_SRC} metric=${METRIC}"
echo "DEVICE=${DEVICE}"
echo "RUNS=${RUNS} BASE_SEED=${BASE_SEED}"
echo "hid_dim=${HID_DIM} num_layers=${NUM_LAYERS} num_heads=${NUM_HEADS} dim_feedforward=${DIM_FEEDFORWARD}"
echo "dropout=${DROPOUT} lr=${LR} weight_decay=${WEIGHT_DECAY} epochs=${EPOCHS} patience=${PATIENCE}"
echo "batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} sampled_nodes=${NUM_SAMPLED_NODES} sampled_supernodes=${NUM_SAMPLED_SUPERNODES} global_nodes=${NUM_GLOBAL_NODES}"
echo "coarsen_rate=${COARSEN_RATE} sampler_update_period=${SAMPLER_UPDATE_PERIOD} reward_eval_batches=${REWARD_EVAL_BATCHES}"

"${PYTHON:-python}" ans_gt_npz.py \
  --out-dir "${OUT_DIR:-results}" \
  --data-root "${DATA_ROOT_EFFECTIVE}" \
  --dataset "${DATASET_NAME}" \
  --metric "${METRIC}" \
  --device "${DEVICE}" \
  --runs "${RUNS}" \
  --base-seed "${BASE_SEED}" \
  --epoch-timeout-seconds "${EPOCH_TIMEOUT_SECONDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --hid-dim "${HID_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --dim-feedforward "${DIM_FEEDFORWARD}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --epochs "${EPOCHS}" \
  --eval-period "${EVAL_EVERY}" \
  --patience "${PATIENCE}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --num-sampled-nodes "${NUM_SAMPLED_NODES}" \
  --num-sampled-supernodes "${NUM_SAMPLED_SUPERNODES}" \
  --num-global-nodes "${NUM_GLOBAL_NODES}" \
  --coarsen-rate "${COARSEN_RATE}" \
  --sampler-update-period "${SAMPLER_UPDATE_PERIOD}" \
  --sampler-pmin "${SAMPLER_PMIN}" \
  --sampler-eta "${SAMPLER_ETA}" \
  --reward-eval-batches "${REWARD_EVAL_BATCHES}" \
  --ppr-alpha "${PPR_ALPHA}" \
  --ppr-steps "${PPR_STEPS}" \
  --ppr-topk "${PPR_TOPK}" \
  --knn-topk "${KNN_TOPK}" \
  --knn-exact-max-nodes "${KNN_EXACT_MAX_NODES}" \
  --knn-candidate-pool "${KNN_CANDIDATE_POOL}"
