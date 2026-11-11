#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_nagphormer.sh <dataset> [cuda:0]
#   DATA_DIR=/path/to/dataset bash run_nagphormer.sh custom
#
# Default behavior:
#   MODEL=nagphormer      -> NrAug explicitly OFF
#   MODEL=nagphormer_plus -> NrAug ON; prefer using run_nagphormer_plus.sh
#
# Dataset-specific presets follow the official JHL-HUST/NAGphormer commands.txt
# when the dataset is covered there. For datasets not covered by the official
# commands, the script falls back to a nearest official-style preset.
# Metric: auto -> ROC-AUC (model selection, output and mean/std) on questions/minesweeper/tolokers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET=${1:-citeseer}

# Data layout:
#   ${DATA_ROOT}/${DATASET}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
# DATA_ROOT may be the folder holding all datasets or (as in the previous version of this script)
# the folder of this one dataset; DATA_DIR always wins.
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}
if [[ "${DATASET}" == "custom" ]]; then
  if [[ -z "${DATA_DIR:-}" ]]; then
    echo "[Error] DATASET=custom requires DATA_DIR=/path/to/dataset" >&2
    exit 1
  fi
elif [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -f "${DATA_ROOT}/adj.npz" ]]; then DATA_DIR="${DATA_ROOT}"; else DATA_DIR="${DATA_ROOT}/${DATASET}"; fi
fi
DEVICE="${2:-${DEVICE:-cuda:1}}"
MODEL="${MODEL:-nagphormer}"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/baseline_suite.py}"
RUNS="${RUNS:-5}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
OPTIMIZER="${OPTIMIZER:-adamw}"
PE_DENSE_FALLBACK_MAX_N="${PE_DENSE_FALLBACK_MAX_N:-10000}"
METRIC="${METRIC:-auto}"

# NrAug switches. For NAGphormer, keep it OFF explicitly to prevent accidental
# comparison leakage. For NAGphormer+, run_nagphormer_plus.sh sets these to ON.
# NAGphormer+ (IEEE TBD'25) only gives search ranges (mask ratio tau in {0.25,0.5,0.75}, p_aug in
# {0.25,0.5,0.75,1.0}, Beta(1,1) mixing); the defaults below lie in those ranges.
NR_AUG="${NR_AUG:-off}"
NR_AUG_PAUG="${NR_AUG_PAUG:-0.5}"
NR_AUG_TAU="${NR_AUG_TAU:-0.25}"
NR_AUG_ALPHA="${NR_AUG_ALPHA:-1.0}"
NR_AUG_BETA="${NR_AUG_BETA:-1.0}"
NR_AUG_EXTRA_ARGS=()
if [[ "${NR_AUG_USE_GNA:-1}" == "0" ]]; then
  NR_AUG_EXTRA_ARGS+=(--no-nr-aug-use-gna)
fi
if [[ "${NR_AUG_USE_LNA:-1}" == "0" ]]; then
  NR_AUG_EXTRA_ARGS+=(--no-nr-aug-use-lna)
fi

if [[ "${DATASET}" == "custom" ]]; then
  DATASET_NAME="$(basename "$DATA_DIR" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
else
  DATASET_NAME="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
fi

# Official defaults from JHL-HUST/NAGphormer commands.txt where available (all commands: dropout 0.1,
# weight decay 1e-5, batch 2000; peak lr / hops / pe_dim / hidden / layers / heads per dataset).
# Important: many datasets use n_layers=1 officially; do not silently override
# them to 2, otherwise NAGphormer gains extra capacity versus the official preset.
HID_DIM=512
NUM_LAYERS=1
NUM_HEADS=8
DROPOUT=0.1
HOPS=7
PE_DIM=10
LR=0.001
WEIGHT_DECAY=1e-5
BATCH_SIZE=2000

case "$DATASET_NAME" in
  cora)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=3;  LR=0.01;  WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  citeseer)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.3; HOPS=7;  PE_DIM=3;  LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  pubmed)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=7;  PE_DIM=15; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official:commands.txt-pubmed ;;
  corafull|cora_full)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=1; DROPOUT=0.1; HOPS=3;  PE_DIM=10; LR=0.01;  WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  computer|computers|amazon_computer|amazon_computers)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=15; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official:commands.txt-computer ;;
  photo|amazon_photo)
    HID_DIM=128; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official:commands.txt-photo ;;
  cs|coauthor_cs)
    HID_DIM=512; NUM_LAYERS=3; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official:commands.txt-cs ;;
  physics|coauthor_physics)
    HID_DIM=128; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=5;  PE_DIM=10; LR=0.01;  WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  aminer|aminer_cs)
    HID_DIM=512; NUM_LAYERS=3; NUM_HEADS=8; DROPOUT=0.1; HOPS=16; PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  reddit)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=10; PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official ;;
  amazon2m|amazon2m_npz|ogbn_products|ogbn_products_npz|products)
    # Amazon2M in the official repo is the ogbn-products graph (different split).
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=1; DROPOUT=0.1; HOPS=10; PE_DIM=3;  LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=official:commands.txt-Amazon2M ;;
  # ---- not in commands.txt: nearest official command
  ogbn_arxiv|arxiv|arxiv_year)
    # large homophilous graph -> Reddit command (arxiv_year: same graph as ogbn-arxiv)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=10; PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-reddit ;;
  wikics)
    # graph of Computer's size and density (11.7k nodes, mean degree 36.9 vs 13.8k, 35.8) -> Computer command
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=15; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-computer ;;
  facebook_large|facebook|lastfm_asia)
    # social networks with binary node attributes (people as nodes, as in Coauthor) -> CS command
    HID_DIM=512; NUM_LAYERS=3; NUM_HEADS=8; DROPOUT=0.1; HOPS=3;  PE_DIM=10; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-cs ;;
  github)
    # social network of Physics' size and density (37.7k nodes, mean degree 15.3 vs 34.5k, 14.4) -> Physics command
    HID_DIM=128; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=5;  PE_DIM=10; LR=0.01;  WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-physics ;;
  dblp)
    # citation graph of PubMed's size (17.7k nodes) -> PubMed command
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=7;  PE_DIM=15; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-pubmed ;;
  pokec)
    # 1.6M-node graph -> command of the largest graph (Amazon2M)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=1; DROPOUT=0.1; HOPS=10; PE_DIM=3;  LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-Amazon2M ;;
  roman_empire|amazon_ratings|minesweeper|tolokers|questions|squirrel)
    # heterophilous 5k-49k-node graphs: no heterophilous command exists -> PubMed command (medium graph,
    # 7 hops = the longest-range medium-graph command)
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=7;  PE_DIM=15; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=fallback:commands.txt-pubmed ;;
  *)
    # Conservative fallback: nearest official-style configuration.
    HID_DIM=512; NUM_LAYERS=1; NUM_HEADS=8; DROPOUT=0.1; HOPS=7; PE_DIM=8; LR=0.001; WEIGHT_DECAY=1e-5; BATCH_SIZE=2000; SRC=generic-defaults ;;
esac

# Optional env overrides for controlled ablations.
HID_DIM="${HID_DIM_OVERRIDE:-$HID_DIM}"
NUM_LAYERS="${NUM_LAYERS_OVERRIDE:-$NUM_LAYERS}"
NUM_HEADS="${NUM_HEADS_OVERRIDE:-$NUM_HEADS}"
DROPOUT="${DROPOUT_OVERRIDE:-$DROPOUT}"
HOPS="${HOPS_OVERRIDE:-$HOPS}"
PE_DIM="${PE_DIM_OVERRIDE:-$PE_DIM}"
LR="${LR_OVERRIDE:-$LR}"
WEIGHT_DECAY="${WEIGHT_DECAY_OVERRIDE:-$WEIGHT_DECAY}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"

# In the official model.py, the effective FFN size is forced to 2 * hidden_dim.
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-$((2 * HID_DIM))}"


echo "==== NAGphormer official-style launcher ===="
echo "data_dir=$DATA_DIR"
echo "dataset_name=$DATASET_NAME hyperparameters=$SRC metric=$METRIC"
echo "model=$MODEL nr_aug=$NR_AUG paug=$NR_AUG_PAUG tau=$NR_AUG_TAU alpha=$NR_AUG_ALPHA beta=$NR_AUG_BETA"
echo "device=$DEVICE"
echo "runs=$RUNS seed=$SEED epochs=$EPOCHS patience=$PATIENCE optimizer=$OPTIMIZER"
echo "hid_dim=$HID_DIM num_layers=$NUM_LAYERS num_heads=$NUM_HEADS dim_feedforward=$DIM_FEEDFORWARD"
echo "dropout=$DROPOUT hops=$HOPS pe_dim=$PE_DIM batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE"
echo "lr=$LR weight_decay=$WEIGHT_DECAY"

"${PYTHON:-python}" "$PY_SCRIPT" \
  --out-dir "${OUT_DIR:-results}" \
  --data-root "$DATA_DIR" \
  --dataset "$DATASET_NAME" \
  --metric "$METRIC" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --eval-every "$EVAL_EVERY" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --dropout "$DROPOUT" \
  --hid-dim "$HID_DIM" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --dim-feedforward "$DIM_FEEDFORWARD" \
  --hops "$HOPS" \
  --pe-dim "$PE_DIM" \
  --pe-dense-fallback-max-n "$PE_DENSE_FALLBACK_MAX_N" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --nr-aug "$NR_AUG" \
  --nr-aug-paug "$NR_AUG_PAUG" \
  --nr-aug-tau "$NR_AUG_TAU" \
  --nr-aug-alpha "$NR_AUG_ALPHA" \
  --nr-aug-beta "$NR_AUG_BETA" \
  "${NR_AUG_EXTRA_ARGS[@]}" \
  --keep-existing-split
