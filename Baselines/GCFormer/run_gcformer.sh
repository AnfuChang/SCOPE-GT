#!/usr/bin/env bash
# GCFormer (NeurIPS'24) under the shared comparison protocol.
#   bash run_gcformer.sh <dataset>                  e.g. bash run_gcformer.sh roman_empire
#   RUNS=10 SPLIT_SUFFIXES="_0 _1 _2 _3 _4 _5 _6 _7 _8 _9" bash run_gcformer.sh roman_empire   (10 official splits)
#   LR=1e-3 DROPOUT=0.3 bash run_gcformer.sh cora   (any hyperparameter below can be overridden by name)
set -euo pipefail

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Fixed by project comparison protocol (identical in every run_*.sh, see ../common/protocol.py)
RUNS="${RUNS:-5}"                      # seeds BASE_SEED .. BASE_SEED+RUNS-1
BASE_SEED="${BASE_SEED:-42}"
EPOCHS="${EPOCHS:-1000}"               # upper bound on epochs
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
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

# user overrides (environment) win over the presets below
TUNABLE=(SAMPLE_NUM_P SAMPLE_NUM_N HIDDEN FFN_DIM N_HEADS N_LAYERS DROPOUT ATT_DROPOUT ALPHA BETA PP_K TEMPERATURE
         LR END_LR WEIGHT_DECAY WARMUP TOT_UPDATES BATCH_SIZE)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- GCFormer hyperparameters: official Solo.sh where the repo has a command for the dataset (the
#      paper's 8 datasets: Photo, ACM, Computer, CoraFull, BlogCatalog, UAI2010, Flickr, RomanEmpire),
#      otherwise the closest official command (tune on validation if the budget allows).
#      14 ACE_GTv6 datasets: official = amazon_photo, amazon_computers, roman_empire;
#      fallback:romanempire = amazon_ratings, minesweeper, tolokers, questions, squirrel, pokec, arxiv_year;
#      fallback:photo = pubmed, coauthor_cs, ogbn_arxiv, ogbn_products
FFN_DIM=64; N_HEADS=1; N_LAYERS=1; WEIGHT_DECAY=1e-5; END_LR=1e-4; WARMUP=400; TOT_UPDATES=1000
case "${KEY}" in
  photo|amazon_photo)                        # Solo.sh: photo
    SAMPLE_NUM_P=3; SAMPLE_NUM_N=5; HIDDEN=128; DROPOUT=0.5; ATT_DROPOUT=0.1; ALPHA=0.3; BETA=0.1
    PP_K=2; TEMPERATURE=5; LR=0.0005; BATCH_SIZE=5000; SRC=official:Solo.sh-photo ;;
  computer|computers|amazon_computers)       # Solo.sh: computer
    SAMPLE_NUM_P=3; SAMPLE_NUM_N=5; HIDDEN=128; DROPOUT=0.1; ATT_DROPOUT=0.1; ALPHA=0.3; BETA=0.1
    PP_K=2; TEMPERATURE=0.5; LR=0.0005; BATCH_SIZE=5000; SRC=official:Solo.sh-computer ;;
  corafull)                                  # Solo.sh: corafull
    SAMPLE_NUM_P=7; SAMPLE_NUM_N=7; HIDDEN=512; DROPOUT=0.3; ATT_DROPOUT=0.5; ALPHA=0.25; BETA=0.01
    PP_K=3; TEMPERATURE=3; LR=0.0005; BATCH_SIZE=2048; SRC=official:Solo.sh-corafull ;;
  roman_empire|romanempire)                  # Solo.sh: romanempire
    SAMPLE_NUM_P=7; SAMPLE_NUM_N=7; HIDDEN=128; DROPOUT=0.5; ATT_DROPOUT=0.5; ALPHA=0.4; BETA=0.01
    PP_K=1; TEMPERATURE=3; LR=0.001; BATCH_SIZE=2048; SRC=official:Solo.sh-romanempire ;;
  amazon_ratings|minesweeper|tolokers|questions|squirrel|chameleon|squirrel_filtered|chameleon_filtered|pokec|arxiv_year)
    # heterophilous graph without an official command -> romanempire command (the paper's only
    # heterophilous benchmark of the Platonov et al. suite)
    SAMPLE_NUM_P=7; SAMPLE_NUM_N=7; HIDDEN=128; DROPOUT=0.5; ATT_DROPOUT=0.5; ALPHA=0.4; BETA=0.01
    PP_K=1; TEMPERATURE=3; LR=0.001; BATCH_SIZE=2048; SRC=fallback:romanempire ;;
  pubmed|coauthor_cs|cs|ogbn_arxiv|arxiv|ogbn_products|products|lastfm_asia|dblp)
    # homophilous graph without an official command -> photo command (lastfm_asia: as coauthor_cs; dblp: as pubmed)
    SAMPLE_NUM_P=3; SAMPLE_NUM_N=5; HIDDEN=128; DROPOUT=0.5; ATT_DROPOUT=0.1; ALPHA=0.3; BETA=0.1
    PP_K=2; TEMPERATURE=5; LR=0.0005; BATCH_SIZE=5000; SRC=fallback:photo ;;
  *)
    # other homophilous graphs without an official command (cora, citeseer, wikics, physics, ...) -> photo
    SAMPLE_NUM_P=3; SAMPLE_NUM_N=5; HIDDEN=128; DROPOUT=0.5; ATT_DROPOUT=0.1; ALPHA=0.3; BETA=0.1
    PP_K=2; TEMPERATURE=5; LR=0.0005; BATCH_SIZE=5000; SRC=fallback:photo ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done

echo "==== GCFormer | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/gcformer_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --sample-num-p "${SAMPLE_NUM_P}" --sample-num-n "${SAMPLE_NUM_N}" --hidden-dim "${HIDDEN}" --ffn-dim "${FFN_DIM}"
  --n-layers "${N_LAYERS}" --n-heads "${N_HEADS}" --dropout "${DROPOUT}" --attention-dropout "${ATT_DROPOUT}"
  --alpha "${ALPHA}" --beta "${BETA}" --pp-k "${PP_K}" --temp "${TEMPERATURE}" --batch-size "${BATCH_SIZE}"
  --warmup-updates "${WARMUP}" --tot-updates "${TOT_UPDATES}" --peak-lr "${LR}" --end-lr "${END_LR}"
  --weight-decay "${WEIGHT_DECAY}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
