#!/usr/bin/env bash
# SwapGT (NeurIPS'25) under the shared comparison protocol.
#   bash run_swapgt.sh <dataset>                     e.g. bash run_swapgt.sh wikics
#   RUNS=10 SPLIT_SUFFIXES="_0 _1 _2 _3 _4 _5 _6 _7 _8 _9" bash run_swapgt.sh roman_empire   (10 official splits)
#   LR=5e-4 NUM_AUG=3 bash run_swapgt.sh cora        (any hyperparameter below can be overridden by name)
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

TUNABLE=(HIDDEN FFN_DIM N_LAYERS N_HEADS DROPOUT ATT_DROPOUT PP_K PROP_ALPHA NUM_AUG DEPTH SAMPLE_NUM
         LAMBDA1 LAMBDA2 LR END_LR WEIGHT_DECAY WARMUP TOT_UPDATES BATCH_SIZE)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- SwapGT hyperparameters: cmd.sh "dense" commands (50/25/25 split, the closer setting to ours);
#      datasets without a command use the closest official command (tune on validation if possible).
#      The paper's datasets are Photo, ACM, Computer, Citeseer, WikiCS, BlogCatalog, UAI2010, Flickr.
#      14 ACE_GTv6 datasets: official = amazon_photo, amazon_computers;
#      fallback:uai = roman_empire, amazon_ratings, minesweeper, tolokers, questions, squirrel, pokec, arxiv_year;
#      fallback:photo = pubmed, coauthor_cs, ogbn_arxiv, ogbn_products
HIDDEN=512; FFN_DIM=64; N_LAYERS=1; N_HEADS=8; DEPTH=2; PROP_ALPHA=0.1
WEIGHT_DECAY=1e-5; END_LR=1e-4; WARMUP=400; TOT_UPDATES=1000; BATCH_SIZE=2048
case "${KEY}" in
  wikics)                                    # cmd.sh dense: wikics
    DROPOUT=0.1; ATT_DROPOUT=0.1; LR=0.001; PP_K=2; SAMPLE_NUM=10; NUM_AUG=2; LAMBDA1=0.3; LAMBDA2=0.3; SRC=official:cmd.sh-wikics ;;
  photo|amazon_photo)                        # cmd.sh dense: photo
    DROPOUT=0.1; ATT_DROPOUT=0.1; LR=0.001; PP_K=3; SAMPLE_NUM=4; NUM_AUG=2; LAMBDA1=10; LAMBDA2=0.1; SRC=official:cmd.sh-photo ;;
  computer|computers|amazon_computers)       # cmd.sh dense: computer
    DROPOUT=0.1; ATT_DROPOUT=0.5; LR=0.0005; PP_K=4; SAMPLE_NUM=6; NUM_AUG=3; LAMBDA1=20; LAMBDA2=0.1; SRC=official:cmd.sh-computer ;;
  citeseer)                                  # cmd.sh dense: citeseer
    DROPOUT=0.1; ATT_DROPOUT=0.1; LR=0.001; PP_K=3; SAMPLE_NUM=10; NUM_AUG=2; LAMBDA1=1; LAMBDA2=0.6; SRC=official:cmd.sh-citeseer ;;
  roman_empire|amazon_ratings|minesweeper|tolokers|questions|squirrel|chameleon|squirrel_filtered|chameleon_filtered|pokec|arxiv_year)
    # heterophilous graph without a command -> uai (dense), the least homophilous dataset with a regular lr
    DROPOUT=0.1; ATT_DROPOUT=0.5; LR=0.001; PP_K=2; SAMPLE_NUM=8; NUM_AUG=2; LAMBDA1=0.6; LAMBDA2=0.2; SRC=fallback:uai ;;
  pubmed|coauthor_cs|cs|ogbn_arxiv|arxiv|ogbn_products|products|lastfm_asia|dblp)
    # homophilous graph without a command -> photo (dense) (lastfm_asia: as coauthor_cs; dblp: as pubmed)
    DROPOUT=0.1; ATT_DROPOUT=0.1; LR=0.001; PP_K=3; SAMPLE_NUM=4; NUM_AUG=2; LAMBDA1=10; LAMBDA2=0.1; SRC=fallback:photo ;;
  *)
    # other homophilous graphs without a command (cora, corafull, physics, facebook, ...) -> photo (dense)
    DROPOUT=0.1; ATT_DROPOUT=0.1; LR=0.001; PP_K=3; SAMPLE_NUM=4; NUM_AUG=2; LAMBDA1=10; LAMBDA2=0.1; SRC=fallback:photo ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done

echo "==== SwapGT | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/swapgt_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --hidden "${HIDDEN}" --ffn-dim "${FFN_DIM}" --n-layers "${N_LAYERS}" --n-heads "${N_HEADS}"
  --dropout "${DROPOUT}" --attention-dropout "${ATT_DROPOUT}" --alpha "${PROP_ALPHA}" --pp-k "${PP_K}"
  --num-aug "${NUM_AUG}" --depth "${DEPTH}" --sample-num "${SAMPLE_NUM}" --lambda1 "${LAMBDA1}" --lambda2 "${LAMBDA2}"
  --batch-size "${BATCH_SIZE}" --warmup-updates "${WARMUP}" --tot-updates "${TOT_UPDATES}"
  --peak-lr "${LR}" --end-lr "${END_LR}" --weight-decay "${WEIGHT_DECAY}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
