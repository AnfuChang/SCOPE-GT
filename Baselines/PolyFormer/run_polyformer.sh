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

if [[ -z "${OUT_ROOT:-}" ]]; then
  OUT_ROOT="$(mktemp -d -t polyformer_node_${DATASET}_XXXXXX)"
  trap 'rm -rf "${OUT_ROOT}"' EXIT
fi
DEVICE=${DEVICE:-cuda:1}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# BASE=auto uses the official mono config when available. Set BASE=cheb to use
# the official Chebyshev config for Table-5 datasets.
# Hyperparameter source per dataset (polyformer_node/configs.py; official = node_classification.sh /
# node_classification_large_graph.sh of the PolyFormer repo):
#   official : roman_empire, minesweeper, tolokers, questions, coauthor_cs, pubmed (mono + cheb),
#              pokec, ogbn_arxiv (mono)
#   fallback : squirrel -> squirrel_filtered (PolyFormer only uses the filtered squirrel),
#              amazon_ratings -> roman_empire, arxiv_year -> ogbn_arxiv (same graph),
#              amazon_photo / amazon_computers / ogbn_products -> self-set configs in configs.py FALLBACK,
#              dblp / corafull -> pubmed (official), lastfm_asia -> coauthor_cs (official)
# minesweeper / tolokers / questions are trained as binary tasks and selected/reported with ROC-AUC.
BASE=${BASE:-auto}
eval "$(${PYTHON} -m polyformer_node.configs --dataset "${DATASET}" --base "${BASE}")"

# Repeated runs.
RUNS=${RUNS:-5}
SEED=${SEED:-42}
SEED_STEP=${SEED_STEP:-1}

# Hyperparameters: environment variables override official/self-set defaults.
BASE=${BASE_OVERRIDE:-${CFG_BASE:-mono}}
EPOCHS=${EPOCHS:-1000}
EVAL_EVERY=${EVAL_EVERY:-5}
PATIENCE=${PATIENCE:-50}
LR=${LR:-${CFG_LR:-0.0005}}
WEIGHT_DECAY=${WEIGHT_DECAY:-${CFG_WEIGHT_DECAY:-0.00005}}
ATTN_LR=${ATTN_LR:-${CFG_ATTN_LR:-0.0005}}
ATTN_WD=${ATTN_WD:-${CFG_ATTN_WD:-0.0001}}
HIDDEN=${HIDDEN:-${CFG_HIDDEN:-64}}
DROPOUT=${DROPOUT:-${CFG_DROPOUT:-0.2}}
DPRATE=${DPRATE:-${CFG_DPRATE:-0.5}}
N_HEAD=${N_HEAD:-${CFG_N_HEAD:-1}}
D_FFN=${D_FFN:-${CFG_D_FFN:-128}}
Q=${Q:-${CFG_Q:-1.0}}
MULTI=${MULTI:-${CFG_MULTI:-1.0}}
K=${K:-${CFG_K:-10}}
NLAYER=${NLAYER:-${CFG_NLAYER:-1}}
METRIC=${METRIC:-${CFG_METRIC:-auto}}
FORCE_BINARY=${FORCE_BINARY:-${CFG_FORCE_BINARY:-0}}
CONFIG_SOURCE=${CFG_CONFIG_SOURCE:-unknown}

# Graph preprocessing.
TO_UNDIRECTED=${TO_UNDIRECTED:-1}
FEATURE_NORM=${FEATURE_NORM:-none}

# Polynomial basis cache. For large datasets, use CACHE_MODE=memmap and put
# CACHE_DIR on a large disk, e.g. /mnt/sdb/changaf25/polyformer_cache.
CACHE_DIR=${CACHE_DIR:-}
CACHE_MODE=${CACHE_MODE:-memmap}
FORCE_RECOMPUTE_BASES=${FORCE_RECOMPUTE_BASES:-0}

# Training mode. PolyFormer applies attention within each node's polynomial-token
# sequence after bases are built, so batch mode is an exact node-wise batching of
# the same model and is much more memory friendly.
TRAIN_MODE=${TRAIN_MODE:-batch}
BATCH_SIZE=${BATCH_SIZE:-${CFG_BATCH_SIZE:-10000}}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-50000}
GRAD_CLIP=${GRAD_CLIP:-0.0}
SAVE_BEST=${SAVE_BEST:-0}

mkdir -p "${OUT_ROOT}"

echo "[PolyFormer] dataset=${DATASET} data_dir=${DATA_DIR} out_root=${OUT_ROOT} runs=${RUNS} seed_start=${SEED}"
echo "[PolyFormer] config_source=${CONFIG_SOURCE} base=${BASE} K=${K} hidden=${HIDDEN} n_head=${N_HEAD} nlayer=${NLAYER} metric=${METRIC}"
echo "[PolyFormer] train_mode=${TRAIN_MODE} batch_size=${BATCH_SIZE} cache_dir=${CACHE_DIR} cache_mode=${CACHE_MODE}"

for ((i=0; i<RUNS; i++)); do
  CUR_SEED=$((SEED + i * SEED_STEP))
  RUN_OUT="${OUT_ROOT}/seed_${CUR_SEED}"
  echo "========== PolyFormer run $((i+1))/${RUNS}; seed=${CUR_SEED}; out=${RUN_OUT} =========="
  ${PYTHON} -m polyformer_node.train_polyformer \
    --data_dir "${DATA_DIR}" \
    --dataset "${DATASET}" \
    --out_dir "${RUN_OUT}" \
    --seed "${CUR_SEED}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --eval_every "${EVAL_EVERY}" \
    --patience "${PATIENCE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --attn_lr "${ATTN_LR}" \
    --attn_wd "${ATTN_WD}" \
    --hidden "${HIDDEN}" \
    --dropout "${DROPOUT}" \
    --dprate "${DPRATE}" \
    --n_head "${N_HEAD}" \
    --d_ffn "${D_FFN}" \
    --q "${Q}" \
    --multi "${MULTI}" \
    --K "${K}" \
    --nlayer "${NLAYER}" \
    --base "${BASE}" \
    --metric "${METRIC}" \
    --force_binary "${FORCE_BINARY}" \
    --to_undirected "${TO_UNDIRECTED}" \
    --feature_norm "${FEATURE_NORM}" \
    --cache_dir "${CACHE_DIR}" \
    --cache_mode "${CACHE_MODE}" \
    --force_recompute_bases "${FORCE_RECOMPUTE_BASES}" \
    --train_mode "${TRAIN_MODE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --grad_clip "${GRAD_CLIP}" \
    --save_best "${SAVE_BEST}" \
    --config_source "${CONFIG_SOURCE}"
done

echo "========== Aggregate PolyFormer repeated runs =========="
${PYTHON} -m polyformer_node.aggregate_repeats \
  --root "${OUT_ROOT}" \
  --runs "${RUNS}" \
  --seed "${SEED}" \
  --seed_step "${SEED_STEP}"
