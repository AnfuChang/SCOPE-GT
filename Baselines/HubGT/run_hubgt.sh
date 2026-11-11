#!/usr/bin/env bash
# HubGT (NeurIPS'25) under the shared comparison protocol.
#   bash run_hubgt.sh <dataset>                      e.g. bash run_hubgt.sh tolokers
#   RUNS=10 SPLIT_SUFFIXES="_0 _1 _2 _3 _4 _5 _6 _7 _8 _9" bash run_hubgt.sh roman_empire   (10 official splits)
#   S0=18 S1=16 S0G=6 bash run_hubgt.sh cora          (any hyperparameter below can be overridden by name)
# First use: the C++ hub-labeling index is compiled (needs Cython>=3.0 and g++ with OpenMP).
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
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/${DATASET}_$(date +%Y%m%d_%H%M%S)}"

TUNABLE=(N_LAYERS NUM_HEADS HIDDEN FFN_DIM DP_INPUT DP_BIAS DP_FFN DP_ATTN AGGR_OUTPUT VAR_VFEAT
         WEIGHT_DECAY LR END_LR SCHED_EPOCHS BATCH NUM_WORKERS SS S0 S0G S1)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- HubGT hyperparameters: defaults of utils/config.py with --epoch 300 from scripts/run.sh.
#      The paper (App. "Hyperparameters") uses one architecture for all experiments -- 4 layers, 8 heads,
#      hidden 128, dropout 0.1 (input, bias) / 0.5 (intermediate), AdamW, 300 epochs -- and tunes only
#      the subgraph composition (s0, s1, s0g, aggr_output, var_vfeat) per dataset with optuna
#      (main_tune.py); the tuned values are not released, so the defaults are used everywhere.
#      (The paper text states lr 1e-4; the released code, used here, has peak_lr 2e-4 -> 1e-9.)
N_LAYERS=4; NUM_HEADS=8; HIDDEN=128; FFN_DIM=128; DP_INPUT=0.1; DP_BIAS=0.1; DP_FFN=0.5; DP_ATTN=0.5
AGGR_OUTPUT=0; VAR_VFEAT=1; WEIGHT_DECAY=0.01; LR=2e-4; END_LR=1e-9; SCHED_EPOCHS=300
BATCH=1024; NUM_WORKERS=4; SS=48; S0=24; S0G=8; S1=12
case "$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')" in
  # datasets of the HubGT paper (main: ogbn-arxiv, pokec; appendix: pubmed, cs, tolokers)
  ogbn_arxiv|arxiv|pokec|pubmed|coauthor_cs|cs|tolokers) SRC="official:shared-config(paper+config.py)" ;;
  # the paper uses the filtered squirrel (Platonov et al.); ours is the original 5,201-node graph
  squirrel) SRC="fallback:shared-config(paper-used-squirrel_filtered)" ;;
  # other main-table datasets: the same shared configuration (the paper releases no other)
  wikics|corafull|facebook_large|facebook|github|amazon_computers|computers|roman_empire|amazon_ratings|minesweeper \
  |arxiv_year|ogbn_products|products|lastfm_asia|dblp) SRC="fallback:shared-config" ;;
  *) SRC="fallback:shared-config" ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done

# ---- compile the official Cython/C++ extension. The decision does not rely on file times (copying the
#      sources to the server can keep old modification times): the compiled module is imported and must
#      report the ACE fix version of Precompute/ppll.h (PPLL_ACE_FIX_VERSION); an older .so segfaults.
REQUIRED_FIX_VERSION=2
# extension_ok [verbose]: import the compiled module from Precompute/ and check its ACE fix version.
#   quiet (default) only answers yes/no; verbose prints the loaded .so, its version, or the import error.
extension_ok() {
  local out status
  out="$(cd "${SCRIPT_DIR}/Precompute" && "${PYTHON}" -c "
import sys; sys.path.insert(0, '.')
import kpll_cppext as m
v = getattr(m, 'ace_fix_version', lambda: 0)()
print('[HubGT] kpll_cppext: %s (ACE fix version %s, need ${REQUIRED_FIX_VERSION})' % (m.__file__, v))
sys.exit(0 if v >= ${REQUIRED_FIX_VERSION} else 3)" 2>&1)" && status=0 || status=$?
  if [[ "${1:-}" == "verbose" ]]; then
    echo "${out}"
  fi
  return "${status}"
}
build_extension() {
  if extension_ok; then return 0; fi    # re-checked under the lock: a parallel run may have just built it
  echo "[HubGT] building Precompute/kpll_cppext (stale or missing build; python setup.py build_ext --inplace --force) ..."
  echo "[HubGT] compiler warnings (-Wsign-compare, deprecated NumPy API) come from the official -Wall -Wextra flags and are harmless"
  if ! (cd "${SCRIPT_DIR}/Precompute" && rm -rf build kpll_cppext.cpp kpll_cppext*.so         && "${PYTHON}" setup.py build_ext --inplace --force); then
    echo "[HubGT] ERROR: compiling Precompute/kpll_cppext failed (see the compiler messages above, lines with 'error:')" >&2
    exit 1
  fi
  echo "[HubGT] build finished; checking the new module ..."
  if ! extension_ok verbose; then
    echo "[HubGT] ERROR: the rebuilt Precompute/kpll_cppext cannot be imported or does not report ACE fix version"          "${REQUIRED_FIX_VERSION} (message above). If the version is 0, Precompute/ppll.h / kpll_cppext.pyx / .pxd on"          "this machine are not the patched files." >&2
    exit 1
  fi
}
if ! extension_ok; then
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${SCRIPT_DIR}/Precompute/.build.lock"
    flock 9
    build_extension
    flock -u 9
    exec 9>&-
  else
    build_extension
  fi
else
  extension_ok verbose
fi
export PYTHONUNBUFFERED=1   # keep the log complete even if a worker or the C++ extension crashes

echo "==== HubGT | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/hubgt_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --n-layers "${N_LAYERS}" --num-heads "${NUM_HEADS}" --hidden-dim "${HIDDEN}" --ffn-dim "${FFN_DIM}"
  --dp-input "${DP_INPUT}" --dp-bias "${DP_BIAS}" --dp-ffn "${DP_FFN}" --dp-attn "${DP_ATTN}"
  --aggr-output "${AGGR_OUTPUT}" --var-vfeat "${VAR_VFEAT}" --weight-decay "${WEIGHT_DECAY}"
  --peak-lr "${LR}" --end-lr "${END_LR}" --sched-epochs "${SCHED_EPOCHS}" --batch "${BATCH}"
  --num-workers "${NUM_WORKERS}" --ss "${SS}" --s0 "${S0}" --s0g "${S0G}" --s1 "${S1}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
if [[ "${REBUILD_INDEX:-0}" == "1" ]]; then CMD+=(--rebuild-index); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
