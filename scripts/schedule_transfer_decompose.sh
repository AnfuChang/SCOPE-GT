#!/usr/bin/env bash
# Scheduled two-GPU launch of the Table IX decomposition (V1/V2/V3, five seeds, LastFM).
# Waits until START_AT, then starts one queue per GPU; each queue runs its backbones one after
# another through scripts/run_transfer_decompose.sh. A summary is merged when both queues end.
#
#   cd /path/to/SCOPE-GT
#   export DATA_BASE=/path/to/datasets
#   nohup bash scripts/schedule_transfer_decompose.sh > logs/schedule_decompose.log 2>&1 &
#
# Environment:
#   START_AT    wall-clock start, any `date -d` time (default 01:00 = the next 1 a.m.); "now" starts at once
#   GPU_A/B     GPU indices as listed by nvidia-smi (default 0 and 1)
#   METHODS_A/B backbones of each queue, run serially (default "hubgt nlgt" and "gophormer swapgt";
#               balanced by the Table IX total times, about 6 h per queue for one seed)
#   SEEDS       default "0 1 2 3 4", matching the Table IX reference launcher
#   PRECHECK    1 (default): run the equivalence self-check now, before waiting; abort on failure
#   PYTHON      interpreter of the environment used for the original Table IX runs (default: python)
#   OUT, REFERENCE, THREADS, V1_PAD_MULTIPLE: passed through to run_transfer_decompose.sh
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "${ROOT}"
: "${DATA_BASE:?Set DATA_BASE to the processed dataset root}"
SCOPE_GT_BASELINES_ROOT="${SCOPE_GT_BASELINES_ROOT:-${ROOT}/Baselines}"
export DATA_BASE SCOPE_GT_BASELINES_ROOT
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 PYTHONUTF8=1
export SEEDS="${SEEDS:-0 1 2 3 4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make CUDA indices match nvidia-smi
PYTHON="${PYTHON:-python}"; export PYTHON
START_AT="${START_AT:-01:00}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"
METHODS_A="${METHODS_A:-hubgt nlgt}"; METHODS_B="${METHODS_B:-gophormer swapgt}"
OUT_BASE="${OUT:-${ROOT}/results/transfer_decompose}"; export OUT="${OUT_BASE}"
REFERENCE="${REFERENCE:-${ROOT}/results/transfer}"; export REFERENCE
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
PRECHECK="${PRECHECK:-1}"
mkdir -p "${LOG_DIR}"
stamp() { date '+%F %T'; }

echo "[$(stamp)] python: $(command -v "${PYTHON}")"
echo "[$(stamp)] GPU ${GPU_A}: ${METHODS_A} | GPU ${GPU_B}: ${METHODS_B} | seeds: ${SEEDS} | out: ${OUT_BASE}"
if [[ -f "${REFERENCE}/plan.json" ]]; then
  echo "[$(stamp)] reference Table IX runs: ${REFERENCE}"
else
  echo "[$(stamp)] WARNING: no ${REFERENCE}/plan.json; pairing checks against Native/+SCMB will be skipped"
fi

if [[ " ${METHODS_A} ${METHODS_B} " == *" hubgt "* ]]; then
  "${PYTHON}" -m SCOPE_GT_transfer.build_hubgt  # compile/check once, before the queues start
fi
if [[ "${PRECHECK}" == 1 ]]; then
  echo "[$(stamp)] equivalence self-check on GPU ${GPU_A}"
  read -r -a CHECK_SEEDS <<< "${SEEDS}"
  CUDA_VISIBLE_DEVICES="${GPU_A}" "${PYTHON}" -m SCOPE_GT_transfer.decompose_check \
    --data-base "${DATA_BASE}" --device cuda:0 --seed "${CHECK_SEEDS[0]}" --methods ${METHODS_A} ${METHODS_B}
fi

if [[ "${START_AT}" != now ]]; then
  target=$(date -d "${START_AT}" +%s)
  if (( target <= $(date +%s) )); then target=$(date -d "tomorrow ${START_AT}" +%s); fi
  echo "[$(stamp)] waiting until $(date -d "@${target}" '+%F %T')"
  while (( $(date +%s) < target )); do sleep $(( target - $(date +%s) > 60 ? 60 : target - $(date +%s) )); done
fi

echo "[$(stamp)] starting both queues"
CUDA_VISIBLE_DEVICES="${GPU_A}" METHODS="${METHODS_A}" bash scripts/run_transfer_decompose.sh \
  > "${LOG_DIR}/decompose_gpu${GPU_A}.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES="${GPU_B}" METHODS="${METHODS_B}" bash scripts/run_transfer_decompose.sh \
  > "${LOG_DIR}/decompose_gpu${GPU_B}.log" 2>&1 &
pid_b=$!
echo "[$(stamp)] GPU ${GPU_A} queue pid ${pid_a} -> ${LOG_DIR}/decompose_gpu${GPU_A}.log"
echo "[$(stamp)] GPU ${GPU_B} queue pid ${pid_b} -> ${LOG_DIR}/decompose_gpu${GPU_B}.log"
status_a=0; status_b=0
wait "${pid_a}" || status_a=$?
echo "[$(stamp)] GPU ${GPU_A} queue finished (exit ${status_a})"
wait "${pid_b}" || status_b=$?
echo "[$(stamp)] GPU ${GPU_B} queue finished (exit ${status_b})"

results=()
for method in ${METHODS_A} ${METHODS_B}; do
  if [[ -f "${OUT_BASE}/${method}/plan.json" ]]; then results+=("${OUT_BASE}/${method}"); fi
done
if (( ${#results[@]} )); then
  reference_args=()
  if [[ -f "${REFERENCE}/plan.json" ]]; then reference_args=(--reference "${REFERENCE}"); fi
  "${PYTHON}" -m SCOPE_GT_transfer.decompose_report --results "${results[@]}" \
    ${reference_args[@]+"${reference_args[@]}"} --out "${OUT_BASE}/summary" \
    || echo "[$(stamp)] summary failed; rerun decompose_report manually"
fi
echo "[$(stamp)] all done (GPU ${GPU_A}: ${status_a}, GPU ${GPU_B}: ${status_b})"
(( status_a == 0 && status_b == 0 ))
