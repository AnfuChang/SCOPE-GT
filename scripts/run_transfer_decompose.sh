#!/usr/bin/env bash
# Table IX decomposition on LastFM: FQ-LG, CQ-U, and optional CQ-E.
# Native and +SCMB are the existing Table IX runs and are not repeated.
#   DATA_BASE=/path/to/datasets bash scripts/run_transfer_decompose.sh
# Environment (in addition to scripts/_common.sh):
#   METHODS          backbones, run one after another (default: "nlgt gophormer hubgt swapgt")
#   OUT              result root; each backbone writes to OUT/<method> (default: results/transfer_decompose)
#   REFERENCE        existing Table IX root (run_transfer.sh OUT) for pairing checks (default: results/transfer if present)
#   V1_PAD_MULTIPLE  V1 length-group granularity, 1 = exact lengths and zero padding (default: 1)
#   SEEDS            paired training seeds (default: "0 1 2 3 4", matching the Table IX reference launcher)
# V1 is skipped for HubGT/SwapGT: their fixed-length inputs contain no padding, so V1 equals Native.
# Extra arguments are passed to SCOPE_GT_transfer.decompose_run (e.g. --dry-run, --overrides FILE).
export SEEDS="${SEEDS:-0 1 2 3 4}"  # must precede _common.sh, which reads SEEDS
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
SCOPE_GT_BASELINES_ROOT="${SCOPE_GT_BASELINES_ROOT:-${ROOT}/Baselines}"
export SCOPE_GT_BASELINES_ROOT

read -r -a METHOD_LIST <<< "${METHODS:-nlgt gophormer hubgt swapgt}"
OUT_BASE="${OUT:-${ROOT}/results/transfer_decompose}"
V1_PAD_MULTIPLE="${V1_PAD_MULTIPLE:-1}"
REFERENCE="${REFERENCE:-${ROOT}/results/transfer}"
REFERENCE_ARGS=()
if [[ -f "${REFERENCE}/plan.json" ]]; then REFERENCE_ARGS=(--reference "${REFERENCE}"); fi

if [[ " ${METHOD_LIST[*]} " == *" hubgt "* && " $* " != *" --dry-run "* && " $* " != *" --help "* ]]; then
  "${PYTHON}" -m SCOPE_GT_transfer.build_hubgt
fi

failed=()
for method in "${METHOD_LIST[@]}"; do
  echo "[$(date '+%F %T')] ${method}: start -> ${OUT_BASE}/${method}"
  if "${PYTHON}" -u -m SCOPE_GT_transfer.decompose_run \
      --data-base "${DATA_BASE}" --out "${OUT_BASE}/${method}" \
      --datasets lastfm_asia --methods "${method}" --variants native_packed cq_noshare cq_sharedenc \
      --seeds "${SEED_LIST[@]}" --device "${INDEXED_DEVICE}" --threads "${THREADS}" \
      --v1-pad-multiple "${V1_PAD_MULTIPLE}" ${REFERENCE_ARGS[@]+"${REFERENCE_ARGS[@]}"} "$@"; then
    echo "[$(date '+%F %T')] ${method}: done"
  else
    echo "[$(date '+%F %T')] ${method}: FAILED (see ${OUT_BASE}/${method}/failures.csv)"; failed+=("${method}")
  fi
done
if (( ${#failed[@]} )); then echo "Failed backbones: ${failed[*]}"; exit 1; fi
