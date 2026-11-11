#!/usr/bin/env bash
# Fig. 5: context budget N_ctx and walk budget W on WikiCS and DBLP, varied one at a time.
#   DATA_BASE=/path/to/datasets bash scripts/run_budget.sh
# N_ctx sweep: 8/16/24/32/48/64 with W fixed (48 on WikiCS, 24 on DBLP).
# W sweep:     12/24/36/48/72/96 with N_ctx fixed (32 on WikiCS, 16 on DBLP).
# Every point is trained from scratch, selected on validation data, and followed by three isolated cold-start
# prediction processes (complete test-set prediction time).
# Output: results/budget/<dataset>/<config>/seed<k>/{train,infer_cold*}/
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/budget}"

"${PYTHON}" -u -m SCOPE_GT_budgets.run budget \
  --data-base "${DATA_BASE}" --out "${OUT}" --datasets wikics dblp --seeds "${SEED_LIST[@]}" \
  --axes k w --k-values 8 16 24 32 48 64 --walk-values 12 24 36 48 72 96 \
  --device "${INDEXED_DEVICE}" --threads "${THREADS}" --resume "$@"
