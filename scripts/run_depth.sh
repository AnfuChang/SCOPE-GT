#!/usr/bin/env bash
# Fig. 6: attention depth L on WikiCS (L = 1/2/4/8/16) and Minesweeper (L = 1/2/4/8), SCOPE-GT center-query reader.
#   DATA_BASE=/path/to/datasets bash scripts/run_depth.sh
# All other settings follow SCOPE_GT/presets.json (50 and 120 training epochs); dense execution layout.
# Output: results/depth/<dataset>/L<depth>/center/seed<k>/ and results/depth/summary/summary.csv
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/depth}"

"${PYTHON}" -u -m SCOPE_GT_depth.run --data-base "${DATA_BASE}" --out-root "${OUT}" \
  --datasets wikics --layers 1 2 4 8 16 --roles center --seeds "${SEED_LIST[@]}" \
  --device "${DEVICE}" --threads "${THREADS}" --resume
"${PYTHON}" -u -m SCOPE_GT_depth.run --data-base "${DATA_BASE}" --out-root "${OUT}" \
  --datasets minesweeper --layers 1 2 4 8 --roles center --seeds "${SEED_LIST[@]}" \
  --device "${DEVICE}" --threads "${THREADS}" --resume
"${PYTHON}" -m SCOPE_GT_depth.collect --results "${OUT}"
