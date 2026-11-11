#!/usr/bin/env bash
# LastFM transfer: Shared-Context Mini-Batching on NLGT, Gophormer, HubGT and SwapGT.
#   DATA_BASE=/path/to/datasets bash scripts/run_transfer.sh
# Native and +SCMB share the backbone sampler, inputs, losses, optimizer, inference rule, checkpoint selection,
# configuration and initial parameters. HubGT needs its C++ index (Cython>=3.0, g++ with OpenMP), compiled below.
# Compatible NPZ adapters are bundled; SCOPE_GT_BASELINES_ROOT optionally overrides them.
# Output: results/transfer/{summary.csv,paired_summary.csv,transfer_table.tex} and per-run records.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/transfer}"

if [[ " $* " != *" --dry-run "* && " $* " != *" --help "* ]]; then
  "${PYTHON}" -m SCOPE_GT_transfer.build_hubgt
fi
"${PYTHON}" -u -m SCOPE_GT_transfer.run \
  --data-base "${DATA_BASE}" --out "${OUT}" \
  --datasets lastfm_asia --methods nlgt gophormer hubgt swapgt --variants native scmb \
  --seeds "${SEED_LIST[@]}" --device "${INDEXED_DEVICE}" --threads "${THREADS}" "$@"
