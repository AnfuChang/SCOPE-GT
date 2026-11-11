#!/usr/bin/env bash
# End-to-end efficiency: graph-size curves and complete-dataset training.
#   DATA_BASE=/path/to/datasets bash scripts/run_efficiency.sh
# Methods: SCOPE-GT, GraphGPS, Exphormer, SGFormer, Gophormer, NLGT, SwapGT (recipes in SCOPE_GT_efficiency/config.py).
# Backbones use bundled Baselines/; SCOPE_GT_BASELINES_ROOT optionally overrides it.
#  (a,b) nested Arxiv-year induced subgraphs with 5k/10k/20k/40k/60k nodes (graph seed 2027, built without labels);
#        3 warmup + 10 measured epochs per method and size.
#  (c): Computers training with per-epoch validation, early stopping and a 7,200 s observation window.
# Epoch, preparation and whole-job watchdogs are disabled by default. The observation window applies
# only to Computers convergence; --observation-seconds 0 disables it explicitly.
# Needs DATA_BASE/arxiv_year and DATA_BASE/amazon_computers. Every job runs in its own process, one at a time.
# Output: results/efficiency/{summary.csv,per_seed.csv,failures.csv}, arxiv_year_scaling/, computers_convergence/
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/efficiency}"

"${PYTHON}" -u -m SCOPE_GT_efficiency.run all \
  --scaling-dataset arxiv_year --sizes 5000 10000 20000 40000 60000 \
  --data-base "${DATA_BASE}" --out "${OUT}" --seeds "${SEED_LIST[@]}" --device "${INDEXED_DEVICE}" --threads "${THREADS}" "$@"
