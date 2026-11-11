#!/usr/bin/env bash
# Table IV: predictive ablations, each variant trained from scratch with the dataset preset.
#   DATA_BASE=/path/to/datasets bash scripts/run_ablation.sh
#   DATA_BASE=/path/to/datasets bash scripts/run_ablation.sh wikics roman_empire
# Variants (SCOPE_GT_ablation/variants.py):
#   A4  w/o EA      --external-evidence attributes_off  zero external supporting-node attributes in path and
#                                                       background summaries; structure and denominators kept
#   A5  w/o EPS     --external-evidence structure_off   path counts, transition mass and path indicators from
#                                                       paths inside S_c only; topology and attribute summaries kept
#   A1  w/o R       --relations off                     (no Graph-Grounded Relation Matrix)
#   A3  w/o S       --sampling uniform_rw               (uniform random walks instead of Context Sampling)
#   A2  Full-query  --query context --execution dense   (context tokens are also queries)
# The full model is the SCOPE-GT row of scripts/run_main.sh.
# Output: results/ablation/<dataset>/{A4,A5,A1,A3,A2}_<timestamp>/
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/ablation}"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(lastfm_asia minesweeper wikics tolokers roman_empire)
fi

for dataset in "${DATASETS[@]}"; do
  "${PYTHON}" -m SCOPE_GT_ablation.launch "${dataset}" --variant external controls \
    --data-root "${DATA_BASE}/${dataset}" --out-root "${OUT}" \
    --seeds "${SEED_LIST[@]}" --device "${DEVICE}" --threads "${THREADS}"
done
