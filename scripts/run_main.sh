#!/usr/bin/env bash
# Tables III-IV: SCOPE-GT node classification (five seeds per fixed split).
#   DATA_BASE=/path/to/datasets bash scripts/run_main.sh                 # all sixteen datasets
#   DATA_BASE=/path/to/datasets bash scripts/run_main.sh wikics dblp     # selected datasets
# SPLIT_INDEX selects one official Platonov mask (default 0); other datasets use their fixed split.
# Settings come from SCOPE_GT/presets.json. Output: results/main/<dataset>/A0_<timestamp>/
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/main}"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(lastfm_asia minesweeper wikics tolokers amazon_computers dblp coauthor_cs pubmed
            corafull facebook_large roman_empire amazon_ratings github arxiv_year pokec ogbn_products)
fi

for dataset in "${DATASETS[@]}"; do
  SPLIT_ARGS=()
  case "${dataset}" in
    minesweeper|roman_empire|amazon_ratings|tolokers) SPLIT_ARGS=(--split-index "${SPLIT_INDEX:-0}") ;;
  esac
  "${PYTHON}" -m SCOPE_GT_ablation.launch "${dataset}" --variant Full \
    --data-root "${DATA_BASE}/${dataset}" --out-root "${OUT}" \
    --seeds "${SEED_LIST[@]}" --device "${DEVICE}" --threads "${THREADS}" "${SPLIT_ARGS[@]}"
done
