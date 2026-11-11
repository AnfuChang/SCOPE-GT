#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
for dataset in corafull ogbn_products; do
  "${PYTHON}" -m SCOPE_GT_mechanisms.run B --dataset "${dataset}" \
    --data-root "${DATA_BASE}/${dataset}" --out "${OUT:-results/mechanism_b}/${dataset}" \
    --threads "${THREADS}" --seeds "${SEED_LIST[@]}" "$@"
done
