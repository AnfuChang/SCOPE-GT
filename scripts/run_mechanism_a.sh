#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
for dataset in wikics corafull; do
  "${PYTHON}" -m SCOPE_GT_mechanisms.run A --dataset "${dataset}" \
    --data-root "${DATA_BASE}/${dataset}" --out "${OUT:-results/mechanism_a}/${dataset}" \
    --device "${DEVICE}" --threads "${THREADS}" --seeds "${SEED_LIST[@]}" "$@"
done
