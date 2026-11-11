#!/usr/bin/env bash
# Table V and Fig. 7: execution strategies across training batch sizes B on WikiCS and CoraFull.
#   DATA_BASE=/path/to/datasets bash scripts/run_batch_size.sh cost       # epoch time / GPU memory (GPU)
#   DATA_BASE=/path/to/datasets bash scripts/run_batch_size.sh sharing    # repeated token reads (CPU)
#   DATA_BASE=/path/to/datasets bash scripts/run_batch_size.sh accuracy   # full training at every B (GPU)
#   bash scripts/run_batch_size.sh collect                                # results/batch_size/summary_*.csv
# cost      Padded, Packed and SCOPE-GT (dedup_packed) on identical sampled inputs at each B; five paired seeds, one isolated
#           process per layout, 3 warmup + 10 measured epochs. Dropout-free logits and gradients are compared on
#           CPU before timing (equivalence.json). B = 16..256 on WikiCS, 16..1024 on CoraFull.
#           Use -- --cost-seed 0 explicitly for the historical seed-0 curves.
# sharing   token reads M and unique token nodes U of every epoch-0 mini-batch (fixed center order), B = 16..2048.
# accuracy  five seeds per B; not plotted in the paper.
# Extra arguments go to the runner, e.g.  bash scripts/run_batch_size.sh cost wikics -- --batch-sizes 64 256
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
OUT="${OUT:-${ROOT}/results/batch_size}"
PHASE="${1:?usage: run_batch_size.sh cost|sharing|accuracy|collect [DATASET ...] [-- runner flags]}"
shift

DATASETS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do DATASETS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(wikics corafull)

case "${PHASE}" in
  cost|accuracy)
    require_data_base
    "${PYTHON}" -u -m SCOPE_GT_batch_size.run "${PHASE}" --datasets "${DATASETS[@]}" \
      --data-base "${DATA_BASE}" --out-root "${OUT}" --device "${DEVICE}" --threads "${THREADS}" \
      --seeds "${SEED_LIST[@]}" "$@" ;;
  sharing)
    require_data_base
    "${PYTHON}" -u -m SCOPE_GT_batch_size.run sharing --datasets "${DATASETS[@]}" \
      --data-base "${DATA_BASE}" --out-root "${OUT}" --device cpu --threads "${THREADS}" \
      --seeds "${SEED_LIST[@]}" "$@" ;;
  collect)
    "${PYTHON}" -m SCOPE_GT_batch_size.collect --out-root "${OUT}" "$@" ;;
  *)
    echo "unknown phase ${PHASE}" >&2; exit 2 ;;
esac
