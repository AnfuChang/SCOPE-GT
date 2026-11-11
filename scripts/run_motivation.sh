#!/usr/bin/env bash
# Section II diagnostics on WikiCS (five paired seeds by default).
#   DATA_BASE=/path/to/datasets bash scripts/run_motivation.sh            # all three
#   DATA_BASE=/path/to/datasets bash scripts/run_motivation.sh relation   # one of: relation overlap query
#  relation  Fig. 1(a): Plain / Semantics / Strength / Neighborhood / All at N_ctx = 4..64
#  overlap   Fig. 1(b): node occurrences in the first mini-batch of 64 NLGT training sequences (CPU only)
#  query     Fig. 3(a): Full / Partial / Center first-layer queries at N_ctx = 4..64
# Output: results/motivation/{relation,overlap,query}/
# The overlap stage uses the bundled NLGT NPZ adapter.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_data_base
OUT="${OUT:-${ROOT}/results/motivation}"
export PYTHONPATH="${ROOT}/Motivation:${PYTHONPATH}"
MOTIVATION_DEVICE="${DEVICE}"
[[ "${MOTIVATION_DEVICE}" == cuda || "${MOTIVATION_DEVICE}" == cpu ]] || MOTIVATION_DEVICE=auto
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(relation overlap query)

for stage in "${STAGES[@]}"; do
  case "${stage}" in
    relation)
      "${PYTHON}" -m R_experiment.motivation \
        --data-root "${DATA_BASE}/wikics" --dataset wikics --out "${OUT}/relation/wikics" \
        --k 4 8 16 32 64 --seeds "${SEED_LIST[@]}" --modes plain witness strength neighborhood all \
        --hops 2 --calib-centers 256 --neighborhood-block-size 512 \
        --hidden-dim 64 --heads 4 --ff-dim 128 --dropout 0.3 --channel-init 0.1 \
        --lr 1e-3 --weight-decay 5e-4 --epochs 100 --patience 0 --batch-size 64 --eval-batch-size 128 \
        --feature-normalize l1 --split-index 0 --metric auto \
        --device "${MOTIVATION_DEVICE}" --threads "${THREADS}" --log-every 10
      ;;
    overlap)
      "${PYTHON}" Motivation/O_experiment/run_overlap.py \
        --data-root "${DATA_BASE}" --datasets wikics --methods nlgt --seeds "${SEED_LIST[@]}" --batch-sizes 64 \
        --out "${OUT}/overlap"
      ;;
    query)
      "${PYTHON}" -m A_experiment.experiment \
        --data-root "${DATA_BASE}/wikics" --dataset wikics --out "${OUT}/query/wikics" \
        --sizes 4 8 16 32 64 --modes full partial center --partial-ratio 0.5 --seeds "${SEED_LIST[@]}" \
        --hops 2 --hidden-dim 64 --epochs 80 --patience 0 --batch-size 256 \
        --feature-normalize l1 --device "${MOTIVATION_DEVICE}" --threads "${THREADS}"
      ;;
    *) echo "unknown stage ${stage}; use relation, overlap or query" >&2; exit 2 ;;
  esac
done
