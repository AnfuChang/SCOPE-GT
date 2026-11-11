# Shared settings sourced by every launcher in scripts/.
#   DATA_BASE  folder with one sub-folder per dataset (required; see data/README.md)
#   PYTHON     Python interpreter (default: python)
#   DEVICE     cuda (default) or cpu, where supported
#   THREADS    CPU threads per training process (default: 6)
#   SEEDS      training seeds (default: "0 1 2 3 4")
#   OUT        output root (default: results/<experiment>)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
if [[ "${DEVICE}" == cuda ]]; then INDEXED_DEVICE=cuda:0; else INDEXED_DEVICE="${DEVICE}"; fi
THREADS="${THREADS:-6}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"

require_data_base() {
  : "${DATA_BASE:?Set DATA_BASE to the folder with one sub-folder per dataset (see data/README.md)}"
  DATA_BASE="$(cd -- "${DATA_BASE}" && pwd -P)"
  export DATA_BASE
}
