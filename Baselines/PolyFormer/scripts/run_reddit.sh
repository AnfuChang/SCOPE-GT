#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
BASE=${BASE:-mono} TRAIN_MODE=${TRAIN_MODE:-batch} BATCH_SIZE=${BATCH_SIZE:-20000} bash run_polyformer.sh reddit
