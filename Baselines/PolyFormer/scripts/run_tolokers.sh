#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
BASE=${BASE:-mono} METRIC=${METRIC:-rocauc} FORCE_BINARY=${FORCE_BINARY:-1} bash run_polyformer.sh tolokers
