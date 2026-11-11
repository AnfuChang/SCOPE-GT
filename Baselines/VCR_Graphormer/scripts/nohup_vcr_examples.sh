#!/usr/bin/env bash
set -euo pipefail

# Set DATA_ROOT to the parent folder containing BILGT-format datasets.
# Example:
#   DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets DEVICE=0 bash scripts/nohup_vcr_examples.sh

DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}
DEVICE=${DEVICE:-0}
OUT=${OUT:-nohup_logs_vcr}
mkdir -p "$OUT"

# One line per dataset, five runs each, seeds 42-46 by default.
nohup bash run_vcr_baseline.sh pubmed  "$DATA_ROOT/pubmed"  "$DEVICE" > "$OUT/vcr_pubmed.log" 2>&1 &
nohup bash run_vcr_baseline.sh corafull "$DATA_ROOT/corafull" "$DEVICE" > "$OUT/vcr_corafull.log" 2>&1 &
nohup bash run_vcr_baseline.sh flickr  "$DATA_ROOT/flickr"  "$DEVICE" > "$OUT/vcr_flickr.log" 2>&1 &
nohup bash run_vcr_baseline.sh github  "$DATA_ROOT/github"  "$DEVICE" > "$OUT/vcr_github.log" 2>&1 &
# nohup bash run_vcr_baseline.sh ogbn_arxiv "$DATA_ROOT/ogbn_arxiv" "$DEVICE" > "$OUT/vcr_ogbn_arxiv.log" 2>&1 &

jobs -l
