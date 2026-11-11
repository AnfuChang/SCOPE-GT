#!/usr/bin/env bash
set -euo pipefail
# Hyperparameters come from the tolokers preset of run_sgformer.sh (fallback: official deezer-europe
# command); only the protocol is set here. ROC-AUC is used for model selection, output and mean/std.
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets} \
RUNS=${RUNS:-5} SEED=${SEED:-42} \
EPOCHS=${EPOCHS:-1000} EVAL_EVERY=${EVAL_EVERY:-5} PATIENCE=${PATIENCE:-100} \
TO_UNDIRECTED=${TO_UNDIRECTED:-1} METRIC=${METRIC:-rocauc} \
bash run_sgformer.sh tolokers
