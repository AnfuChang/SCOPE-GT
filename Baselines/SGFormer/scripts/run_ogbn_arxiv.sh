#!/usr/bin/env bash
set -euo pipefail
# Hyperparameters come from the ogbn_arxiv preset of run_sgformer.sh (official large/run.sh command);
# only the protocol is set here. Any variable of run_sgformer.sh can still be overridden by name.
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets} \
RUNS=${RUNS:-5} SEED=${SEED:-42} \
EPOCHS=${EPOCHS:-1000} EVAL_EVERY=${EVAL_EVERY:-5} PATIENCE=${PATIENCE:-100} \
TO_UNDIRECTED=${TO_UNDIRECTED:-1} METRIC=${METRIC:-acc} \
bash run_sgformer.sh ogbn_arxiv
