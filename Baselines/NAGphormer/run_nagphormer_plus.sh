#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_nagphormer_plus.sh <dataset> [cuda:0]
#
# This wrapper runs the same backbone as run_nagphormer.sh, but enables
# Neighborhood Augmentation so the reported model name is nagphormer_plus.
# Backbone hyperparameters per dataset: the NAGphormer presets of run_nagphormer.sh (NAGphormer+
# reuses the NAGphormer configuration; its paper covers PubMed, Computer, Photo, CS and Amazon2M =
# ogbn-products among our datasets). NrAug: NAGphormer+ only reports search ranges (tau in
# {0.25,0.5,0.75}, p_aug in {0.25,0.5,0.75,1.0}, Beta(1,1)); NR_AUG_TAU=0.25 and NR_AUG_PAUG=0.5 are
# used for every dataset unless overridden.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL="${MODEL:-nagphormer_plus}"
export NR_AUG="${NR_AUG:-on}"
exec bash "${SCRIPT_DIR}/run_nagphormer.sh" "$@"
