#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
DATASETS=(citeseer tolokers PubMed facebook_large roman_empire Questions Flickr ogbn_arxiv reddit ogbn_products ogbn_mag ethanol Paper100m amazon_computers amazon_photo coauthor_cs wikics coauthor_physics github)
for ds in "${DATASETS[@]}"; do
  echo "============================== ${ds} =============================="
  bash run_polyformer.sh "${ds}"
done
