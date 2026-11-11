#!/usr/bin/env bash
# GT-SNT (AAAI'26, github.com/Zhhuizhe/GT-SNT) under the shared comparison protocol.
#   bash run_gtsnt.sh <dataset>                         e.g. bash run_gtsnt.sh cora
#   RUNS=10 SPLIT_SUFFIXES="_0 _1 _2 _3 _4 _5 _6 _7 _8 _9" bash run_gtsnt.sh roman_empire   (10 official splits)
#   LR=1e-3 T=4 bash run_gtsnt.sh amazon_photo          (any hyperparameter below can be overridden by name)
# Needs torch_geometric, spikingjelly==0.0.0.0.14 (checked below) and ../common/protocol.py; torch_sparse optional.
set -euo pipefail

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Fixed by project comparison protocol (identical in every run_*.sh, see ../common/protocol.py)
RUNS="${RUNS:-5}"                      # seeds BASE_SEED .. BASE_SEED+RUNS-1 (official configs: 1 run, seed 123)
BASE_SEED="${BASE_SEED:-42}"
EPOCHS="${EPOCHS:-}"                   # empty -> max(1000, official epochs); one epoch = one full-graph step
                                       # (or one pass over the training batches on ogbn-products / pokec)
PATIENCE="${PATIENCE:-30}"             # validation checks without improvement
EVAL_EVERY="${EVAL_EVERY:-5}"          # one validation check every EVAL_EVERY epochs (+ epoch 1 and the last)
SPLIT_SUFFIXES="${SPLIT_SUFFIXES:-}"   # empty -> train.npy/val.npy/test.npy; run r uses suffix r % count
METRIC="${METRIC:-}"                   # empty -> ROC-AUC for questions/minesweeper/tolokers, accuracy otherwise

DATASET=${1:-cora}
# Data layout: ${DATA_ROOT}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets/${DATASET}}
# DATA_ROOT may also be the folder that holds all datasets
if [[ ! -f "${DATA_ROOT}/adj.npz" && -f "${DATA_ROOT}/${DATASET}/adj.npz" ]]; then DATA_ROOT="${DATA_ROOT}/${DATASET}"; fi
DEVICE="${DEVICE:-cuda:1}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

TUNABLE=(PIPELINE HID CB LAYERS HEADS T NEURON V_THRESHOLD DROPOUT ATTN_DROPOUT MAX_CODES NORMALIZE RESIDUAL
         BN LR WD BATCH_SIZE FEATURE_NORM OFFICIAL_EPOCHS)
for v in "${TUNABLE[@]}"; do eval "OVR_${v}=\"\${${v}:-}\""; done

# ---- GT-SNT hyperparameters: the repository's per-dataset configs (official/config/*.yaml) -- the paper has no
#      hyperparameter table. Arguments: hid_channels, cb_channels, num_layers, num_heads, T, v_threshold, dropout,
#      attn_dropout, maximum_codes_num, normalize, enable_residual, enable_norm, lr, wd; all configs use the PLIF
#      neuron, "default" edge coding and cross-entropy. Feature preprocessing as dataset.py: NormalizeFeatures
#      for Amazon (Computers, Photo) and Coauthor, raw features otherwise.
#      Datasets without a config use the closest one (marked fallback:<config>):
#        amazon_photo, amazon_computers, corafull, facebook_large, wikics -> cs (homophilous, medium size,
#          the only medium config with residual + BatchNorm); NormalizeFeatures for the bag-of-words ones
#        roman_empire, amazon_ratings, minesweeper, tolokers, questions -> cs as well: the paper's heterophilous
#          datasets (Actor, Deezer) have no released config; raw features as in their official loaders
#        arxiv_year -> ogbn-arxiv (same graph and features)
#        pokec -> ogbn-products (subgraph pipeline: 1.6M nodes do not fit the full-graph script) with batch 2048:
#          816,320 training nodes / 2048 = 399 optimizer steps per epoch, as in the official products setting
#          (196,615 / 512 = 384); batch 512 would give 1,594 steps per epoch, and every step runs the tokenizer
#          on the full graph and the GCN on a 3-hop subgraph holding most of the edges
#        lastfm_asia -> cs (NormalizeFeatures): social network with binary node attributes, people as nodes as in
#          Coauthor (facebook_large is in the same group)
#        github -> physics (NormalizeFeatures): social network of Physics' size and density (37.7k nodes, mean
#          degree 15.3 vs 34.5k, 14.4)
#        dblp -> the pubmed setting (config/pubmed.yaml, raw features): citation graph of PubMed's size
#      These fallbacks are a starting point; tune them on the validation set only (see README).
NEURON=PLIF; MAX_CODES=0; BATCH_SIZE=512; FEATURE_NORM=none
#      cfg <pipeline> <hid> <cb> <layers> <heads> <T> <v_threshold> <dropout> <attn_dropout> <normalize>
#          <residual> <bn> <lr> <wd> <official epochs>
cfg() { PIPELINE=$1; HID=$2; CB=$3; LAYERS=$4; HEADS=$5; T=$6; V_THRESHOLD=$7; DROPOUT=$8; ATTN_DROPOUT=$9
        NORMALIZE=${10}; RESIDUAL=${11}; BN=${12}; LR=${13}; WD=${14}; OFFICIAL_EPOCHS=${15}; }
cfg_cora()     { cfg full 512 8 2 4 3 1.0 0.5 0.5 0 0 0 0.001 0.001 1000; }       # config/cora.yaml
cfg_citeseer() { cfg full 512 16 2 4 3 1.0 0.5 0.5 1 0 0 0.001 0.001 1000; }      # config/citeseer.yaml
cfg_pubmed()   { cfg full 256 8 2 4 3 0.6 0.2 0.8 0 0 0 0.001 0.01 1000; }        # config/pubmed.yaml
cfg_cs()       { cfg full 512 8 1 4 4 1.0 0.5 0.5 1 1 1 0.0001 0.0001 1000; }     # config/cs.yaml
cfg_physics()  { cfg full 256 8 1 4 3 1.0 0.2 0.5 1 1 0 0.001 0.00001 1000; }     # config/physics.yaml
cfg_arxiv()    { cfg full 128 6 4 2 8 0.6 0.2 0.8 1 0 1 0.001 0.0001 2000; }      # config/ogbn-arxiv.yaml
#      (ogbn-arxiv.yaml also lists batch_size 4096, read only by node_cls_subgraph.py; the README runs arxiv
#       with node_cls.py, i.e. full graph)
cfg_products() { cfg subgraph 32 8 3 2 3 1.0 0.5 0.5 0 0 0 0.001 0.001 2000; MAX_CODES=64; BATCH_SIZE=512; }  # ogbn-products.yaml
case "${KEY}" in
  # ---- official configs
  cora)                       cfg_cora;     SRC="official:config/cora.yaml" ;;
  citeseer)                   cfg_citeseer; SRC="official:config/citeseer.yaml" ;;
  pubmed)                     cfg_pubmed;   SRC="official:config/pubmed.yaml" ;;
  coauthor_cs|cs)             cfg_cs;       FEATURE_NORM=pyg; SRC="official:config/cs.yaml" ;;
  coauthor_physics|physics)   cfg_physics;  FEATURE_NORM=pyg; SRC="official:config/physics.yaml" ;;
  ogbn_arxiv|arxiv)           cfg_arxiv;    SRC="official:config/ogbn-arxiv.yaml" ;;
  ogbn_products|products)     cfg_products; SRC="official:config/ogbn-products.yaml" ;;
  # ---- no config for the dataset: closest one
  amazon_photo|photo|amazon_computers|computers|computer|corafull|facebook_large|facebook|lastfm_asia)
                              cfg_cs;       FEATURE_NORM=pyg; SRC="fallback:cs" ;;
  dblp)                       cfg_pubmed;   SRC="fallback:config/pubmed.yaml" ;;
  github)                     cfg_physics;  FEATURE_NORM=pyg; SRC="fallback:config/physics.yaml" ;;
  wikics|roman_empire|amazon_ratings|minesweeper|tolokers|questions)
                              cfg_cs;       SRC="fallback:cs" ;;
  arxiv_year)                 cfg_arxiv;    SRC="fallback:ogbn-arxiv" ;;
  pokec)                      cfg_products; BATCH_SIZE=2048; SRC="fallback:ogbn-products+batch2048" ;;
  *) echo "note: '${DATASET}' has no setting of its own, using the cs config" >&2
                              cfg_cs;       SRC="fallback:cs" ;;
esac
for v in "${TUNABLE[@]}"; do eval "o=\"\${OVR_${v}}\""; if [[ -n "${o}" ]]; then eval "${v}=\"\${o}\""; SRC="${SRC}+${v}"; fi; done
if [[ -z "${EPOCHS}" ]]; then EPOCHS=$(( OFFICIAL_EPOCHS > 1000 ? OFFICIAL_EPOCHS : 1000 )); fi

# ---- dependency check: fails in seconds instead of after loading the data
DEPS_REPORT="$("${PYTHON}" - <<'PY' 2>&1 || true
import importlib
for module, spec in [("torch_geometric", "torch_geometric"), ("spikingjelly.clock_driven.neuron", "spikingjelly==0.0.0.0.14")]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"MISSING {spec}    [{type(exc).__name__}: {exc}]")
PY
)"
if grep -q "^MISSING" <<<"${DEPS_REPORT}"; then
  echo "[GT-SNT] missing Python packages in $(command -v "${PYTHON}"):" >&2
  grep "^MISSING" <<<"${DEPS_REPORT}" | sed 's/^MISSING /  - /' >&2
  echo "[GT-SNT] install into this environment, e.g.:  ${PYTHON} -m pip install \"spikingjelly==0.0.0.0.14\"" >&2
  exit 2
fi

echo "==== GT-SNT | dataset=${DATASET} | hyperparameters: ${SRC} ===="
echo "protocol: RUNS=${RUNS} BASE_SEED=${BASE_SEED} EPOCHS=${EPOCHS} (official ${OFFICIAL_EPOCHS}) PATIENCE=${PATIENCE} EVAL_EVERY=${EVAL_EVERY} SPLIT_SUFFIXES='${SPLIT_SUFFIXES}'"

CMD=("${PYTHON}" "${SCRIPT_DIR}/gtsnt_npz.py"
  --data-root "${DATA_ROOT}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --device "${DEVICE}"
  --runs "${RUNS}" --base-seed "${BASE_SEED}" --epochs "${EPOCHS}" --patience "${PATIENCE}" --eval-every "${EVAL_EVERY}"
  --pipeline "${PIPELINE}" --hid-channels "${HID}" --cb-channels "${CB}" --num-layers "${LAYERS}"
  --num-heads "${HEADS}" --T "${T}" --neuron "${NEURON}" --v-threshold "${V_THRESHOLD}" --dropout "${DROPOUT}"
  --attn-dropout "${ATTN_DROPOUT}" --maximum-codes-num "${MAX_CODES}" --normalize "${NORMALIZE}"
  --enable-residual "${RESIDUAL}" --enable-norm "${BN}" --lr "${LR}" --wd "${WD}" --batch-size "${BATCH_SIZE}"
  --feature-norm "${FEATURE_NORM}" --official-epochs "${OFFICIAL_EPOCHS}")
if [[ -n "${SPLIT_SUFFIXES}" ]]; then CMD+=(--split-suffixes ${SPLIT_SUFFIXES}); fi
if [[ -n "${METRIC}" ]]; then CMD+=(--metric "${METRIC}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
exec "${CMD[@]}"
