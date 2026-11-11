#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-flickr}

# Paths. DATA_ROOT follows the layout used by the BiLGT project:
#   ${DATA_ROOT}/${DATASET}/{adj.npz,feat.npy,label.npy,train.npy,val.npy,test.npy}
DATA_ROOT=${DATA_ROOT:-/mnt/sdb/changaf25/sdust/datasets}
if [[ "${DATASET}" == "custom" ]]; then
  if [[ -z "${DATA_DIR:-}" ]]; then
    echo "[Error] DATASET=custom requires DATA_DIR=/path/to/dataset" >&2
    exit 1
  fi
else
  DATA_DIR=${DATA_DIR:-${DATA_ROOT}/${DATASET}}
fi

if [[ -z "${OUT_ROOT:-}" ]]; then
  OUT_ROOT="$(mktemp -d -t sgformer_node_${DATASET}_XXXXXX)"
  trap 'rm -rf "${OUT_ROOT}"' EXIT
fi
DEVICE=${DEVICE:-cuda:0}

# Repeated runs.
RUNS=${RUNS:-5}
SEED=${SEED:-42}
SEED_STEP=${SEED_STEP:-1}

# ---------------------------------------------------------------------------------------------
# Presets for the 14 datasets of the ACE_GTv6 experiments (': "${VAR:=v}"' -> environment wins).
# Adapter names: ALPHA = official graph_weight (weight of the GCN branch), BETA = official alpha
# (residual weight of the attention layer; the large-graph code averages, i.e. 0.5), DROPOUT /
# WEIGHT_DECAY = Transformer side (ours_* / trans_*), GNN_DROPOUT / GNN_WEIGHT_DECAY = GCN branch.
# Official SGFormer commands (medium/run.sh, large/run.sh of the repo):
#   pubmed        : GCN 4 layers, hidden 64, lr 5e-3, GCN wd 5e-4 / dropout 0.5, 1 attention layer,
#                   graph_weight 0.8, attention dropout 0.3 / wd 0.01, alpha 0.5, no feature norm
#   ogbn-arxiv    : hidden 256, lr 1e-3, GCN 3 layers (dropout 0.5, wd 0), 1 attention layer
#                   (dropout 0.5, wd 0), graph_weight 0.5, full batch
#   amazon2m      : (= ogbn-products graph) hidden 256, lr 1e-2, GCN 3 layers, dropout 0, wd 0,
#                   graph_weight 0.5, mini-batches of 100k nodes
#   pokec         : hidden 64, lr 1e-2, GCN 2 layers, dropout 0, wd 0, graph_weight 0.5, batches of 100k
#   chameleon     : GCN 2 layers, hidden 64, lr 1e-3, wd 1e-3, dropout 0.6, 1 attention layer, alpha 0.5
#   deezer-europe : GCN 2 layers, hidden 96, lr 1e-2, wd 5e-5, dropout 0.4, 1 attention layer, alpha 0.5
# (the "Squirrel" and "Actor" lines of medium/run.sh run --method difformer, i.e. DIFFormer, so the
#  repo has no SGFormer command for squirrel)
#   official : pubmed, ogbn_arxiv, ogbn_products (amazon2m), pokec
#   fallback : squirrel, roman_empire, amazon_ratings -> chameleon; minesweeper, tolokers, questions
#              -> deezer-europe (binary ROC-AUC); amazon_photo, amazon_computers, coauthor_cs, wikics, corafull,
#              facebook_large, github, lastfm_asia, dblp -> pubmed; arxiv_year -> ogbn-arxiv (same graph)
# ---------------------------------------------------------------------------------------------
KEY="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
sgf_pubmed() {
  : "${HIDDEN_DIM:=64}"; : "${GCN_LAYERS:=4}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.8}"; : "${BETA:=0.5}"
  : "${LR:=0.005}"; : "${GNN_DROPOUT:=0.5}"; : "${GNN_WEIGHT_DECAY:=0.0005}"; : "${DROPOUT:=0.3}"; : "${WEIGHT_DECAY:=0.01}"
  : "${FEATURE_NORM:=none}"
}
sgf_arxiv() {
  : "${HIDDEN_DIM:=256}"; : "${GCN_LAYERS:=3}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.5}"; : "${BETA:=0.5}"
  : "${LR:=0.001}"; : "${GNN_DROPOUT:=0.5}"; : "${GNN_WEIGHT_DECAY:=0.0}"; : "${DROPOUT:=0.5}"; : "${WEIGHT_DECAY:=0.0}"
  # large/run.sh ogbn-arxiv: --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act (no --gnn_use_init)
  : "${ARCH:=official_large}"; : "${GNN_USE_INIT:=0}"
}
# Large graphs (ogbn_arxiv, arxiv_year, ogbn_products, pokec) use ARCH=official_large, the model of the
# official large/ours.py: the GCN branch has its own input layer, ReLU after every layer and an initial
# residual (+ the input-layer output), which the adapter model (SGFormerNode) lacks; without it the
# heterophilous arxiv_year / pokec fall to MLP level. Partition-trained graphs are evaluated on the
# whole graph on the CPU (EVAL_MODE=full_cpu), as evaluate_large of the official main-batch.py; the
# former EVAL_MODE=batch evaluated contiguous node-id blocks and dropped most edges at test time.
sgf_chameleon() {
  : "${HIDDEN_DIM:=64}"; : "${GCN_LAYERS:=2}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.8}"; : "${BETA:=0.5}"
  : "${LR:=0.001}"; : "${GNN_DROPOUT:=0.6}"; : "${GNN_WEIGHT_DECAY:=0.001}"; : "${DROPOUT:=0.6}"; : "${WEIGHT_DECAY:=0.001}"
}
sgf_deezer() {
  : "${HIDDEN_DIM:=96}"; : "${GCN_LAYERS:=2}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.8}"; : "${BETA:=0.5}"
  : "${LR:=0.01}"; : "${GNN_DROPOUT:=0.4}"; : "${GNN_WEIGHT_DECAY:=0.00005}"; : "${DROPOUT:=0.4}"; : "${WEIGHT_DECAY:=0.00005}"
}
case "${KEY}" in
  pubmed)
    sgf_pubmed; SRC="official:medium/run.sh-pubmed" ;;
  ogbn_arxiv|arxiv)
    sgf_arxiv; SRC="official:large/run.sh-ogbn-arxiv" ;;
  ogbn_products|products|amazon2m)
    : "${HIDDEN_DIM:=256}"; : "${GCN_LAYERS:=3}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.5}"; : "${BETA:=0.5}"
    : "${LR:=0.01}"; : "${GNN_DROPOUT:=0.0}"; : "${GNN_WEIGHT_DECAY:=0.0}"; : "${DROPOUT:=0.0}"; : "${WEIGHT_DECAY:=0.0}"
    : "${TRAIN_MODE:=batch}"; : "${EVAL_MODE:=full_cpu}"; : "${BATCH_SIZE:=100000}"
    : "${ARCH:=official_large}"; : "${GNN_USE_INIT:=1}"   # large/run.sh: ... --gnn_use_init --gnn_use_act
    SRC="official:large/run.sh-amazon2m" ;;
  pokec)
    : "${HIDDEN_DIM:=64}"; : "${GCN_LAYERS:=2}"; : "${TRANS_LAYERS:=1}"; : "${ALPHA:=0.5}"; : "${BETA:=0.5}"
    : "${LR:=0.01}"; : "${GNN_DROPOUT:=0.0}"; : "${GNN_WEIGHT_DECAY:=0.0}"; : "${DROPOUT:=0.0}"; : "${WEIGHT_DECAY:=0.0}"
    : "${TRAIN_MODE:=batch}"; : "${EVAL_MODE:=full_cpu}"; : "${BATCH_SIZE:=100000}"
    : "${ARCH:=official_large}"; : "${GNN_USE_INIT:=1}"   # large/run.sh: ... --gnn_use_init --gnn_use_act
    SRC="official:large/run.sh-pokec" ;;
  arxiv_year)
    sgf_arxiv; SRC="fallback:ogbn-arxiv" ;;
  squirrel|roman_empire|amazon_ratings)
    sgf_chameleon; SRC="fallback:medium/run.sh-chameleon" ;;
  minesweeper|tolokers|questions)
    sgf_deezer; SRC="fallback:medium/run.sh-deezer-europe" ;;
  amazon_photo|photo|amazon_computers|computers|computer|coauthor_cs|cs|lastfm_asia|dblp|wikics|corafull \
  |facebook_large|facebook|github)
    # homophilous / social medium graphs: the repository's only homophilous medium-graph command (PubMed); wikics,
    # corafull, facebook_large and github used to get the generic defaults below
    sgf_pubmed; SRC="fallback:medium/run.sh-pubmed" ;;
  *)
    SRC="generic-defaults" ;;
esac

# Training hyperparameters. These are deliberately simple and close to the paper's search space.
EPOCHS=${EPOCHS:-1000}
EVAL_EVERY=${EVAL_EVERY:-5}
PATIENCE=${PATIENCE:-100}
HIDDEN_DIM=${HIDDEN_DIM:-256}
GCN_LAYERS=${GCN_LAYERS:-2}
TRANS_LAYERS=${TRANS_LAYERS:-2}
ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.5}
DROPOUT=${DROPOUT:-0.3}
GNN_DROPOUT=${GNN_DROPOUT:-${DROPOUT}}
LR=${LR:-0.005}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
GNN_WEIGHT_DECAY=${GNN_WEIGHT_DECAY:-${WEIGHT_DECAY}}
ARCH=${ARCH:-adapter}
GNN_USE_INIT=${GNN_USE_INIT:-0}
# the official training loops do not clip gradients; the adapter keeps its former default
if [[ "${ARCH}" == "official_large" ]]; then GRAD_CLIP=${GRAD_CLIP:-0}; else GRAD_CLIP=${GRAD_CLIP:-1.0}; fi

# Metric: auto -> ROC-AUC (model selection, per-run output and mean/std) on questions / minesweeper /
# tolokers (and ogbn-proteins); accuracy on the other single-label datasets.
METRIC=${METRIC:-auto}

# SGFormer paper assumes a symmetric adjacency. Set TO_UNDIRECTED=0 if you want to preserve a directed input graph.
TO_UNDIRECTED=${TO_UNDIRECTED:-1}
FEATURE_NORM=${FEATURE_NORM:-none}

# Full-graph mode is the faithful default for medium graphs and ogbn-arxiv in the paper.
# Use TRAIN_MODE=batch / EVAL_MODE=batch when the graph cannot fit into GPU memory.
TRAIN_MODE=${TRAIN_MODE:-full}
EVAL_MODE=${EVAL_MODE:-full}
BATCH_SIZE=${BATCH_SIZE:-20000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-50000}

PYTHON=${PYTHON:-python}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "[SGFormer] dataset=${DATASET} data_dir=${DATA_DIR} out_root=${OUT_ROOT} runs=${RUNS} seed_start=${SEED}"
echo "[SGFormer] hyperparameters=${SRC} metric=${METRIC} train_mode=${TRAIN_MODE} eval_mode=${EVAL_MODE} hidden=${HIDDEN_DIM} gcn_layers=${GCN_LAYERS} trans_layers=${TRANS_LAYERS} alpha=${ALPHA} beta=${BETA}"
echo "[SGFormer] arch=${ARCH} gnn_use_init=${GNN_USE_INIT} grad_clip=${GRAD_CLIP}"
echo "[SGFormer] lr=${LR} dropout(trans/gnn)=${DROPOUT}/${GNN_DROPOUT} weight_decay(trans/gnn)=${WEIGHT_DECAY}/${GNN_WEIGHT_DECAY} batch_size=${BATCH_SIZE}"
mkdir -p "${OUT_ROOT}"

for ((i=0; i<RUNS; i++)); do
  CUR_SEED=$((SEED + i * SEED_STEP))
  RUN_OUT="${OUT_ROOT}/seed_${CUR_SEED}"
  echo "========== SGFormer run $((i+1))/${RUNS}; seed=${CUR_SEED}; out=${RUN_OUT} =========="
  ${PYTHON} -m sgformer_node.train_sgformer \
    --data_dir "${DATA_DIR}" \
    --dataset "${DATASET}" \
    --out_dir "${RUN_OUT}" \
    --seed "${CUR_SEED}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --eval_every "${EVAL_EVERY}" \
    --patience "${PATIENCE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --gcn_layers "${GCN_LAYERS}" \
    --trans_layers "${TRANS_LAYERS}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --dropout "${DROPOUT}" \
    --gnn_dropout "${GNN_DROPOUT}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --gnn_weight_decay "${GNN_WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --metric "${METRIC}" \
    --to_undirected "${TO_UNDIRECTED}" \
    --feature_norm "${FEATURE_NORM}" \
    --train_mode "${TRAIN_MODE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_mode "${EVAL_MODE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --arch "${ARCH}" \
    --gnn_use_init "${GNN_USE_INIT}" \
    ${EXTRA_ARGS:-}   # e.g. EXTRA_ARGS="--gnn_use_residual 0" for diagnostics
done

echo "========== Aggregate SGFormer repeated runs =========="
${PYTHON} -m sgformer_node.aggregate_repeats \
  --root "${OUT_ROOT}" \
  --runs "${RUNS}" \
  --seed "${SEED}" \
  --seed_step "${SEED_STEP}"
