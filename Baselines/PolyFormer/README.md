# PolyFormer Node Baseline for BiLGT-format Data

This package adapts the released **PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer** source code to the unified node-classification format used in our project.

The implementation is based on the official GitHub source you provided (`PolyFormer-main.zip`), especially:

- `node_classification/training.py`
- `node_classification/mymodels.py`
- `node_classification/layers/PolyFormerBlock.py`
- `node_classification/node_classification.sh`
- `node_classification_large_graph/training_batch.py`
- `node_classification_large_graph/node_classification_large_graph.sh`

It is **not** an unrelated reimplementation. The model block, PolyAttn structure, token-wise MLP, Q/K attention over polynomial tokens, bias decay, FFN, and optimizer parameter groups are kept aligned with the released source. The main changes are dataset I/O, split handling, metric logging, repeated-run aggregation, and memory-friendly batching after polynomial-token construction.

## 1. Input format

Every dataset directory must contain:

```text
adj.npz
feat.npy
label.npy
train.npy
val.npy
test.npy
```

`train.npy`, `val.npy`, and `test.npy` are read directly. The code never creates a new split.

Both integer-index split arrays and boolean masks are accepted.

## 2. Quick start

```bash
unzip polyformer_node_baseline.zip
cd polyformer_node_baseline

DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets \
DEVICE=cuda:0 \
RUNS=5 \
bash run_polyformer.sh citeseer
```

For one of the common datasets:

```bash
DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets bash scripts/run_ogbn_arxiv.sh
DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets bash scripts/run_tolokers.sh
DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets bash scripts/run_flickr.sh
```

For a custom directory:

```bash
DATASET=custom \
DATA_DIR=/mnt/sdb/changaf25/sdust/datasets/my_dataset \
RUNS=5 \
bash run_polyformer.sh custom
```

## 3. Five-run output

By default, the script runs seeds:

```text
42, 43, 44, 45, 46
```

Each seed writes:

```text
outputs/polyformer_node/<dataset>/seed_<seed>/config.json
outputs/polyformer_node/<dataset>/seed_<seed>/log.tsv
outputs/polyformer_node/<dataset>/seed_<seed>/summary.json
outputs/polyformer_node/<dataset>/seed_<seed>/best_model.pt
```

After all runs, it writes:

```text
outputs/polyformer_node/<dataset>/repeat_summary.json
outputs/polyformer_node/<dataset>/repeat_summary.tsv
```

The aggregate summary includes exactly the fields needed for a paper comparison:

```text
test_score_mean_percent
test_score_std_percent
val_score_mean_percent
val_score_std_percent
best_epoch_mean
best_epoch_std
stop_epoch_mean
best_epoch_peak_mem_mb_mean
best_epoch_peak_mem_mb_std
best_epoch_train_s_mean
best_epoch_train_s_std
best_epoch_complete_s_mean
best_epoch_complete_s_std
precompute_bases_s_mean
```

`best_epoch_complete_s_mean` is the average wall-clock time of the convergence epoch, including train + validation/test evaluation for that epoch. `best_epoch_peak_mem_mb_mean` is the average maximum CUDA memory allocated during the convergence epoch.

## 4. Official and self-set hyperparameters

The file `polyformer_node/configs.py` contains two groups of configurations.

### Official configs transcribed from the released source

For Table-5 datasets, both official `mono` and official `cheb` configs are included:

```text
citeseer
pubmed
coauthor_cs / cs
coauthor_physics / physics
chameleon_filtered
squirrel_filtered
minesweeper
tolokers
roman_empire
questions
```

For Table-6 large datasets, the official large-graph configs are included:

```text
twitch_gamer
pokec
ogbn_arxiv
ogbn_papers100m / Paper100m
```

`BASE=auto` defaults to the official `mono` configuration when available. To run the official Chebyshev variant:

```bash
BASE=cheb bash run_polyformer.sh pubmed
```

### Self-set configs for our common datasets not covered by the official scripts

The following datasets were not present in the official PolyFormer shell scripts, so this package provides conservative self-set configs based on the nearest official family:

```text
cora
facebook_large
Flickr
reddit
ogbn_products
ogbn_mag
ethanol
amazon_computers
amazon_photo
wikics
github
```

Aliases (`OFFICIAL_ALIAS` in configs.py): `dblp` and `corafull` use the official PubMed command (config_source
`fallback:pubmed`; corafull previously fell back to the generic `DEFAULT`), `lastfm_asia` the official Coauthor-CS
command (`fallback:coauthor_cs`).

These can and should be overridden when you do final tuning. Example:

```bash
HIDDEN=256 K=6 LR=0.0005 ATTN_LR=0.001 bash run_polyformer.sh cora
```

## 5. Important knobs

```bash
# repeated runs
RUNS=5
SEED=42
SEED_STEP=1

# device
DEVICE=cuda:0

# official base family
BASE=mono     # mono / cheb / auto

# training
EPOCHS=2000
PATIENCE=250
EVAL_EVERY=1
TRAIN_MODE=batch   # batch is memory-friendly after polynomial bases are built
BATCH_SIZE=10000
EVAL_BATCH_SIZE=50000

# metrics
METRIC=auto    # auto / acc / rocauc
FORCE_BINARY=0 # set 1 for scalar 0/1 ROC-AUC datasets

# graph preprocessing
TO_UNDIRECTED=1
FEATURE_NORM=none

# polynomial token cache
CACHE_DIR=outputs/polyformer_node/<dataset>/basis_cache
CACHE_MODE=memmap
FORCE_RECOMPUTE_BASES=0
```

## 6. Why batch mode is valid here

PolyFormer first constructs per-node polynomial tokens:

```text
H_i = [h_i^0, h_i^1, ..., h_i^K]
```

After these tokens are built, PolyAttn computes attention **within each node's own polynomial-token sequence**, not across all nodes. Therefore, slicing nodes into batches after polynomial-token construction does not change the forward computation for a node. It only reduces GPU memory. This is different from ordinary full-graph attention where batching would change the visible context.

For strict alignment with the original small-graph training loop, set:

```bash
TRAIN_MODE=full bash run_polyformer.sh citeseer
```

For large datasets, keep `TRAIN_MODE=batch`.

## 7. Environment

The code intentionally avoids PyG/DGL dataset loaders and reads only NumPy/SciPy files. Required Python packages:

```bash
pip install numpy scipy scikit-learn torch
```

If your project conda environment already runs the SGFormer baseline, it should normally have everything needed here.

## 8. Notes for paper reporting

Recommended report columns:

```text
dataset
method
base
metric
test_score_mean_percent
test_score_std_percent
best_epoch_mean
stop_epoch_mean
best_epoch_peak_mem_mb_mean
best_epoch_complete_s_mean
config_source
```

For datasets whose config source is `self-set`, state in the appendix that PolyFormer did not provide official hyperparameters for that dataset and that the shown settings follow the nearest official dataset family.
