> Release note: this is retained adapter documentation from the supplied source tree. The bundled dependency-light model is a project implementation of the paper design. Upstream training code is not included in this folder. See [the current registry](../registry.json) and [baseline documentation](../../docs/baselines.md) for entrypoints and provenance.

# BILGT-compatible Exphormer baseline

This folder keeps the original Exphormer/GraphGPS code and adds a project-compatible full-graph node-classification baseline:

- `exphormer_npz.py`: standalone PyTorch implementation of `GCN + Exphormer` for a single transductive graph.
- `run_exphormer.sh`: dataset-aware launcher, five runs by default, seeds `42,43,44,45,46`.

The standalone script intentionally avoids `torch_geometric` and `torch_scatter`, because the project environment often already has PyTorch/DGL but may not have all PyG binary wheels. The model follows the official Exphormer attention pattern: sparse attention over input/local edges, random regular expander edges, and self-loops; the default large-transductive setting uses the paper's `GCN+Exphormer` design with no virtual nodes.

## Expected input data format

The dataset directory can be either `${DATA_ROOT}/${DATASET}` or passed directly through `DATA_DIR=/path/to/dataset`.

Supported feature file names:

```text
x.npy, feat.npy, features.npy, node_feat.npy, node_features.npy,
x.npz, feat.npz, features.npz, node_feat.npz, x.pt, features.pt
```

Supported label file names:

```text
y.npy, label.npy, labels.npy, node_label.npy, node_labels.npy,
y.npz, labels.npz, y.pt, labels.pt
```

Supported graph file names:

```text
adj.npz, adjacency.npz, graph.npz, edge_index.npy, edges.npy,
edge_index.npz, edge_index.pt, edges.pt
```

Supported split files:

```text
train.npy / val.npy / test.npy
train_idx.npy / val_idx.npy / test_idx.npy
train_mask.npy / val_mask.npy / test_mask.npy
```

or a named split npz such as `split.npz`, `splits.npz`, or `split_idx.npz` with keys such as `train`, `val`, `valid`, and `test`.

## Quick start

```bash
cd Exphormer-main
conda activate dgl_env

DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets \
DEVICE=cuda:0 \
bash run_exphormer.sh pubmed
```

Run on an explicit dataset directory:

```bash
DATA_DIR=/mnt/sdb/changaf25/sdust/datasets/flickr \
DEVICE=cuda:1 \
bash run_exphormer.sh flickr
```

The launcher prints each run and a final summary in the same fields used by the project tables:

```text
Test Acc(%)    Tepoch(s)    Mem.(MB)
```

A CSV summary is written to:

```text
results_bilgt_exphormer/<dataset>/summary.csv
```

## Dataset-aware hyperparameters

When a dataset is covered by the Exphormer paper's large transductive table, the launcher uses those settings:

| Dataset | Layers | Hidden | Heads | Dropout | LR | Epochs | Expander degree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | 3 | 96 | 2 | 0.3 | 0.01 | 600 | 6 |
| Amazon Computer | 4 | 80 | 2 | 0.4 | 0.001 | 150 | 6 |
| Amazon Photo | 4 | 64 | 2 | 0.4 | 0.001 | 100 | 6 |
| Coauthor CS | 4 | 72 | 2 | 0.4 | 0.001 | 70 | 6 |
| Coauthor Physics | 4 | 72 | 2 | 0.4 | 0.001 | 70 | 6 |

For project datasets not in the paper, such as `cora`, `citeseer`, `pubmed`, `corafull`, `flickr`, `github`, and `tolokers`, the launcher uses the common comparison setting:

```text
hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=256,
dropout=0.5, lr=0.01, epochs=300, patience=100,
runs=5, seeds=42..46, expander_degree=6
```

Every parameter can be overridden through environment variables. Examples:

```bash
LR=0.001 EPOCHS=500 PATIENCE=150 bash run_exphormer.sh flickr cuda:0
HIDDEN_DIM_OVERRIDE=128 NUM_LAYERS_OVERRIDE=4 bash run_exphormer.sh ogbn_arxiv cuda:1
EXTRA_ARGS="--feature_norm row --norm layer" bash run_exphormer.sh pubmed cuda:0
```

## Notes on reproducibility

1. Validation-gated testing is used: the reported `Test Acc(%)` is the test accuracy at the best validation epoch.
2. `Tepoch(s)` is the average full epoch wall-clock time for the executed epochs of a run.
3. `Mem.(MB)` is `torch.cuda.max_memory_allocated` for CUDA runs; on CPU it falls back to process RSS.
4. The script reads the provided train/validation/test splits. It does not create new splits by default.
5. The model treats the input graph as bidirectional by default, matching Exphormer's statement that its sparse interaction construction uses bidirectional edges. Use `EXTRA_ARGS="--keep_directed"` to disable this.
