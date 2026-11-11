> Release note: this is retained adapter documentation from the supplied source tree. The bundled dependency-light model is a project implementation of the paper design. Upstream training code is not included in this folder. See [the current registry](../registry.json) and [baseline documentation](../../docs/baselines.md) for entrypoints and provenance.

# GraphGPS baseline adapted for BILGT comparison

This package starts from the uploaded official `GraphGPS-main` source and adds a dependency-light BILGT adapter for **single-graph transductive node classification**:

- `npz_adapter/graphgps_node_baseline.py`
- `run_graphgps.sh`

The adapter follows the key GraphGPS design: each layer computes a **local real-edge MPNN branch** and a **global attention branch** in parallel, sums them, and applies a 2-layer FFN. For large transductive graphs, the default launcher uses a Performer-style linear global attention branch to avoid materializing the `N x N` attention matrix.

## Supported input format

Put the following files under `${DATA_ROOT}/${DATASET}` or pass a direct `${DATA_DIR}`:

```text
feat.npy / features.npy / x.npy
labels.npy / label.npy / y.npy
adj.npz                  # scipy sparse square adjacency
# OR edge_index.npy / edges.npy with shape [2,E] or [E,2]
train.npy / train_idx.npy / train_mask.npy
val.npy   / val_idx.npy   / valid.npy / valid_idx.npy / val_mask.npy
test.npy  / test_idx.npy  / test_mask.npy
```

A single `data.npz`, `graph.npz`, or `dataset.npz` bundle is also supported if it contains equivalent keys such as `x`, `y`, `edge_index`/`adj`, `train`, `val`, and `test`.

## Basic command

```bash
conda activate dgl_env   # or your project environment
DATA_ROOT=/mnt/sdb/changaf25/AURA_Former/datasets \
DEVICE=cuda:0 \
RUNS=5 \
bash run_graphgps.sh pubmed
```

For a direct directory:

```bash
DATA_DIR=/mnt/sdb/changaf25/AURA_Former/datasets/flickr \
DEVICE=cuda:1 \
bash run_graphgps.sh flickr
```

## Output format

The script runs five seeds by default: `42, 43, 44, 45, 46`.  It prints one line for each run and a final averaged table:

```text
Test Acc(%)    Tepoch(s)    Mem.(MB)
xx.xxxx±x.xxxx yy.yyyy      zz.zz
```

A CSV summary is written to:

```text
results_bilgt_graphgps/<dataset>/graphgps_<dataset>_summary.csv
```

## Dataset-aware defaults in the launcher

The paper benchmarks GraphGPS mainly on graph-level tasks, so for BILGT node-classification datasets the launcher uses fair-comparison defaults close to the user's existing baselines:

```text
hidden=128, layers=2, heads=4, dropout=0.5,
epochs=300, patience=100, runs=5, seed=42
```

Dataset-specific behavior:

- `cora`, `citeseer`: dense `Transformer` global attention unless `GLOBAL_MODEL` is overridden.
- `pubmed`, `corafull`, `flickr`, `github`, `ogbn-arxiv`, `tolokers`, `questions`: linear/Performer-style global attention by default to avoid quadratic attention memory.
- Large datasets use `lr=0.001` and `weight_decay=1e-4` unless manually overridden.

All settings can be overridden through environment variables, for example:

```bash
GLOBAL_MODEL=transformer HIDDEN=64 LR=5e-4 DROPOUT=0.2 bash run_graphgps.sh cora
```

## Notes

1. This adapter does not remove or rewrite the original GraphGPS repository. It adds a BILGT-compatible entry point because the official GraphGym pipeline targets its own PyG/OGB loaders and graph-level benchmarks.
2. The adapter intentionally avoids `torch_geometric`, `torch_scatter`, and `performer-pytorch`, which are often the fragile dependencies in a shared baseline environment.
3. Optional Laplacian PE is available through `LAP_PE_K=4`, but it can be expensive on large single graphs. Degree-based local structural encoding is enabled by default.
