# VCR-Graphormer BILGT Baseline Reproduction

This package adapts the official **VCR-Graphormer** code to the BILGT-style baseline protocol.
It keeps the original `TransformerModel` architecture and replaces the original DGL dataset entry
points with a custom loader for your project's fixed train/val/test splits.

## 1. Data format

Each dataset directory should contain the same files used by your method:

```text
<dataset>/
  feat.npy              # or features.npy / x.npy / feat.npz / features.npz
  labels.npy            # or label.npy / y.npy
  adj.npz               # scipy sparse adjacency; alternatively edge_index.npy / edges.npy
  train.npy             # node indices or boolean mask
  val.npy               # node indices or boolean mask
  test.npy              # node indices or boolean mask
```

The code **does not re-split** data. It directly uses the supplied split files.

## 2. Main command

```bash
bash run_vcr_baseline.sh <dataset> <data_dir> [device]
```

Example:

```bash
bash run_vcr_baseline.sh pubmed /mnt/sdb/changaf25/sdust/datasets/pubmed 0
```

The default protocol is 5 runs with seeds `42,43,44,45,46`:

```bash
NUM_RUNS=5 SEED_START=42 bash run_vcr_baseline.sh flickr /mnt/sdb/changaf25/sdust/datasets/flickr 0
```

## 3. Output metrics

The final summary line matches your baseline table format:

```text
Test Acc(%)    训练平均Tepoch(s)    训练最大Mem.(MB)
```

Per-run and averaged results are saved under `results_vcr_bilgt/` as both JSON and CSV.
The reported training epoch time excludes one-time offline tokenization by default, because VCR-Graphormer
explicitly moves graph tokenization offline. Add `--include_preprocess_in_time` through `EXTRA_ARGS` when you
want to amortize tokenization into run 1.

## 4. Parameter policy

`run_vcr_baseline.sh` uses the official demo hyperparameters when the dataset appears in the VCR-Graphormer
paper/source:

| Dataset | hops | structure supernodes | structure tokens | content tokens | hidden | heads | layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| PubMed | 7 | 10 | 16 | 13 | 512 | 8 | 1 |
| CoraFull | 3 | 20 | 5 | 5 | 512 | 8 | 1 |
| Computer | 3 | 10 | 11 | 16 | 512 | 8 | 1 |
| Photo | 3 | 10 | 5 | 5 | 512 | 8 | 1 |
| CS | 3 | 20 | 5 | 5 | 512 | 8 | 3 |
| Physics | 4 | 50 | 10 | 20 | 128 | 8 | 1 |
| Reddit | 6 | 400 | 20 | 20 | 512 | 8 | 1 |
| Aminer | 16 | 100 | 5 | 9 | 512 | 16 | 3 |
| Amazon2M / ogbn-products | 10 | 300 | 5 | 5 | 512 | 1 | 1 |
| Squirrel | 5 | 10 | 10 | 1 | 512 | 8 | 1 |
| Actor | 2 | 10 | 10 | 1 | 512 | 1 | 1 |
| Texas | 2 | 10 | 5 | 5 | 512 | 8 | 1 |
| arxiv-year | 8 | 500 | 16 | 6 | 128 | 8 | 1 |

For datasets not covered by the paper, such as Flickr and GitHub, the script uses conservative VCR-style
settings (`hops=3`, `structure/content tokens=10`, `hidden=256`) so the method remains scalable. You can
override every hyperparameter by environment variables, for example:

```bash
HOPS=4 STRUCT_SUPER=100 STRUCT_TOK=16 CONTENT_TOK=8 HIDDEN=512 HEADS=8 \
  bash run_vcr_baseline.sh github /path/to/github 0
```

## 5. Offline tokenization and cache

VCR-Graphormer first builds token lists consisting of:

1. target node feature token;
2. local diffusion tokens from normalized graph propagation;
3. structure-aware virtual-connection PPR tokens;
4. content-aware virtual-connection PPR tokens.

The token cache is stored in:

```text
<data_dir>/VCRGraphormer_cache/
```

The five runs share the same token cache. Rebuild it with:

```bash
EXTRA_ARGS="--force_retokenize" bash run_vcr_baseline.sh pubmed /path/to/pubmed 0
```

## 6. METIS / partition fallback

The paper uses METIS-like graph partitioning for structure-aware supernodes. This package tries DGL/METIS when
available. If DGL/METIS is unavailable in your conda environment, it falls back to a deterministic balanced BFS
partition so the code can still run. For strictest reproduction, install a DGL build with METIS support and run:

```bash
PARTITION_METHOD=metis bash run_vcr_baseline.sh pubmed /path/to/pubmed 0
```

## 7. nohup examples

```bash
DATA_ROOT=/mnt/sdb/changaf25/sdust/datasets DEVICE=0 bash scripts/nohup_vcr_examples.sh
```

This starts one nohup process per listed dataset and writes logs to `nohup_logs_vcr/`.

## 8. Files added or changed

Added:

- `vcr_bilgt_data.py`: BILGT fixed-split dataset loader.
- `vcr_tokenization.py`: dependency-light VCR tokenization with cache.
- `vcr_bilgt_baseline.py`: 5-run training/evaluation entry point.
- `run_vcr_baseline.sh`: dataset-aware launcher.
- `scripts/nohup_vcr_examples.sh`: nohup launch examples.
- `README_BILGT_VCR.md`: this guide.

Changed:

- `model.py`: patched a batch-size-one `squeeze()` issue to preserve the batch dimension during evaluation.

