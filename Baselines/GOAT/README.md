# GOAT under the shared comparison protocol

GOAT (Kong et al., ICML 2023) runs here through `../common/protocol.py`, like GCFormer, SwapGT, HubGT, GT-SNT and Polynormer. Data, splits, seeds, early stopping, checkpoint selection, metrics and the cost columns therefore match the other baselines.

```bash
cd Baselines/GOAT
bash run_goat.sh tolokers                      # 5 seeds, hyperparameters from the case table in run_goat.sh
DEVICE=cuda:0 bash run_goat.sh ogbn_products
python test_goat.py                            # checks against official/ (needs torch_geometric + einops, CPU)
```

| File | Role |
|---|---|
| `goat_npz.py` | Protocol adapter: data, node2vec encodings, loaders, training/evaluation closures for `fit` |
| `goat_model.py` | `official/large_model.py` without torch_sparse / PyG / einops; identical outputs and gradients (test_goat.py) |
| `goat_sampler.py` | `official/sampler.py` LocalSampler semantics (sample_adj rounds over all collected nodes), vectorised over the batch |
| `node2vec_pe.py` | Positional encodings with the settings of `official/pos_enc/*_node2vec.py`, cached in `cache/` |
| `run_goat.sh` | Per-dataset hyperparameters (official README commands or documented fallback) and protocol defaults |
| `official/` | Unmodified upstream files (commit in `UPSTREAM.txt`) |
| `legacy_bilgt/` | Previous BILGT runner, kept for traceability; not used |

## What changed relative to the previous runner (`legacy_bilgt/`)

- Data, runs, early stopping (30 checks, every 5 epochs), selection of the best validation checkpoint, a single test evaluation, ROC-AUC on Minesweeper/Tolokers and the timing/memory definitions now come from `common/protocol.py`. Previously the runner had its own loader and loop, reported accuracy only, and took the test score at the best validation check.
- Sampler: the official `sample_adj` rounds draw neighbours for every node collected so far. The previous port drew them only for the latest hop, so hop 2 and hop 3 were built from fewer nodes. It was also a pure-Python loop per node.
- Positional encodings: node2vec as in the official scripts. The previous fallback, a random projection of the features, is not GOAT's input.
- Binary datasets use a single logit with BCE, as `train_bi`/`test_bi` do for all binary datasets of the official script.
- The previous launcher ignored its `<data_dir>` argument (default `/mnt/sda/...` instead of `/mnt/sdb/...`), and its README and nohup script called a missing `run_goat_baseline.sh`.

## Remaining differences from the official script

- Symmetric graph without self loops for every dataset, including Arxiv-year, which `arxiv_ERM_ns.py` keeps directed. This is the input every method receives.
- Epoch budget and stopping follow the protocol (upper bound 1000, early stopping) instead of a fixed 500 epochs.
- node2vec random walks are drawn on the training device and use the same transition rule as torch_cluster (p = q = 1).
