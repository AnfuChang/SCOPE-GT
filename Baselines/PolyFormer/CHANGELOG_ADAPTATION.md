# PolyFormer adaptation change log

## Kept from official source

- PolyFormer block structure: `PolyAttn + FFN`.
- Per-node polynomial-token attention over `K+1` tokens.
- Token-wise MLP before Q/K projection.
- `tanh(QK^T/sqrt(d))` attention score transform.
- Head-wise polynomial-order bias scale with `(j+1)^(-q)` decay.
- Residual connection after PolyAttn and FFN.
- Final `sum` over polynomial tokens and two-layer classifier.
- Adam optimizer with separate learning rate / weight decay for `attnmodule` parameters.
- Official `mono` and `cheb` polynomial bases.

## Changed for our project

1. Removed PyG/DGL dataset loaders.
   - New loader reads only `adj.npz`, `feat.npy`, `label.npy`, `train.npy`, `val.npy`, `test.npy`.

2. Removed all internal random split generation.
   - The provided split files are used directly.

3. Added repeat runner.
   - `run_polyformer.sh` runs 5 seeds by default and writes aggregate JSON/TSV.

4. Added paper-ready convergence statistics.
   - Test score mean/std at best validation epoch.
   - Mean/std of best epoch.
   - Mean/std of peak CUDA memory in the best epoch.
   - Mean/std of training and complete wall-clock time in the best epoch.

5. Added official config registry.
   - Hyperparameters from official `node_classification.sh` and `node_classification_large_graph.sh` are transcribed into `polyformer_node/configs.py`.
   - Datasets not covered by the official scripts use clearly marked `self-set` configs.

6. Added polynomial basis cache.
   - Bases can be stored under `CACHE_DIR` and loaded as memory maps with `CACHE_MODE=memmap`.

7. Added exact node-wise batch mode after polynomial-token construction.
   - This is memory-friendly because PolyFormer attention is within each node's polynomial-token sequence.
   - Set `TRAIN_MODE=full` to mimic the official small-graph full-forward loop more strictly.

## What is not changed

- No label information is used for preprocessing.
- No extra loss, no augmentation, no edge features, no graph positional encoding, and no tuning tricks were added.
- The algorithm remains PolyFormer/PolyAttn, not a new method.
