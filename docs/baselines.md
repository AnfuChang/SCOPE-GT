# Comparison implementations and commands

The compatible NPZ adapters are bundled in `Baselines/`. The model, sampler,
tokenizer and training adapter used by an experiment together define its
implementation. Cloning an upstream repository alone does not recreate this
interface. `SCOPE_GT_BASELINES_ROOT` is an optional replacement for the bundled
tree when using the efficiency, budget, motivation and transfer harnesses.

## Implementation provenance

An **upstream implementation** is the authors' original source at an identified
revision. An **adapted implementation** imports or ports that model while changing
data loading, split handling, device operations or the experiment loop. A
**paper-guided reimplementation** implements the published design separately.
These descriptions identify code provenance; they do not imply identical
accuracy, numerical behavior or training speed between implementations.

| Method key | Bundled entry under `Baselines/` | Source and changes |
|---|---|---|
| `graphgps` | `GraphGPS/npz_adapter/graphgps_node_baseline.py` | Dependency-light project implementation of the local/global GraphGPS design, with NPZ loading and dataset recipes; not the upstream GraphGym training pipeline. |
| `exphormer` | `Exphormer/exphormer_npz.py` | Dependency-light project implementation of GCN + sparse local/expander attention; not the upstream GraphGym pipeline. |
| `sgformer` | `SGFormer/sgformer_node/train_sgformer.py` | Project NPZ adaptation with small-graph and large-graph architectures; exact upstream revision was not retained. |
| `gtsnt` | `GT_SNT/gtsnt_npz.py` | Imports the upstream model from `official/nn`; adapts NPZ data, training, edge handling and selected operations. Upstream `9b2962d83c844bc26865b2f81bb62b907fc66152`. |
| `nagphormer` | `NAGphormer/baseline_suite.py` | Project NPZ implementation of hop tokens and attention, with NrAug disabled in `run_nagphormer.sh`; exact upstream revision was not retained. |
| `polyformer` | `PolyFormer/polyformer_node/train_polyformer.py` | Adaptation from a supplied upstream archive; preserves polynomial bases and attention, changes loading and training protocol. Exact archive revision was not retained. |
| `gophormer` | `Gophormer/gophormer_npz.py` | **Dense paper-guided reimplementation** with ego-graph sampling, proximity bias, global tokens, consistency loss and multi-sample prediction. It is separate from [the upstream DGL implementation](https://github.com/AndyJZhao/Gophormer). Four training samples and five inference samples are explicit in the comparison recipes. |
| `ansgt` | `ANS_GT/ans_gt_npz.py` | Project NPZ implementation of adaptive context sampling and attention; exact upstream revision was not retained. |
| `vcr` | `VCR_Graphormer/vcr_bilgt_baseline.py` | Adapted architecture with project NPZ tokenization. The actual partition method and tokenizer settings must be retained, particularly when a METIS dependency is absent. Exact upstream revision was not retained. |
| `gcformer` | `GCFormer/gcformer_npz.py` | Upstream model and schedule with NPZ/training adapter. Upstream `75173ea85884e215d5b7b6cbec528c5bb907bd53`. |
| `nlgt` | `NLGT/nlgt_npz.py` | Project NPZ implementation with sampler and optional neighbor-label inputs; exact upstream revision was not retained. Classification and resource recipes have distinct label-input policies. |
| `swapgt` | `SwapGT/swapgt_npz.py` | Upstream model with device/shape fixes and NPZ/training adapter. Upstream `92d727ac51e367e47dcc3363a2f0bc9ba85704bb`. |
| `hubgt` | `HubGT/hubgt_npz.py` | Upstream model and hierarchical index with NPZ/training adapter. Upstream `5909099c4bfd45f0575ac989e5e92bdb883f2579`. |
| `goat` | `GOAT/goat_npz.py` | Ports upstream model operations to plain PyTorch and imports upstream vector quantization; NPZ/training adapter. Upstream `f29bfcd2e717d62c498616916972482c9f7a215b`. |

Machine-readable paths and recorded upstream revisions are in
[`registry.json`](../Baselines/registry.json). Existing `UPSTREAM.txt` and license
files are retained beside the sources. The repository's top-level license does
not replace third-party licenses. GT-SNT's supplied upstream record states that
no upstream license file was found; that source record remains attached.

[`source_inventory.json`](../Baselines/source_inventory.json) identifies this
packaged source snapshot by SHA256 for each file and for the inventory. It also
retains source-file hashes before the release integration changes. An upstream
commit identifies the upstream origin, not the revision of the adapter. For
sources without a recorded upstream commit, the file hashes identify the
packaged adapter exactly. They do not recover an unrecorded historical server
revision or establish that current files produced every reported measurement.

## Dependencies

Use Linux, Python 3.10 and the core package versions in `requirements.txt`.
Install the appropriate PyTorch CUDA build before the remaining requirements:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-baselines.txt
```

`requirements-baselines.txt` specifies installation requirements, rather than a
recovered historical environment lock. A run saves its actual package versions.
Most NPZ adapters use PyTorch, NumPy and SciPy directly, without PyG or DGL.
Additional requirements are method-specific:

- GT-SNT imports `torch_geometric` and `yaml`: install `torch-geometric` and
  `PyYAML`. See its adapter docstring for the optional `torch_sparse` paths.
- HubGT requires Cython, a Linux C++ compiler and OpenMP. Build the extension in
  the target environment with `python -m SCOPE_GT_transfer.build_hubgt`.
- VCR uses tqdm and numba. Its METIS partition option additionally requires a
  compatible DGL installation; `auto` may select a different available
  partitioner. Preserve the resulting tokenizer configuration in the run record.
- Original upstream scripts retained for attribution can have additional
  dependencies; the supported NPZ entrypoints are the ones listed above.

## Classification commands and effective configuration

From the repository root, use the wrapper to keep outputs and provenance:

```bash
export DATA_BASE=/path/to/processed/datasets
python scripts/run_baseline.py gophormer wikics --device cuda:0 --runs 5
python scripts/run_baseline.py goat ogbn_products --device cuda:0 --runs 5
python scripts/run_baseline.py swapgt lastfm_asia --dry-run
```

Replace the method key with any row in the table. The wrapper selects the
corresponding `run_*.sh` in `Baselines/<method>/`, sets the data/output directory
and selected interpreter, and keeps the recipe's model settings. Use `--data-dir`
for an exact processed dataset directory, `--python` for another interpreter,
`--out` for a fresh output directory, and `--seed-start` to select the first seed
(classification recipes default to 42 through 46). Environment overrides supported
by a method's launcher remain available. `--dry-run` prints a launch plan; it does
not claim to have parsed the model's full configuration or run training.

The **effective configuration** is the parser defaults plus the dataset recipe
plus explicit overrides, rather than just the contents of a preset file. Each
training entry saves these parsed arguments to
`provenance/effective_config_<pid>.json` when run through the wrapper. The native
config/summary files additionally retain settings resolved after data loading.
The wrapper keeps `stdout.log`, `launch.json`, `inputs.json`, `source.json`,
`pip-freeze.txt`, launcher runtime information and the exit code. These records
are produced at execution time; they are not bundled experimental results.

Standalone classification logs can include validation in their older timing
summaries. Use the measurement harness below for the paper's training-only
resource windows, not an arbitrary timing field printed by a classification
launcher.

### ANS-GT runtime budget

In the classification comparison, **OOT means that a single training epoch
exceeds 43,200 seconds (12 hours)**. The ANS-GT classification launcher enforces
this limit by default, separately for each epoch and seed. The timed region
contains training data iteration, online sampling, forward/backward computation
and optimizer updates. CUDA work is synchronized at the boundaries. One-off
loading/preprocessing, validation/test evaluation and between-epoch sampler
updates are outside this training-epoch limit.

```bash
for dataset in arxiv_year pokec ogbn_products; do
  python scripts/run_baseline.py ansgt "$dataset" --device cuda:0 \
    --runs 5 --seed-start 42 --epoch-timeout-seconds 43200 \
    --out "results/baselines/ansgt_epoch12h/${dataset}"
done
```

The native shell uses `EPOCH_TIMEOUT_SECONDS=43200` and forwards it as
`--epoch-timeout-seconds` to the ANS-GT trainer. Set it to 0 explicitly to
disable the limit. The watchdog also terminates a silent or blocked training
epoch, writes `epoch_timeout_seed<seed>_epoch<epoch>.json` with the elapsed
training-epoch time and exits with code 124. When run through the classification
wrapper, `provenance/exit.json` records `status="epoch_timeout"` and links the
epoch record. Preserve unrelated failures as failures rather than labeling them
OOT or OOM.

The wrapper's optional `--timeout-seconds` remains a **whole-invocation** limit,
not an epoch limit. Its default is 0 (disabled), and it does not define
classification OOT. Other baseline recipes, efficiency measurements and transfer runs
retain their separate runtime protocols; the ANS-GT classification OOT cutoff
does not change them. Changing this release's watchdog does not generate new
training measurements.

## Efficiency and transfer comparisons

Effective resource recipes are in `SCOPE_GT_efficiency/config.py`, with shared
NLGT/SGFormer settings in `SCOPE_GT_budgets/profiles.json` and SCOPE-GT settings
in `SCOPE_GT/presets.json`. Transfer recipes are in `SCOPE_GT_transfer/config.py`
and `extra_backbones.py`. Their settings differ from standalone classification
recipes; do not substitute one experiment's config for another.

```bash
# Inspect resource configurations without starting training (core dependencies needed).
python -m SCOPE_GT_efficiency.run convergence --data-base "$DATA_BASE" \
  --convergence-dataset ogbn_products --dry-run

# Complete training-center traversal per epoch, all seven resource methods.
python -m SCOPE_GT_efficiency.run convergence --data-base "$DATA_BASE" \
  --convergence-dataset ogbn_products --out results/efficiency_products \
  --seeds 0 1 2 3 4 --device cuda:0
python -m SCOPE_GT_efficiency.run convergence --data-base "$DATA_BASE" \
  --convergence-dataset pokec --out results/efficiency_pokec \
  --seeds 0 1 2 3 4 --device cuda:0
bash scripts/run_efficiency.sh
bash scripts/run_transfer.sh
```

The efficiency worker records actual supervised centers and optimizer updates
for each epoch and rejects incomplete training-center coverage. Warmup and
validation are outside epoch time. A full-graph method may make one optimizer
update, while a mini-batch method makes multiple updates; equal center coverage
does not imply identical optimization. Resource runs retain effective `spec.json`,
source/data/runtime identities, per-epoch history and success/failure status.
Resource limits and OOM/timeout outcomes remain explicit.

The efficiency launcher's epoch, preparation and whole-job watchdogs default to
disabled (`--epoch-timeout 0 --preprocess-timeout 0 --job-timeout 0`), so complete
Pokec and Products jobs are not stopped after 3,600 or 7,200 seconds. Positive
values explicitly enable these watchdogs. Computers convergence separately retains
its 7,200-second observation window, counted from subprocess launch; it reports
`observation_limit` when that window ends. Set `--observation-seconds 0` to disable
that window. The observation window does not limit scaling, pilot or other-dataset
jobs. The interpreter/runtime probe retains its separate 60-second timeout.

## Processed-input identifiers

```bash
python scripts/input_manifest.py --data-root "$DATA_BASE/ogbn_products" \
  --out results/input_manifests/ogbn_products.json
```

The manifest records the exact six file names, byte sizes and SHA256 hashes for
`adj.npz`, `feat.npy`, `label.npy`, `train.npy`, `val.npy` and `test.npy`, plus
shapes, stored adjacency entries and split counts. `input_id` hashes that file
inventory. For an explicitly selected official mask, use `--split-suffix _split0`
through `_split9`. The baseline wrapper uses the unsuffixed fixed files, which
alias official split 0 for the four Platonov datasets. Hashes identify processed
inputs before method-specific normalization or tokenization. Those transforms
belong to the effective configuration. A dataset name and split percentages alone
do not identify the exact inputs.
