# Implementation and reproducibility notes

This document describes the presets and execution protocols used with SCOPE-GT. The canonical classification settings are in [`SCOPE_GT/presets.json`](../SCOPE_GT/presets.json); explicit CLI arguments override a preset. Sampling depth `max_hops` is distinct from attention depth `layers`.

The [experiment index](experiment-index.md) maps the paper studies to launchers, repetition units, and output files. The [mechanism benchmark notes](../SCOPE_GT_mechanisms/README.md) define the shared-encoder and exact-retrieval controls.

## Training and prediction

The default independent SCOPE-GT experiment seeds are **0, 1, 2, 3, 4** in the shell launchers and Python entry points. The default thread count is 6. Use `SEEDS="0"` with a shell launcher, or `--seeds 0` with a Python entry point, only when explicitly requesting a different run count. Classification, predictive ablations, depth, budget, native/SCMB transfer, transfer decomposition, mechanism benchmarks, batch-size sweeps and overlap sampling default to five seeds. Transfer controls must use the same seed list as their reference runs (see [transfer controls](transfer-decomposition.md)); a changed default does not regenerate historical results. Optional efficiency pilots use five separate seeds, 100 through 104. Standalone baseline classification recipes default to five runs with their documented seed start. Repeated epochs, inference views and profile iterations within a seed are separate measurement units.

The main reader has two attention layers and four heads, relation width 64, feed-forward width `2D`, and AdamW optimization with a cosine learning-rate schedule. A validation-selected checkpoint is used for final prediction. The training loss is supervised cross-entropy with the per-dataset label smoothing listed below. WikiCS additionally averages cross-entropy over two independently sampled training contexts and adds `0.5 * mean(sum((softmax(z0)-softmax(z1))**2))`; the other presets use one training context and no consistency term. This WikiCS setting is a dataset-specific training configuration, not a separate model module.

All classification presets average logits from three fixed contexts at validation and final prediction. Final prediction covers every node in the test mask. Per-epoch validation can use the configured fixed subset; this does not change final test coverage. The training and validation budgets in the table are numbers of centers, with 0 meaning the complete corresponding split. All classification presets use the complete training split; the 20,000-center cap on the three large datasets applies only to checkpoint-selection validation. Do not treat a budgeted epoch as a full-split traversal.

In particular, `arxiv_year`, `pokec` and `ogbn_products` set
`centers_per_epoch=0`: every training center is visited once per epoch, in shuffled
mini-batches. `val_max_centers=20000` only limits checkpoint-selection validation.
Explicit CLI overrides can change coverage and must be kept in the run record.

### Sampling and model presets

`W` falls back to `Nctx` when `walks` is null or absent. Attribute budget 0 means all retained supporting attributes; Tolokers disables path attributes but keeps connection statistics and background attributes. Each shared node encoder has an input projection followed by the listed number of residual MLP blocks (layer normalization, GELU and dropout).

| Dataset | Nctx / W / Hs | D | Encoder blocks | Hop shares | Background proposals | Path attrs | Feature normalization |
|---|---|---:|---:|---|---:|---|---|
| LastFM | 24 / 24 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | l1 |
| Minesweeper | 24 / 48 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | none |
| WikiCS | 32 / 48 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | none |
| Tolokers | 48 / 48 / 2 | 128 | 2 | 0.75 / 0.25 | 8 | off | none |
| Computers | 48 / 48 / 2 | 128 | 2 | 0.75 / 0.25 | 4 | on | auto |
| DBLP | 16 / 24 / 3 | 128 | 1 | 0.4 / 0.4 / 0.2 | 4 | on | l1 |
| CS | 16 / 32 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | l1 |
| PubMed | 16 / 16 / 2 | 64 | 1 | 0.75 / 0.25 | 4 | on | auto |
| CoraFull | 24 / 24 / 3 | 128 | 1 | 0.4 / 0.4 / 0.2 | 4 | on | auto |
| Facebook | 24 / 24 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | auto |
| Roman | 12 / 12 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | auto |
| Ratings | 16 / 16 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | auto |
| GitHub | 32 / 32 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | auto |
| Arxiv-year | 48 / 48 / 3 | 128 | 2 | 0.35 / 0.35 / 0.3 | 4 | on | auto |
| Pokec | 64 / 64 / 2 | 128 | 2 | 0.5 / 0.5 | 4 | on | auto |
| Products | 48 / 48 / 3 | 128 | 2 | 0.5 / 0.3 / 0.2 | 4 | on | auto |

### Optimization presets

| Dataset | Train B / eval B | LR | Weight decay | Dropout / input dropout | Epoch cap / patience | Label smoothing | Train / validation center budget | Eval interval |
|---|---|---|---|---|---|---:|---|---:|
| LastFM | 64 / 256 | 0.0005 | 0.001 | 0.3 / 0.5 | 50 / 50 | 0 | 0 / 0 | 1 |
| Minesweeper | 64 / 256 | 0.002 | 0.01 | 0.2 / 0 | 120 / 120 | 0 | 0 / 0 | 1 |
| WikiCS | 64 / 256 | 0.001 | 0.05 | 0.5 / 0.3 | 50 / 50 | 0.05 | 0 / 0 | 1 |
| Tolokers | 64 / 256 | 0.001 | 0.01 | 0.5 / 0 | 150 / 150 | 0 | 0 / 0 | 1 |
| Computers | 64 / 256 | 0.0005 | 0.01 | 0.4 / 0.4 | 50 / 50 | 0.05 | 0 / 0 | 1 |
| DBLP | 64 / 256 | 0.001 | 0.05 | 0.3 / 0.5 | 20 / 20 | 0.1 | 0 / 0 | 1 |
| CS | 64 / 256 | 0.001 | 0.05 | 0.4 / 0.5 | 20 / 20 | 0.1 | 0 / 0 | 1 |
| PubMed | 64 / 256 | 0.0005 | 0.01 | 0.5 / 0.2 | 40 / 40 | 0.05 | 0 / 0 | 1 |
| CoraFull | 64 / 256 | 0.001 | 0.05 | 0.3 / 0.5 | 20 / 20 | 0.1 | 0 / 0 | 1 |
| Facebook | 64 / 256 | 0.001 | 0.01 | 0.3 / 0.3 | 30 / 30 | 0 | 0 / 0 | 1 |
| Roman | 64 / 256 | 0.001 | 0.05 | 0.3 / 0 | 120 / 120 | 0 | 0 / 0 | 1 |
| Ratings | 64 / 256 | 0.001 | 0.01 | 0.4 / 0 | 150 / 150 | 0 | 0 / 0 | 1 |
| GitHub | 64 / 256 | 0.001 | 0.01 | 0.3 / 0.3 | 40 / 40 | 0 | 0 / 0 | 1 |
| Arxiv-year | 128 / 512 | 0.001 | 0.01 | 0.3 / 0 | 150 / 150 | 0 | 0 / 20000 | 2 |
| Pokec | 128 / 512 | 0.001 | 0.01 | 0.3 / 0 | 200 / 200 | 0 | 0 / 20000 | 2 |
| Products | 128 / 512 | 0.001 | 0.01 | 0.3 / 0 | 100 / 100 | 0 | 0 / 20000 | 2 |

Gradient clipping uses norm 1.0. The relation-scale initialization is 0.05 for topology/connection and 0.05 for background except Minesweeper and Roman, whose topology/connection initialization is 0.1. Complete settings, including all exceptions, remain in the JSON presets rather than being duplicated in prose.

## Graph inputs and splits

The six input files and dataset sources are listed in [`data/README.md`](../data/README.md). All compared methods receive the same processed graph, features, labels and fixed train/validation/test masks. The official converter preserves Platonov's 50/25/25 masks for Minesweeper, Roman, Ratings and Tolokers. Their trainer and classification launcher default to official split 0. Select another mask with `--split-index 0` through `9` (or `--split-suffix _split0` through `_split9`); the shell main launcher also accepts `SPLIT_INDEX`. Each selected split is fixed across five training seeds, and its reported mean and sample standard deviation are computed across these five runs. Different split IDs are kept separate. The unsuffixed files alias official split 0 for interoperability. Products uses OGB's official sales-ranking split. WikiCS uses official split 0 and excludes its stopping nodes from the three supervised splits. Arxiv-year and reconstructed Pokec use LINKX split 0; Pokec uses the documented 19-dimensional profile encoding. LastFM/DBLP conversion uses a stratified split with seed 0. A split-generation seed is distinct from a training seed.

Keep the exact processed split arrays when rerunning. The trainer records SHA256 fingerprints of the six inputs in its run configuration; these identify the actual processed input rather than only its approximate split ratios. Data-download or reconstruction availability must be resolved before a reproduction run; the framework does not regenerate unknown historical masks from the three percentages alone.

## Sampling and relation semantics

Candidate discovery has a budget of at most `W * Hs` adjacency moves per center. This is a discovery budget, not a bound on all original-graph distance or evidence queries. The candidate pool is capped at `2 * Nctx` before final selection, using original-graph distance buckets, hop quotas, and redistribution of unused capacity. Background proposals are formed before exclusion, are filtered against the sequence and connecting-path intermediates, and are not refilled. Path attribute truncation preserves path counts and transition masses; background exclusions are formed before the attribute budget is applied.

Connection statistics are calibrated without labels using fixed training-center/context draws (`stat_fit_centers=1024`). The encoder uses standardized `log(1+count)` for known connections and standardized `log(mass)` for positive connections. Means and population standard deviations are fitted separately by distance when a group has at least 32 records, with pooled statistics as fallback. A degenerate standard deviation falls back to 1. Availability indicators distinguish missing evidence from an observed zero connection. See [`fit_statistic_scales`](../SCOPE_GT/relation.py).

Topology, connection and background vectors are normalized independently by dividing each component by `max(1, RMS(component))` before layer-specific relation injection. Masks zero unavailable components. The learned self-topology vector is distinct from the non-self relation rows. Node encodings and base keys/values are recomputed in each forward pass; reusable structural arrays do not represent historical learned embeddings.

Full-query updates private context states in the first attention layer. Center-to-context reads retain the same center-specific relation vectors as SCOPE-GT; context queries attend to base keys and values without relation vectors. No context-to-context relation matrix is constructed. The final layer keeps the center read only.

## Diagnostic protocols

The relation-evidence and query-role diagnostic launchers default to paired seeds 0, 1, 2, 3, 4, width 64, four heads and two attention layers. `SEEDS` and `THREADS` override their shell defaults. Within each comparison, sampled sequences and initial parameters are matched. Query roles vary in the first layer; final center reads and token-wise feed-forward updates remain defined by the diagnostic. These runs are separate from the main classification presets. Changing the default does not regenerate previously produced figures; rerun and aggregate the requested seeds before reporting a new run count. Legacy three-seed diagnostics can be reproduced with `SEEDS="0 1 2"`.

The NLGT overlap launcher samples five seeds by default. The displayed illustration uses seed 0's first complete batch of 64 WikiCS training sequences, fanouts 10/5/2 and a maximum of 168 tokens. Padding and virtual tokens are excluded; the display ends at the longest valid sequence. It uses the bundled NLGT NPZ adapter. `SEEDS="0"` reproduces the illustration's sampling scope.

## Execution and batch-size protocols

The batch-size cost sweep uses `B={16,32,64,128,256}` on WikiCS and additionally `512,1024` on CoraFull. It defaults to five paired seeds, one isolated process per layout/seed, three warmup epochs and ten measured training epochs; WikiCS retains its two training views, while CoraFull uses one. Each measured epoch traverses the complete train mask (580 and 11,875 centers in the corresponding inputs). Loading and offline calibration are outside this epoch window. Time includes online sampling, relation-evidence construction, batch preparation and training. Peak GPU memory is allocated tensor memory during the measured training window. The collector first averages epoch time within each seed and takes that seed's maximum GPU allocation, then reports means and sample standard deviations across seeds. Historical curves used seed 0; request `--cost-seed 0` explicitly for that scope. Legacy and multi-seed output layouts are read separately rather than pooled.

The batch-size study is distinct from the shared-encoder mechanism experiment. Sharing counts default to five sampler seeds, using epoch 0/view 0 with the same center order across B and identical per-center sampled inputs within each seed. Historical sharing curves use seed 0. Sharing counts extend to B=512 on WikiCS and B=2048 on CoraFull. Large-graph sharing statistics can use an explicitly requested fixed 20,000-center training subset; these are input-reuse statistics, not complete graph-training or final-prediction measurements.

For numerical checks, the reference layouts are compared on the first two cached mini-batches on CPU with dropout disabled. Logit and parameter-gradient comparisons concern this limited deterministic check; they do not assert equality of stochastic training trajectories. The resource-timing runs retain configured dropout and the training objective. Check-specific outputs are produced by the execution scripts when run, and are distinct from scientific accuracy results.

## External comparison recipes

Validation-based configuration selection for datasets without public baseline
settings is described in the [README](../README.md#configuration-selection).
The [ANS-GT runtime budget](baselines.md#ans-gt-runtime-budget) defines
classification OOT as one training epoch exceeding twelve hours (43,200 seconds).
This per-epoch limit is distinct from optional whole-job timeouts and does not
apply to the separate efficiency or transfer protocols.

Baseline model implementations are bundled in `Baselines/`; `SCOPE_GT_BASELINES_ROOT` optionally selects a replacement NPZ adapter tree. See [baseline provenance and commands](baselines.md). An unmodified upstream repository is not interchangeable with this interface. The source and runtime metadata of an external run are recorded by the launchers. Core SCOPE-GT training does not depend on this directory. Complete operating points are specified by [`SCOPE_GT_efficiency/config.py`](../SCOPE_GT_efficiency/config.py), [`SCOPE_GT_budgets/profiles.json`](../SCOPE_GT_budgets/profiles.json), and the transfer configuration modules. They remain separate from the SCOPE-GT classification presets.

| Method | Arxiv-year operating point | Computers operating point |
|---|---|---|
| GraphGPS | width 64, 2 layers, Performer, degree positional features | width 64, 2 layers, Performer, 4 Laplacian features |
| Exphormer | width 96, 3 layers | width 80, 4 layers |
| SGFormer | width 256, 1 attention + 3 GCN layers | width 64, 1 attention + 4 GCN layers |
| NLGT | width 128, 4 layers, fanouts 10/5/2, B64, 10 prediction votes | same structural settings |
| Gophormer | paper-guided dense adapter, width 64, 4 layers, fanouts 7/7, B128 | width 64, 3 layers, fanouts 7/7, B32 |
| SwapGT | width 512, 1 attention layer, B2048 | width 512, 1 attention layer, B2048 |

The Gophormer adapter retains four training samples and five inference samples. Full-graph methods make one full-graph supervised update per epoch, while tokenized methods make mini-batch updates. Computers convergence validates every epoch and retains method-specific stopping rules under a separate 7,200-second observation window from subprocess launch (`--observation-seconds`; 0 disables); efficiency epoch, preparation and whole-job watchdogs default to disabled, and other datasets and studies have no observation window. SCOPE-GT uses three evaluation contexts; NLGT uses ten prediction votes. The efficiency/transfer NPZ profile uses `strict_no_neighbor_labels=true` for NLGT; this must not be confused with separately reported classification baselines that allow masked-center training-neighbor labels.

Nested Arxiv-year subgraphs are generated without labels. Their node counts 5,000/10,000/20,000/40,000/60,000 correspond to 941/4,217/16,993/64,138/144,438 undirected edges in the recorded construction. Scale curves therefore change both node count and density. The public efficiency launcher defaults to five independent seeds (0 through 4). Each epoch covers every training center; per-epoch records include both supervised-center and optimizer-update counts. Use `--seeds 0` explicitly for a single-seed resource study; repeated epochs within a seed remain within-run observations.

## Transfer boundaries

The LastFM transfer launcher uses five paired seeds. Native and +SCMB retain each backbone's sampler, optimizer, schedule, losses, validation selection, input features and inference views/votes. Gophormer virtual-token memory stays sequence-specific. HubGT uses four layers, width 128, eight heads, 48 tokens and B1024, retaining its hierarchical index, virtual-root features and distance biases.

SwapGT uses one layer, width 512, eight heads and B2048. Its raw-attribute and propagated-topology branches remain distinct feature sources; each branch has one original and two swapped five-token sequences. All six readouts attend to the full concatenated 30-token input, retaining native fusion and alignment losses. Sharing keys include both node identity and feature source, so raw and propagated features of the same node are not merged.

## Entry points

```bash
bash scripts/run_main.sh wikics
bash scripts/run_ablation.sh wikics
bash scripts/run_budget.sh
bash scripts/run_depth.sh
bash scripts/run_batch_size.sh cost
bash scripts/run_batch_size.sh sharing
bash scripts/run_batch_size.sh collect
bash scripts/run_motivation.sh relation
bash scripts/run_motivation.sh query
```

Bundled-adapter experiments:

```bash
bash scripts/run_efficiency.sh
bash scripts/run_transfer.sh
bash scripts/run_motivation.sh overlap
```

For configuration inspection without training, use the Python `--help` and supported `--dry-run` options. Data roots and output locations are supplied at runtime; no machine-specific paths are required in the repository.
