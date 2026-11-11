# Sharing and exact-retrieval benchmarks

These launchers measure the execution effects of encoder/K/V sharing and three-hop query reuse. Their measured windows are resource benchmarks rather than training-to-convergence accuracy runs.

Install the paper's CUDA PyTorch build first (see the repository requirements),
then `pip install -r requirements-experiments.txt`. Run commands from the repo root.

## A: encoder sharing versus base K/V sharing

```bash
DATA_BASE=/path/to/datasets bash scripts/run_mechanism_a.sh
```

Default: WikiCS and CoraFull, batch 256, five paired seeds, three warmup and ten
measured complete training epochs. The dataset presets supply the remaining
model, context budgets, dropout and objective, including native consistency/views.
No loss or views are silently changed to CE-only. To deliberately select CE-only,
pass `-- --consistency 0`; all three variants must be remeasured under that choice.
Use `--layouts shared_encoder` to run only the new middle variant when compatible
endpoint records already exist. The collector does not silently merge external
endpoint records: compare their manifests before combining results.

| Name | Encoder pool | K/V projection input |
| --- | --- | --- |
| packed_unshared | per-center node pool | every valid read, including self |
| shared_encoder | unique token/support node pool | every valid read, including self |
| dedup_packed | unique token/support node pool | unique token nodes |

All variants use the same packed attention/read ordering and original-graph
relations. Before timing, each worker checks dropout-free CPU logits and all
parameter gradients against dedup_packed on two cached batches. Normal dropout
is retained in timed training; shared encoders reuse a node's stochastic encoding,
whereas per-center encoders can draw separate masks. This is not a claim of
identical stochastic optimization trajectories. Comparisons B/C have the same
encoder randomness; A/B also change its sharing semantics.

`encoded_rows`, `valid_reads` include all training views and self reads where
applicable. `kv_projection_rows` sums input rows across layers for **one** of K/V;
K and V each process this number of rows. Encoder counts include supporting nodes.
Existing host stage fields are dispatch timings, not synchronized GPU-stage times.
Use a separate optional profiler pass to inspect GPU work:

```bash
python -m SCOPE_GT_mechanisms.run A --dataset wikics \
  --data-root /data/wikics --out results/a_profile --profile
```

`profile.json` is a Chrome trace with encoder/relation/K/V ranges. Profiling never
enters the measured epoch window. These new three-way measurements should not be
spliced with Table V endpoints from a different version/recipe: A intentionally
uses per-read K/V projections, and row counts expose that scope.

## B: identical evidence, different three-hop query reuse

```bash
DATA_BASE=/path/to/datasets bash scripts/run_mechanism_b.sh
```

Default: CoraFull and Products, full graph indices, 2,048 uniformly sampled training
centers, batch 256, five paired seeds. This is a fixed-request retrieval
microbenchmark, not a full-graph training or prediction measurement. The workload
is saved once per seed; both workers use exactly the same tokens and random
background streams. Both reconstruct common-neighbor evidence without discovery
caches. Candidate discovery/calibration and feature encoding are outside timing.

The independent reference calls the same vectorized, chunked exact DAG kernel
with a private root identity per pair. It retains within-pair coalescing and
cheaper-branch selection; only cross-pair reuse is removed. It is not a Python
path-enumeration baseline. The timed window covers the complete `relations`
operation (topology, connection and background); differences isolate the changed
three-hop retrieval organization. Two-hop costs common to both remain included.

Per-batch NPZ files retain evidence arrays. The parent verifies all array fields:
integers/sets exactly, floats with rtol=1e-6 and atol=1e-8. Failed checks invalidate
both variants for summary. `equivalence.json` records errors per field. The RSS
metric is this isolated worker's resident memory sampled every 20 ms during
retrieval, including already resident graph/features. Sampling can miss very brief
spikes; feature loading is identical but not timed. Access counter units are kept
separate; do not add edge queries to adjacency reads as a purported byte count.

## Configuration and outputs

Single dataset example (also works on Windows with appropriate paths):

```bash
python -m SCOPE_GT_mechanisms.run B --dataset corafull \
  --data-root /data/corafull --out results/b_corafull \
  --centers 2048 --seeds 0 1 2 3 4 --timeout 3600
```

The output directory must be empty; inputs and raw evidence are never overwritten.
Each seed has a cached workload/initial weights and a SHA-256 manifest of source,
input files, configuration and workload. Each variant runs in a separate process
with its own log, spec and result. Jobs default to no timeout; set an explicit
per-job limit if needed. Failures/timeouts remain recorded and cause nonzero exit.

`per_seed.csv` aggregates measured epochs within each seed. `summary.csv` reports
independent-seed means and sample SD only when all requested seeds succeeded.
Per-run GPU peaks are the maximum over measured epochs; counters/time are epoch
means. `paired_differences.csv` retains seed-matched time differences and ratios.
Missing GPU values on CPU are blank, never zero. Raw rows and checks are in each
worker's `result.json`. To regenerate summaries after interruption:

```bash
python -m SCOPE_GT_mechanisms.run collect --out results/b_corafull
```

Shared scientific overrides follow `--` and are saved in the manifest. Identity,
center-query, layout and full training traversal cannot be overridden. B requires
three-hop, original-graph, full-attribute evidence. A uses all train nodes; B uses
only the declared microbenchmark centers. No automatic baseline adapters are needed.

The direct classification trainer now defaults to five seeds and six threads, matching its shell launcher. The benchmark runner explicitly selects each worker seed and thread count, so this CLI-default alignment does not change the benchmark computation.
