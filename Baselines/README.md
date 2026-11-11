# Baseline source snapshot

This directory bundles the NPZ comparison implementations used by the project,
their compatible model dependencies, launchers and retained attribution files.
Datasets, results, logs, caches, checkpoints, compiled extensions and unused test
or profiling files are excluded. Upstream directories may therefore contain a
source subset rather than a complete upstream checkout.

Read [the implementation and command guide](../docs/baselines.md) first.
[`registry.json`](registry.json) identifies each entrypoint and its provenance;
[`source_inventory.json`](source_inventory.json) identifies the packaged files.
An upstream commit, when recorded, identifies origin. The adapter version is
defined by its own file hashes.

The default classification entry is:

```bash
python scripts/run_baseline.py gophormer wikics --data-base /path/to/datasets --runs 5
```

Run it from the repository root. Resource and transfer studies use their own
explicit harness configurations. Dense Gophormer is a paper-guided project
reimplementation, distinct from the upstream DGL implementation.

The repository's top-level license does not supersede the licenses or attribution
records retained in individual baseline folders.
