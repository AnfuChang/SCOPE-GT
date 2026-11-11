#!/usr/bin/env python3
"""Measure node overlap using optional external NPZ backbone adapters.

No model is trained. Each seed samples one sequence per training center once;
the exact same ordered sequences are regrouped for every requested batch size.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROTOCOL = "O_node_overlap_v1"
METHODS = ("nlgt", "gophormer")
DEFAULT_CONFIG = {
    "nlgt": {"fanouts": [10, 5, 2], "max_nodes": 168},
    "gophormer": {"fanouts": [7, 7], "global_nodes": 2, "prox_order": 1},
}


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def import_source(path, name):
    path = Path(path).resolve()
    module_name = "_node_overlap_" + name + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def original_nodes(nodes, valid_mask=None, node_types=None):
    """Return unique original IDs, excluding masked, negative and non-real tokens.

    node_types, if supplied, is a boolean original-node mask. Valid original IDs
    must be integers. Native sampler outputs contain no padding/global nodes.
    """
    ids = np.asarray(nodes)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Each sequence must be a one-dimensional integer node-ID array")
    keep = ids >= 0
    for name, value in (("valid_mask", valid_mask), ("node_types", node_types)):
        if value is not None:
            mask = np.asarray(value)
            if mask.shape != ids.shape or mask.dtype != np.bool_:
                raise ValueError(name + " must be a boolean array with the node-ID shape")
            keep &= mask
    return np.unique(ids[keep].astype(np.int64, copy=False))


def batch_statistics(sequences, centers, batch_size):
    """M sums per-sequence unique real IDs; U is the batch union, without padding."""
    if batch_size < 1 or len(sequences) != len(centers):
        raise ValueError("Positive batch size and matching sequence/center counts required")
    centers = np.asarray(centers)
    if centers.ndim != 1 or not np.issubdtype(centers.dtype, np.integer):
        raise ValueError("Centers must be one-dimensional integer IDs")
    if len(np.unique(centers)) != len(centers):
        raise ValueError("This cross-center protocol requires exactly one sequence per center")
    clean = [original_nodes(sequence) for sequence in sequences]
    if not clean or any(len(nodes) == 0 for nodes in clean):
        raise ValueError("Empty sequence collection or sequence with no original nodes")
    if np.any(centers < 0) or any(int(center) not in nodes for center, nodes in zip(centers, clean)):
        raise ValueError("Each nonnegative center ID must occur among its sequence's original nodes")
    for index, start in enumerate(range(0, len(clean), batch_size)):
        part = clean[start:start + batch_size]
        lengths = np.asarray([len(nodes) for nodes in part])
        occurrences = int(lengths.sum())
        unique = int(len(np.unique(np.concatenate(part))))
        yield {
            "batch": index, "batch_size_actual": len(part),
            "partial_batch": len(part) < batch_size,
            "occurrences": occurrences, "unique_nodes": unique,
            "duplicate_occurrences": occurrences - unique,
            "duplicate_fraction": 1.0 - unique / occurrences,
            "occurrence_ratio": occurrences / unique,
            "length_min": int(lengths.min()), "length_mean": float(lengths.mean()),
            "length_max": int(lengths.max()),
        }


def sample_sequences(graph, method, config, seed, baselines_dir, max_centers=0):
    """Call the external NPZ sampler directly; no models or proximity matrices."""
    order = np.random.default_rng(seed).permutation(graph.splits["train"])
    if max_centers:
        order = order[:max_centers]
    if method == "nlgt":
        source = baselines_dir / "NLGT" / "nlgt_npz.py"
        module = import_source(source, method)
        sampler = module.GraphSampler(graph.adj, config["fanouts"])
        sequences = []
        for index, center in enumerate(order):
            # Exact seed formula and truncation from SampledNodeDataset.__getitem__.
            rng = np.random.default_rng(seed + index * 10007 + int(center) * 97)
            sequences.append(np.asarray(sampler.sample_subgraph(int(center), rng)[:config["max_nodes"]], dtype=np.int64))
    else:
        source = baselines_dir / "Gophormer" / "gophormer_npz.py"
        module = import_source(source, method)
        indptr, indices = graph.adj.indptr, graph.adj.indices
        light_graph = SimpleNamespace(neighbors=[indices[indptr[i]:indptr[i + 1]] for i in range(graph.adj.shape[0])])
        sampler = module.EgoGraphSampler(light_graph, config["fanouts"], config["global_nodes"], config["prox_order"])
        rng = np.random.RandomState(seed)
        sequences = [np.asarray(sampler.sample_nodes(int(center), rng), dtype=np.int64) for center in order]
    for center, nodes in zip(order, sequences):
        if len(nodes) == 0 or int(nodes[0]) != int(center) or nodes.min() < 0 or nodes.max() >= graph.adj.shape[0]:
            raise ValueError("Sampler returned an invalid center-first original-node sequence")
    return order.astype(np.int64), sequences, source


def save_sequences(path, dataset, method, seed, centers, sequences, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum([len(nodes) for nodes in sequences])].astype(np.int64)
    np.savez_compressed(path, nodes=np.concatenate(sequences), offsets=offsets, centers=centers,
                        dataset=np.asarray(dataset), method=np.asarray(method), seed=np.asarray(seed),
                        provenance_json=np.asarray(json.dumps(provenance, ensure_ascii=False)))


def load_sequences(path):
    """Read an exported NPZ or explicit original-node JSONL; reject repeated centers."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            nodes, offsets, centers = data["nodes"], data["offsets"], data["centers"]
            if (offsets.ndim != 1 or len(offsets) != len(centers) + 1 or offsets[0] != 0
                    or offsets[-1] != len(nodes) or np.any(np.diff(offsets) <= 0)
                    or not np.issubdtype(offsets.dtype, np.integer)):
                raise ValueError("Invalid ragged sequence offsets")
            sequences = [nodes[offsets[i]:offsets[i + 1]].copy() for i in range(len(centers))]
            metadata = {name: data[name].item() for name in ("dataset", "method", "seed")}
            metadata["provenance"] = json.loads(data["provenance_json"].item())
            centers = centers.copy()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("Empty sequence cache")
        identities = {(row["dataset"], row["method"], row["seed"]) for row in rows}
        if len(identities) != 1:
            raise ValueError("One cache file must contain exactly one dataset/method/seed")
        dataset, method, seed = next(iter(identities))
        metadata = {"dataset": dataset, "method": method, "seed": seed,
                    "provenance": rows[0].get("provenance", {})}
        if not metadata["provenance"]:
            raise ValueError("JSONL first row requires provenance describing the actual sampler and graph")
        centers = np.asarray([row["center"] for row in rows])
        sequences = [original_nodes(row["nodes"], row.get("valid_mask"), row.get("original_node_mask")) for row in rows]
    list(batch_statistics(sequences, centers, max(1, len(centers))))
    if (not isinstance(metadata["dataset"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", metadata["dataset"])
            or metadata["method"] not in METHODS or not isinstance(metadata["seed"], int) or metadata["seed"] < 0):
        raise ValueError("Cache method or seed is invalid")
    if not isinstance(metadata["provenance"].get("fixture_only"), bool):
        raise ValueError("Cache provenance requires an explicit boolean fixture_only")
    return metadata, centers, sequences


def seed_statistics(rows):
    occurrences = sum(row["occurrences"] for row in rows)
    unique_sum = sum(row["unique_nodes"] for row in rows)
    sequences = sum(row["batch_size_actual"] for row in rows)
    return {
        "batches": len(rows), "sequences": sequences,
        "occurrences": occurrences, "batch_unique_nodes_sum": unique_sum,
        "duplicate_occurrences": occurrences - unique_sum,
        "duplicate_fraction": 1.0 - unique_sum / occurrences,
        "occurrence_ratio": occurrences / unique_sum,
        "mean_sequence_length": occurrences / sequences,
        "batch_duplicate_fraction_mean": float(np.mean([r["duplicate_fraction"] for r in rows])),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty results table")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive_unique(values, label, allow_zero=False):
    if len(set(values)) != len(values) or any(x < (0 if allow_zero else 1) for x in values):
        raise ValueError(label + " must be unique " + ("nonnegative" if allow_zero else "positive") + " integers")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Parent of fixed-protocol dataset directories")
    parser.add_argument("--datasets", nargs="+", default=["wikics"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--baselines-dir", type=Path,
                        help="External NPZ adapter tree; defaults to SCOPE_GT_BASELINES_ROOT")
    parser.add_argument("--config", type=Path, help="JSON mapping method names to sampler-parameter overrides")
    parser.add_argument("--cache-input", type=Path, nargs="+", help="Exported NPZ/JSONL files covering the requested grid")
    parser.add_argument("--max-centers", type=int, default=0, help="0 uses the full fixed training split; otherwise explicit center limit")
    parser.add_argument("--fixture-only", action="store_true", help="Mark engineering fixtures; publication plotting rejects these")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    positive_unique(args.seeds, "seeds", True)
    positive_unique(args.batch_sizes, "batch sizes")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.methods)) != len(args.methods):
        parser.error("datasets and methods must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in args.datasets):
        parser.error("dataset names must contain only letters, digits, underscores or hyphens")
    if args.max_centers < 0:
        parser.error("max-centers must be nonnegative")
    if not args.cache_input and args.data_root is None:
        parser.error("--data-root is required unless --cache-input is supplied")
    if args.cache_input and (args.config or args.max_centers):
        parser.error("--config and --max-centers apply to new sampling, not fixed sequence caches")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        if set(overrides) - set(METHODS):
            raise ValueError("Unknown method in sampler configuration")
        for method, settings in overrides.items():
            if set(settings) - set(config[method]):
                raise ValueError("Unknown sampler parameter for " + method)
            config[method].update(settings)
    for method, settings in config.items():
        if not settings["fanouts"] or any(not isinstance(x, int) or x == 0 or x < -1 for x in settings["fanouts"]):
            raise ValueError("fanouts must contain positive integers or -1")
        if method == "nlgt" and (settings["max_nodes"] < 1 or any(x < 1 for x in settings["fanouts"])):
            raise ValueError("NLGT requires positive max_nodes and fanouts")
    manifest = {
        "protocol": PROTOCOL, "fixture_only": args.fixture_only, "scope": "original_graph_node_ids",
        "aggregation": "per seed: 1-sum_batch(U)/sum_batch(M); summary: arithmetic mean and sample std across seeds",
        "datasets": args.datasets, "methods": args.methods, "batch_sizes": args.batch_sizes, "seeds": args.seeds,
        "views_per_center": 1, "include_center": True, "within_sequence_deduplication": True,
        "padding_and_virtual_tokens_excluded": True, "partial_batches_included": True,
        "sampling_once_then_regroup": True, "max_centers": args.max_centers,
        "sampler_config": None if args.cache_input else config,
        "input_kind": "cached_sequences" if args.cache_input else "native_sampler",
        "method_implementations": {"nlgt": "external NPZ adaptation: GraphSampler.sample_subgraph",
                                   "gophormer": "external dense paper-guided adaptation: EgoGraphSampler.sample_nodes"},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "sources": {}, "inputs": {}, "sequence_caches": {},
    }
    manifest["sources"]["O_experiment/run_overlap.py"] = sha256_file(__file__)
    if args.config:
        manifest["inputs"]["config.json"] = sha256_file(args.config)
    batch_rows, seed_rows = [], []

    def consume(dataset, method, seed, centers, sequences, provenance):
        cache = out / "sampled_sequences" / f"{dataset}_{method}_seed{seed}.npz"
        save_sequences(cache, dataset, method, seed, centers, sequences, provenance)
        manifest["sequence_caches"][cache.relative_to(out).as_posix()] = sha256_file(cache)
        for size in args.batch_sizes:
            rows = [{"protocol": PROTOCOL, "dataset": dataset, "method": method, "seed": seed,
                     "batch_size": size, **values} for values in batch_statistics(sequences, centers, size)]
            batch_rows.extend(rows)
            seed_rows.append({"protocol": PROTOCOL, "dataset": dataset, "method": method,
                              "seed": seed, "batch_size": size, **seed_statistics(rows)})
        print(f"sampled {dataset}/{method}/seed{seed}: {len(centers)} centers; regrouped {len(args.batch_sizes)} batch sizes", flush=True)

    if args.cache_input:
        found = {}
        manifest["cache_provenance"] = {}
        manifest["max_centers"] = None  # Each cache retains its actual original sampling scope.
        for path in args.cache_input:
            metadata, centers, sequences = load_sequences(path)
            if metadata["provenance"]["fixture_only"] and not args.fixture_only:
                raise ValueError("A fixture sequence cache requires --fixture-only; it cannot become paper evidence")
            key = (metadata["dataset"], metadata["method"], metadata["seed"])
            if key in found:
                raise ValueError("Duplicate dataset/method/seed cache")
            found[key] = (centers, sequences, metadata["provenance"])
            identity = f"{key[0]}/{key[1]}/seed{key[2]}"
            manifest["inputs"][identity] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
            manifest["cache_provenance"][identity] = metadata["provenance"]
        expected = {(d, m, s) for d in args.datasets for m in args.methods for s in args.seeds}
        if set(found) != expected:
            raise ValueError(f"Cache grid mismatch: missing={expected-set(found)}, extra={set(found)-expected}")
        for key, (centers, sequences, provenance) in found.items():
            consume(*key, centers, sequences, provenance)
    else:
        # Retain direct-script use as well as the repository launcher.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from SCOPE_GT.external_backbones import baseline_root, baseline_path
        baselines = baseline_root(args.baselines_dir)
        baseline_path('common/data.py', baselines)
        sys.path.insert(0, str(baselines))
        loader_path = baselines / "common" / "data.py"
        loader = import_source(loader_path, "common_data")
        for path in sorted((baselines / "common").glob("*.py")):
            manifest["sources"][path.relative_to(baselines).as_posix()] = sha256_file(path)
        for dataset in args.datasets:
            if Path(dataset).name != dataset:
                raise ValueError("Dataset names must be single directory names")
            graph = loader.load_graph(args.data_root / dataset, normalize="none", directed=False)
            manifest["inputs"][dataset] = {"files": graph.fingerprint, "num_nodes": graph.adj.shape[0],
                                            "num_edges_directed_storage": graph.adj.nnz,
                                            "training_centers": len(graph.splits["train"]),
                                            "split": "supplied train.npy/val.npy/test.npy", "directed": False}
            for method in args.methods:
                for seed in args.seeds:
                    centers, sequences, source = sample_sequences(graph, method, config[method], seed, baselines, args.max_centers)
                    manifest["sources"][source.relative_to(baselines).as_posix()] = sha256_file(source)
                    launcher = source.parent / ("run_nlgt.sh" if method == "nlgt" else "run_gophormer.sh")
                    if launcher.is_file():
                        manifest["sources"][launcher.relative_to(baselines).as_posix()] = sha256_file(launcher)
                    provenance = {"graph": manifest["inputs"][dataset], "sampler_source": source.relative_to(baselines).as_posix(),
                                  "sampler_sha256": sha256_file(source), "sampler_config": config[method],
                                  "views_per_center": 1, "fixture_only": args.fixture_only}
                    consume(dataset, method, seed, centers, sequences, provenance)
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["method"], row["batch_size"])].append(row)
    summary = []
    for (dataset, method, size), rows in sorted(grouped.items()):
        values = [row["duplicate_fraction"] for row in rows]
        summary.append({"protocol": PROTOCOL, "dataset": dataset, "method": method, "batch_size": size,
                        "n": len(rows), "seeds": ";".join(str(r["seed"]) for r in sorted(rows, key=lambda r:r["seed"])),
                        "mean_duplicate_fraction": float(np.mean(values)),
                        "std_duplicate_fraction": float(np.std(values, ddof=1)) if len(values) > 1 else "",
                        "mean_occurrences": float(np.mean([r["occurrences"] for r in rows])),
                        "mean_batch_unique_nodes_sum": float(np.mean([r["batch_unique_nodes_sum"] for r in rows])),
                        "mean_sequence_length": float(np.mean([r["mean_sequence_length"] for r in rows])),
                        "mean_sequences": float(np.mean([r["sequences"] for r in rows]))})
    for filename, rows in (("batches.csv", batch_rows), ("seeds.csv", seed_rows), ("summary.csv", summary)):
        write_csv(out / filename, rows)
    manifest["outputs"] = {name: sha256_file(out / name) for name in ("batches.csv", "seeds.csv", "summary.csv")}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(summary)} summary rows to {out / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
