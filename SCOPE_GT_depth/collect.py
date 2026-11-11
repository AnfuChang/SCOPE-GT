"""Collect completed query-role/depth runs without imputing missing measurements.

The common L=1 model is exposed as three explicitly shared curve anchors. These
rows retain their original source path and are never counted as three independent
training runs. Scores and paired differences are expressed in percentage points.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROLES = ("center", "partial", "full")
ROLE_LABELS = {"center": "Center (ours)", "partial": "Partial-query", "full": "Full-query"}
BASE_NUMERIC_FIELDS = (
    "test_accuracy", "test_auc", "test_loss", "val_accuracy", "val_auc", "val_loss",
    "epoch_seconds", "epoch_seconds_all", "train_seconds", "time_to_best_seconds",
    "train_peak_allocated_bytes", "inference_peak_allocated_bytes", "inference_seconds",
    "parameters", "reader_forward_backward_seconds", "reader_forward_seconds",
    "center_effective_rank", "best_epoch", "epochs_run",
)


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read result JSON: {path}: {exc}") from exc


def _mean_std(values):
    values = [float(v) for v in values if finite_number(v)]
    return (statistics.mean(values) if values else None,
            statistics.stdev(values) if len(values) > 1 else None, len(values))


def _metric_name(value, dataset):
    if value in ("auc", "roc_auc", "roc-auc"):
        return "auc"
    if value in ("acc", "accuracy"):
        return "accuracy"
    if value in (None, "auto"):
        return "auc" if dataset in ("minesweeper", "tolokers", "questions") else "accuracy"
    raise ValueError(f"Unknown primary metric {value!r} for {dataset}")


def collect_results(results, datasets=None, layers=None):
    """Return ``runs``, ``summary`` and ``paired_deltas`` lists of dictionaries.

    Only completed ``seed*/metrics.json`` files with a finite primary score are
    included. A missing or null diagnostic is retained as a missing value, never
    changed to zero. Standard deviations use ddof=1 and are missing for n=1.
    """
    root = Path(results).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    wanted_datasets = set(datasets) if datasets else None
    wanted_layers = set(layers) if layers else None
    runs = []
    numeric_fields = set(BASE_NUMERIC_FIELDS)
    seen = set()
    config_cache = {}
    for path in sorted(root.glob("*/L*/*/seed*/metrics.json")):
        dataset_dir, layer_dir, role_dir, seed_dir = path.relative_to(root).parts[:4]
        if wanted_datasets and dataset_dir not in wanted_datasets:
            continue
        try:
            depth, seed = int(layer_dir[1:]), int(seed_dir[4:])
        except ValueError:
            continue
        if depth < 1 or seed < 0 or role_dir not in ROLES:
            continue
        if wanted_layers and depth not in wanted_layers:
            continue
        data = _load_json(path)
        if data.get("status", "completed") not in ("completed", "complete", "success", "ok"):
            print(f"[collect] skip non-completed run: {path}", file=sys.stderr)
            continue
        for key, expected in (("dataset", dataset_dir), ("role", role_dir),
                              ("layers", depth), ("seed", seed)):
            if key in data and data[key] != expected:
                raise ValueError(f"Result identity differs from its path: {path}: {key}={data[key]!r}")
        config_path = path.parent.parent / "config.json"
        if config_path not in config_cache:
            config_cache[config_path] = _load_json(config_path) if config_path.is_file() else {}
        config = config_cache[config_path]
        args = config.get("args", {})
        metric = _metric_name(data.get("metric", args.get("metric")), dataset_dir)
        score = data.get(f"test_{metric}")
        if not finite_number(score):
            print(f"[collect] skip run without finite test_{metric}: {path}", file=sys.stderr)
            continue
        if not 0 <= score <= 1:
            raise ValueError(f"Expected test_{metric} in [0,1], received {score!r}: {path}")
        identity = (dataset_dir, depth, role_dir, seed)
        if identity in seen:
            raise ValueError(f"Duplicate run identity: {identity}")
        seen.add(identity)
        numeric_fields.update(k for k, v in data.items() if finite_number(v)
                              and k not in ("seed", "layers"))
        run = {"dataset": dataset_dir, "layers": depth, "role": role_dir, "seed": seed,
               "metric": metric, "score_pct": 100. * score, "shared_anchor": False,
               "source_role": role_dir, "source_metrics_path": str(path)}
        for key in ('initial_state_sha256', 'evaluation_evidence_sha256'):
            if key in data:
                run[key] = data[key]
        if config.get('contract'):
            common = {k: v for k, v in config['contract'].items() if k not in ('role', 'variant')}
            run['common_contract'] = json.dumps(common, sort_keys=True, ensure_ascii=False)
        if config.get('dataset_sha256'):
            run['dataset_sha256'] = json.dumps(config['dataset_sha256'], sort_keys=True)
        if config.get('source_sha256'):
            run['source_sha256'] = json.dumps(config['source_sha256'], sort_keys=True)
        for key, value in data.items():
            if key not in run and (finite_number(value) or value is None):
                run[key] = value
        runs.append(run)

    # L=1 has no useful context update after the final read. The sole trained
    # center model therefore anchors each curve; provenance makes the reuse clear.
    for run in list(runs):
        if run["layers"] == 1 and run["role"] == "center":
            run["shared_anchor"] = True
            for role in ("partial", "full"):
                identity = (run["dataset"], 1, role, run["seed"])
                if identity not in seen:
                    alias = dict(run, role=role, source_role="center", shared_anchor=True)
                    runs.append(alias)
                    seen.add(identity)

    role_order = {role: i for i, role in enumerate(ROLES)}
    runs.sort(key=lambda r: (r["dataset"], r["layers"], role_order[r["role"]], r["seed"]))
    comparisons = defaultdict(list)
    for run in runs:
        comparisons[(run['dataset'], run['layers'], run['seed'])].append(run)
    for identity, rows in comparisons.items():
        for key in ('initial_state_sha256', 'evaluation_evidence_sha256', 'common_contract',
                    'dataset_sha256', 'source_sha256', 'parameters'):
            values = {row[key] for row in rows if row.get(key) is not None}
            if len(values) > 1:
                raise ValueError(f'Query-role controls differ for {identity}: {key}')
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["layers"], run["role"])].append(run)
    summary = []
    fields = ("score_pct",) + tuple(sorted(numeric_fields))
    for (dataset, depth, role), rows in groups.items():
        metrics = {r["metric"] for r in rows}
        if len(metrics) != 1:
            raise ValueError(f"Mixed primary metrics in {dataset}/L{depth}/{role}")
        row = {"dataset": dataset, "layers": depth, "role": role, "metric": rows[0]["metric"],
               "n": len(rows), "seeds": ";".join(str(r["seed"]) for r in rows),
               "shared_anchor": all(r["shared_anchor"] for r in rows)}
        for key in fields:
            mean, std, count = _mean_std(r.get(key) for r in rows)
            row.update({key + "_mean": mean, key + "_std": std, key + "_n": count})
        summary.append(row)

    pairs = []
    for dataset, depth in sorted({(r["dataset"], r["layers"]) for r in runs}):
        center = {r["seed"]: r for r in groups.get((dataset, depth, "center"), [])}
        for other in ("full", "partial"):
            control = {r["seed"]: r for r in groups.get((dataset, depth, other), [])}
            common = sorted(center.keys() & control.keys())
            if not common:
                continue
            if any(center[s]["metric"] != control[s]["metric"] for s in common):
                raise ValueError(f"Primary metric mismatch in {dataset}/L{depth}/center-{other}")
            deltas = [center[s]["score_pct"] - control[s]["score_pct"] for s in common]
            mean, std, count = _mean_std(deltas)
            pairs.append({"dataset": dataset, "layers": depth, "comparison": "center-minus-" + other,
                          "metric": center[common[0]]["metric"], "n": count,
                          "seeds": ";".join(map(str, common)), "delta_pp_mean": mean,
                          "delta_pp_std": std, "positive_count": sum(d > 0 for d in deltas),
                          "negative_count": sum(d < 0 for d in deltas),
                          "zero_count": sum(d == 0 for d in deltas),
                          "shared_anchor": all(center[s]["source_metrics_path"] == control[s]["source_metrics_path"]
                                               for s in common),
                          "seed_deltas_pp": json.dumps(dict(zip(map(str, common), deltas)), sort_keys=True)})
    return {"runs": runs, "summary": summary, "paired_deltas": pairs}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_collection(collection, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in collection.items():
        write_csv(out_dir / (name + ".csv"), rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args(argv)
    collection = collect_results(args.results, args.datasets, args.layers)
    out = args.out_dir or args.results / "summary"
    write_collection(collection, out)
    actual = len({r["source_metrics_path"] for r in collection["runs"]})
    print(f"[collect] {actual} completed independent runs; {len(collection['summary'])} curve points; {out.resolve()}")


if __name__ == "__main__":
    main()
