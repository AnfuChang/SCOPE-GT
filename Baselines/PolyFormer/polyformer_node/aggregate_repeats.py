from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser("Aggregate repeated PolyFormer runs")
    p.add_argument("--root", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_step", type=int, default=1)
    args = p.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for i in range(args.runs):
        seed = args.seed + i * args.seed_step
        path = root / f"seed_{seed}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=float)

    ddof = 1 if len(rows) > 1 else 0
    test_pct = arr("best_test_at_best_val") * 100.0
    val_pct = arr("best_val_score") * 100.0
    best_epoch = arr("best_epoch")
    stop_epoch = arr("stop_epoch")
    max_mem = arr("max_peak_mem_mb")
    avg_train_s = arr("avg_train_epoch_s")
    avg_complete_s = arr("avg_complete_epoch_s")
    precompute_s = arr("precompute_bases_s")

    agg = {
        "runs": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "dataset": rows[0].get("dataset", "unknown") if rows else "unknown",
        "metric": rows[0].get("metric", "unknown") if rows else "unknown",
        "config_source": rows[0].get("config_source", "unknown") if rows else "unknown",
        "base": rows[0].get("base", "unknown") if rows else "unknown",
        "test_score_mean_percent": float(np.nanmean(test_pct)),
        "test_score_std_percent": float(np.nanstd(test_pct, ddof=ddof)),
        "val_score_mean_percent": float(np.nanmean(val_pct)),
        "val_score_std_percent": float(np.nanstd(val_pct, ddof=ddof)),
        "best_epoch_mean": float(np.nanmean(best_epoch)),
        "best_epoch_std": float(np.nanstd(best_epoch, ddof=ddof)),
        "stop_epoch_mean": float(np.nanmean(stop_epoch)),
        "max_peak_mem_mb_mean": float(np.nanmean(max_mem)),
        "max_peak_mem_mb_std": float(np.nanstd(max_mem, ddof=ddof)),
        "avg_train_epoch_s_mean": float(np.nanmean(avg_train_s)),
        "avg_train_epoch_s_std": float(np.nanstd(avg_train_s, ddof=ddof)),
        "avg_complete_epoch_s_mean": float(np.nanmean(avg_complete_s)),
        "avg_complete_epoch_s_std": float(np.nanstd(avg_complete_s, ddof=ddof)),
        "precompute_bases_s_mean": float(np.nanmean(precompute_s)),
    }

    # rocauc on questions / minesweeper / tolokers (train_polyformer.choose_metric): the runs were
    # selected by validation ROC-AUC and the test scores above are ROC-AUC
    label = "ROC_AUC" if str(agg["metric"]).lower() in {"rocauc", "auc", "roc_auc"} else "Acc"
    print("\n==== Summary ====")
    print(f"Test {label}: {agg['test_score_mean_percent']:.2f} \u00b1 {agg['test_score_std_percent']:.2f}")
    print(f"Tepoch (s): {agg['avg_complete_epoch_s_mean']:.2f}")
    print(f"Mem. (MB): {float(np.nanmax(max_mem)):.2f}")
    # Tepoch above includes the per-epoch evaluation; the efficiency block uses training-only time.
    # Same labels/definitions as common/protocol.py print_efficiency.
    ttb = arr("time_to_best_s")
    print("==== Efficiency ====")
    params = int(rows[0].get("params", 0))
    print(f"Params: {params:,} ({params / 1e6:.3f}M)")
    print(f"Train time/epoch (s): {float(np.nanmean(avg_train_s)):.4f}")
    print(f"Time to best (s): {float(np.nanmean(ttb)):.2f} \u00b1 {float(np.nanstd(ttb, ddof=ddof)):.2f} "
          f"(best epoch {float(np.nanmean(best_epoch)):.1f})")
    print(f"Preprocessing (s): {float(np.nanmean(precompute_s)):.2f}")
    print(f"Time to best incl. preprocessing (s): {float(np.nanmean(ttb + precompute_s)):.2f}")
    print(f"Peak GPU mem (MB): {float(np.nanmax(max_mem)):.2f}")


if __name__ == "__main__":
    main()
