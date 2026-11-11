from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate repeated SGFormer runs")
    parser.add_argument("--root", required=True, help="Repeat root containing seed_xxx/summary.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_step", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for i in range(args.runs):
        seed = args.seed + i * args.seed_step
        summary_path = root / f"seed_{seed}" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append(s)

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=float)

    test_percent = arr("best_test_at_best_val") * 100.0
    val_percent = arr("best_val_score") * 100.0
    train_s = arr("avg_train_epoch_s")
    complete_s = arr("avg_complete_epoch_s")
    mem_mb = arr("avg_peak_mem_mb")
    max_mem_mb = arr("max_peak_mem_mb")

    ddof = 1 if len(rows) > 1 else 0
    agg = {
        "runs": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "metric": rows[0].get("metric", "unknown"),
        "test_score_mean_percent": float(np.mean(test_percent)),
        "test_score_var_percent": float(np.var(test_percent, ddof=ddof)),
        "test_score_std_percent": float(np.std(test_percent, ddof=ddof)),
        "val_score_mean_percent": float(np.mean(val_percent)),
        "val_score_var_percent": float(np.var(val_percent, ddof=ddof)),
        "val_score_std_percent": float(np.std(val_percent, ddof=ddof)),
        "avg_train_epoch_s_mean": float(np.mean(train_s)),
        "avg_train_epoch_s_var": float(np.var(train_s, ddof=ddof)),
        "avg_complete_epoch_s_mean": float(np.mean(complete_s)),
        "avg_complete_epoch_s_var": float(np.var(complete_s, ddof=ddof)),
        "avg_peak_mem_mb_mean": float(np.mean(mem_mb)),
        "avg_peak_mem_mb_var": float(np.var(mem_mb, ddof=ddof)),
        "max_peak_mem_mb_mean": float(np.mean(max_mem_mb)),
        "max_peak_mem_mb_max": float(np.max(max_mem_mb)),
    }

    # rocauc on questions / minesweeper / tolokers (train_sgformer.choose_metric): the runs were
    # selected by validation ROC-AUC and the test scores above are ROC-AUC
    label = "ROC_AUC" if str(agg["metric"]).lower() in {"rocauc", "auc", "roc_auc"} else "Acc"
    print("\n==== Summary ====")
    print(f"Test {label}: {agg['test_score_mean_percent']:.2f} \u00b1 {agg['test_score_std_percent']:.2f}")
    print(f"Tepoch (s): {agg['avg_complete_epoch_s_mean']:.2f}")
    print(f"Mem. (MB): {agg['max_peak_mem_mb_max']:.2f}")
    # Tepoch above includes the per-epoch evaluation; the efficiency block uses training-only time.
    # Same labels/definitions as common/protocol.py print_efficiency.
    ttb = np.asarray([float(r.get("time_to_best_s", float("nan"))) for r in rows], dtype=float)
    best_epoch = np.asarray([float(r.get("best_epoch", float("nan"))) for r in rows], dtype=float)
    params = int(rows[0].get("params", 0))
    print("==== Efficiency ====")
    print(f"Params: {params:,} ({params / 1e6:.3f}M)")
    print(f"Train time/epoch (s): {float(np.mean(train_s)):.4f}")
    print(f"Time to best (s): {float(np.nanmean(ttb)):.2f} \u00b1 {float(np.nanstd(ttb, ddof=ddof)):.2f} "
          f"(best epoch {float(np.nanmean(best_epoch)):.1f})")
    print(f"Peak GPU mem (MB): {agg['max_peak_mem_mb_max']:.2f}")


if __name__ == "__main__":
    main()
