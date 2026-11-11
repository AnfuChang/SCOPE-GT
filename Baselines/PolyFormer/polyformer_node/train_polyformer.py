from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from polyformer_node.data import bases_to_tensor, build_polynomial_bases, load_bilgt_graph
from polyformer_node.metrics import compute_metric
from polyformer_node.model import PolyFormer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def tensor_labels(labels: np.ndarray, device: torch.device, problem_type: str) -> torch.Tensor:
    if problem_type in {"binary", "multilabel"}:
        return torch.from_numpy(labels.astype(np.float32)).to(device)
    return torch.from_numpy(labels.astype(np.int64)).to(device)


def criterion(logits: torch.Tensor, labels: torch.Tensor, problem_type: str) -> torch.Tensor:
    if problem_type == "binary":
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float().reshape(-1))
    if problem_type == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels.long())


def make_optimizer(model: PolyFormer, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameters: List[Dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "attnmodule" in name:
            parameters.append({"params": param, "lr": args.attn_lr, "weight_decay": args.attn_wd})
        elif "ffnmodule" in name:
            parameters.append({"params": param, "lr": args.lr, "weight_decay": args.weight_decay})
        else:
            parameters.append({"params": param, "lr": args.lr, "weight_decay": args.weight_decay})
    return torch.optim.Adam(parameters)


def choose_metric(dataset: str, explicit: str, problem_type: str) -> str:
    if explicit and explicit.lower() != "auto":
        m = explicit.lower()
        if m == "accuracy":
            return "acc"
        if m in {"auc", "roc_auc"}:
            return "rocauc"
        return m
    name = dataset.lower().replace("-", "_")
    if problem_type in {"binary", "multilabel"} or name in {"minesweeper", "tolokers", "questions", "ogbn_proteins"}:
        return "rocauc"
    return "acc"


def iter_batches(idx: np.ndarray, batch_size: int, shuffle: bool) -> Iterable[np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    if shuffle:
        idx = np.random.permutation(idx)
    if batch_size <= 0 or batch_size >= len(idx):
        yield idx
    else:
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]


def forward_nodes(
    model: PolyFormer,
    bases: Sequence[np.ndarray],
    nodes: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x = bases_to_tensor(bases, device, nodes)
    return model(x)


def train_one_epoch_full(
    model: PolyFormer,
    optimizer: torch.optim.Optimizer,
    bases: Sequence[np.ndarray],
    labels_t: torch.Tensor,
    train_idx: np.ndarray,
    device: torch.device,
    problem_type: str,
    grad_clip: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    all_nodes = np.arange(len(bases[0]), dtype=np.int64)
    logits = forward_nodes(model, bases, all_nodes, device)
    idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    loss = criterion(logits[idx_t], labels_t[idx_t], problem_type)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def train_one_epoch_batch(
    model: PolyFormer,
    optimizer: torch.optim.Optimizer,
    bases: Sequence[np.ndarray],
    labels_t: torch.Tensor,
    train_idx: np.ndarray,
    device: torch.device,
    problem_type: str,
    batch_size: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for nodes in iter_batches(train_idx, batch_size, shuffle=True):
        if len(nodes) == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = forward_nodes(model, bases, nodes, device)
        yb = labels_t[torch.as_tensor(nodes, dtype=torch.long, device=device)]
        loss = criterion(logits, yb, problem_type)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(nodes)
        total += len(nodes)
        del logits, yb, loss
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_nodes(
    model: PolyFormer,
    bases: Sequence[np.ndarray],
    labels_t: torch.Tensor,
    idx: np.ndarray,
    device: torch.device,
    problem_type: str,
    metric: str,
    eval_batch_size: int,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    weights: List[int] = []
    logits_parts: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []
    for nodes in iter_batches(idx, eval_batch_size, shuffle=False):
        if len(nodes) == 0:
            continue
        logits = forward_nodes(model, bases, nodes, device)
        yb = labels_t[torch.as_tensor(nodes, dtype=torch.long, device=device)]
        loss = criterion(logits, yb, problem_type)
        losses.append(float(loss.item()))
        weights.append(len(nodes))
        logits_parts.append(logits.detach().cpu())
        labels_parts.append(yb.detach().cpu())
    if not weights:
        return float("nan"), float("nan")
    logits_all = torch.cat(logits_parts, dim=0)
    labels_all = torch.cat(labels_parts, dim=0)
    score = compute_metric(logits_all, labels_all, metric, problem_type)
    loss_mean = float(np.average(np.asarray(losses), weights=np.asarray(weights)))
    return loss_mean, float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PolyFormer node-classification baseline for BiLGT-format data")
    parser.add_argument("--data_dir", required=True, help="Directory containing adj.npz, feat.npy, label.npy, train.npy, val.npy, test.npy")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100, help="Patience counted by evaluation rounds")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--attn_lr", type=float, default=0.0005)
    parser.add_argument("--attn_wd", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dprate", type=float, default=0.5)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--d_ffn", type=int, default=128)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--multi", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--nlayer", type=int, default=1)
    parser.add_argument("--base", choices=["mono", "cheb"], default="mono")
    parser.add_argument("--metric", default="auto", choices=["auto", "acc", "accuracy", "rocauc", "auc", "roc_auc"])
    parser.add_argument("--force_binary", type=int, default=0, help="Use one-logit BCE for scalar 0/1 labels")
    parser.add_argument("--to_undirected", type=int, default=1)
    parser.add_argument("--feature_norm", default="none", choices=["none", "row", "standard"])
    parser.add_argument("--cache_dir", default="", help="Optional polynomial-basis cache directory")
    parser.add_argument("--cache_mode", default="ram", choices=["ram", "memmap"])
    parser.add_argument("--force_recompute_bases", type=int, default=0)
    parser.add_argument("--train_mode", default="batch", choices=["full", "batch"], help="full matches official full-forward style; batch is exact node-wise batching after polynomial bases are built")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--save_best", type=int, default=1)
    parser.add_argument("--config_source", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.provenance import record_config
    record_config(args, __file__)
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load once with preliminary metric; final metric may be inferred after problem type is known.
    graph = load_bilgt_graph(
        args.data_dir,
        dataset_name=args.dataset,
        metric=args.metric,
        force_binary=bool(args.force_binary),
        to_undirected=bool(args.to_undirected),
        feature_norm=args.feature_norm,
    )
    metric = choose_metric(args.dataset, args.metric, graph.problem_type)
    # If metric auto resolved to rocauc and labels are scalar 0/1, reload to use binary loss.
    if args.metric.lower() == "auto" and metric == "rocauc" and graph.problem_type == "multiclass" and graph.num_classes <= 2:
        graph = load_bilgt_graph(
            args.data_dir,
            dataset_name=args.dataset,
            metric="rocauc",
            force_binary=True,
            to_undirected=bool(args.to_undirected),
            feature_norm=args.feature_norm,
        )

    config_for_json = vars(args).copy()
    config_for_json.update({"resolved_metric": metric, "problem_type": graph.problem_type})
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_for_json, f, indent=2, ensure_ascii=False)

    print(
        f"[Data] dataset={args.dataset} N={graph.num_nodes} E={graph.adj.nnz} F={graph.num_features} "
        f"classes={graph.num_classes} out_dim={graph.out_dim} problem={graph.problem_type}"
    )
    print(f"[Split] train={len(graph.train_idx)} val={len(graph.val_idx)} test={len(graph.test_idx)}")
    print(
        f"[Config] source={args.config_source} base={args.base} K={args.K} hidden={args.hidden} n_head={args.n_head} "
        f"nlayer={args.nlayer} metric={metric} train_mode={args.train_mode} seed={args.seed}"
    )

    pre_start = time.perf_counter()
    bases = build_polynomial_bases(
        graph,
        dataset=args.dataset,
        base=args.base,
        K=args.K,
        cache_dir=args.cache_dir or None,
        cache_mode=args.cache_mode,
        force_recompute=bool(args.force_recompute_bases),
    )
    precompute_s = time.perf_counter() - pre_start
    print(f"[Bases] built/loaded {len(bases)} polynomial token matrices in {precompute_s:.4f}s; cache_dir={args.cache_dir or '<none>'}")

    model = PolyFormer(
        graph.num_features,
        args.hidden,
        graph.out_dim,
        K=args.K,
        nlayer=args.nlayer,
        n_head=args.n_head,
        d_ffn=args.d_ffn,
        q=args.q,
        multi=args.multi,
        dropout=args.dropout,
        dprate=args.dprate,
    ).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = make_optimizer(model, args)
    labels_t = tensor_labels(graph.labels, device, graph.problem_type)

    best_val = -float("inf")
    best_test = -float("inf")
    best_epoch = 0
    best_row: Optional[Dict[str, Any]] = None
    bad_eval_rounds = 0
    history: List[Dict[str, Any]] = []

    log_path = out_dir / "log.tsv"
    fields = [
        "epoch", "evaluated", "train_loss", "val_loss", "test_loss", "val_score", "test_score",
        "train_epoch_s", "val_eval_s", "complete_epoch_s", "peak_mem_mb", "best_val", "best_test_at_best_val"
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            reset_peak_memory(device)
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            if args.train_mode == "full":
                train_loss = train_one_epoch_full(
                    model, optimizer, bases, labels_t, graph.train_idx, device, graph.problem_type, args.grad_clip
                )
            else:
                train_loss = train_one_epoch_batch(
                    model, optimizer, bases, labels_t, graph.train_idx, device, graph.problem_type, args.batch_size, args.grad_clip
                )
            cuda_sync()
            train_epoch_s = time.perf_counter() - train_start

            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            val_loss = test_loss = val_score = test_score = float("nan")
            val_eval_s = 0.0
            if do_eval:
                val_start = time.perf_counter()
                val_loss, val_score = evaluate_nodes(
                    model, bases, labels_t, graph.val_idx, device, graph.problem_type, metric, args.eval_batch_size
                )
                val_eval_s = time.perf_counter() - val_start   # logits are copied to the CPU (synchronised)
                test_loss, test_score = evaluate_nodes(
                    model, bases, labels_t, graph.test_idx, device, graph.problem_type, metric, args.eval_batch_size
                )
                improved = (not math.isnan(val_score)) and val_score > best_val
                if improved:
                    best_val = val_score
                    best_test = test_score
                    best_epoch = epoch
                    bad_eval_rounds = 0
                    if args.save_best:
                        torch.save(model.state_dict(), out_dir / "best_model.pt")
                else:
                    bad_eval_rounds += 1

            cuda_sync()
            complete_epoch_s = time.perf_counter() - epoch_start
            mem = peak_memory_mb(device)
            row: Dict[str, Any] = {
                "epoch": epoch,
                "evaluated": int(do_eval),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "val_score": val_score,
                "test_score": test_score,
                "train_epoch_s": train_epoch_s,
                "val_eval_s": val_eval_s,
                "complete_epoch_s": complete_epoch_s,
                "peak_mem_mb": mem,
                "best_val": best_val,
                "best_test_at_best_val": best_test,
            }
            history.append(row)
            if do_eval and epoch == best_epoch:
                best_row = dict(row)
            writer.writerow(row)
            f.flush()

            if do_eval:
                print(
                    f"[Epoch {epoch:04d}] train_loss={train_loss:.6f} val_{metric}={val_score:.6f} "
                    f"test_{metric}={test_score:.6f} best_val={best_val:.6f} best_test={best_test:.6f} "
                    f"train_s={train_epoch_s:.4f} complete_s={complete_epoch_s:.4f} mem_mb={mem:.2f}"
                )
            else:
                print(f"[Epoch {epoch:04d}] train_loss={train_loss:.6f} train_s={train_epoch_s:.4f} complete_s={complete_epoch_s:.4f} mem_mb={mem:.2f}")

            if args.patience > 0 and bad_eval_rounds >= args.patience:
                print(f"[EarlyStop] no improvement for {bad_eval_rounds} eval rounds; stop at epoch={epoch}")
                break

    if best_row is None and history:
        # Should only happen when no evaluation produced a valid score.
        best_row = min(history, key=lambda r: r["epoch"])

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "metric": metric,
        "problem_type": graph.problem_type,
        "config_source": args.config_source,
        "base": args.base,
        "K": args.K,
        "hidden": args.hidden,
        "n_head": args.n_head,
        "nlayer": args.nlayer,
        "precompute_bases_s": precompute_s,
        "best_epoch": int(best_epoch),
        "stop_epoch": int(history[-1]["epoch"]) if history else 0,
        "best_val_score": float(best_val),
        "best_test_at_best_val": float(best_test),
        "best_val_score_percent": float(best_val * 100.0),
        "best_test_at_best_val_percent": float(best_test * 100.0),
        "best_epoch_train_s": float(best_row.get("train_epoch_s", float("nan"))) if best_row else float("nan"),
        "best_epoch_complete_s": float(best_row.get("complete_epoch_s", float("nan"))) if best_row else float("nan"),
        "best_epoch_peak_mem_mb": float(best_row.get("peak_mem_mb", float("nan"))) if best_row else float("nan"),
        "avg_train_epoch_s": float(np.mean([r["train_epoch_s"] for r in history])) if history else float("nan"),
        "avg_complete_epoch_s": float(np.mean([r["complete_epoch_s"] for r in history])) if history else float("nan"),
        "avg_peak_mem_mb": float(np.mean([r["peak_mem_mb"] for r in history])) if history else float("nan"),
        "max_peak_mem_mb": float(np.max([r["peak_mem_mb"] for r in history])) if history else float("nan"),
        "epochs_run": len(history),
        "params": num_params,
        # training + validation time of epochs 1..best_epoch (test evaluation / checkpoint saving excluded)
        "time_to_best_s": float(sum(r["train_epoch_s"] + r["val_eval_s"] for r in history if r["epoch"] <= best_epoch)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[Summary] " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
