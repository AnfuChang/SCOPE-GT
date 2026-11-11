from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sgformer_node.data import (
    GraphData,
    load_bilgt_graph,
    normalized_subgraph,
    scipy_to_torch_sparse,
)
from sgformer_node.metrics import compute_metric
from sgformer_node.model import SGFormerNode, SGFormerOfficialLarge


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
    if problem_type == "multilabel":
        return torch.from_numpy(labels.astype(np.float32)).to(device)
    return torch.from_numpy(labels.astype(np.int64)).to(device)


def criterion(logits: torch.Tensor, labels: torch.Tensor, problem_type: str) -> torch.Tensor:
    if problem_type == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels.long())


@torch.no_grad()
def evaluate_full(
    model: SGFormerNode,
    x: torch.Tensor,
    adj_norm_t: torch.Tensor,
    labels_t: torch.Tensor,
    idx: np.ndarray,
    metric: str,
    problem_type: str,
) -> Tuple[float, float]:
    model.eval()
    logits = model(x, adj_norm_t)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=logits.device)
    loss = criterion(logits[idx_t], labels_t[idx_t], problem_type).item()
    score = compute_metric(logits[idx_t], labels_t[idx_t], metric, problem_type)
    return float(loss), float(score)


@torch.no_grad()
def predict_in_batches(
    model: SGFormerNode,
    graph: GraphData,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Mini-batch partition inference. This approximates full-graph inference for very large graphs."""
    model.eval()
    n = graph.num_nodes
    out_dim = graph.num_classes
    logits_cpu = torch.empty((n, out_dim), dtype=torch.float32, device="cpu")
    x_np = graph.features
    for start in range(0, n, batch_size):
        nodes = np.arange(start, min(start + batch_size, n), dtype=np.int64)
        sub_adj = normalized_subgraph(graph.adj, nodes)
        sub_adj_t = scipy_to_torch_sparse(sub_adj, device)
        xb = torch.from_numpy(x_np[nodes]).to(device)
        logits = model(xb, sub_adj_t).detach().cpu()
        logits_cpu[nodes] = logits
    return logits_cpu


@torch.no_grad()
def evaluate_batch(
    model: SGFormerNode,
    graph: GraphData,
    labels_t_cpu: torch.Tensor,
    idx: np.ndarray,
    metric: str,
    problem_type: str,
    device: torch.device,
    eval_batch_size: int,
) -> Tuple[float, float]:
    logits = predict_in_batches(model, graph, device, eval_batch_size)
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    loss = criterion(logits[idx_t], labels_t_cpu[idx_t], problem_type).item()
    score = compute_metric(logits[idx_t], labels_t_cpu[idx_t], metric, problem_type)
    return float(loss), float(score)


@torch.no_grad()
def predict_full_cpu(model, graph: GraphData, cpu_cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Whole-graph inference on the CPU, as evaluate_large(..., device="cpu") of the official main-batch.py:
    the model is trained on random partitions but evaluated with every edge of the graph."""
    if not cpu_cache:
        cpu_cache["x"] = torch.from_numpy(graph.features)
        cpu_cache["adj"] = scipy_to_torch_sparse(graph.adj_norm, torch.device("cpu"))
    cpu_model = copy.deepcopy(model).cpu().eval()
    return cpu_model(cpu_cache["x"], cpu_cache["adj"])


def scores_from_logits(logits: torch.Tensor, labels_t_cpu: torch.Tensor, idx: np.ndarray, metric: str,
                       problem_type: str) -> Tuple[float, float]:
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    loss = criterion(logits[idx_t], labels_t_cpu[idx_t], problem_type).item()
    return float(loss), float(compute_metric(logits[idx_t], labels_t_cpu[idx_t], metric, problem_type))


def train_one_epoch_full(
    model: SGFormerNode,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    adj_norm_t: torch.Tensor,
    labels_t: torch.Tensor,
    train_idx: np.ndarray,
    problem_type: str,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=x.device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x, adj_norm_t)
    loss = criterion(logits[idx_t], labels_t[idx_t], problem_type)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item()), float(logits.shape[0])


def train_one_epoch_batch(
    model: SGFormerNode,
    optimizer: torch.optim.Optimizer,
    graph: GraphData,
    labels_t: torch.Tensor,
    train_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    problem_type: str,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    n = graph.num_nodes
    perm = np.random.permutation(n).astype(np.int64)
    total_loss = 0.0
    total_train = 0
    for start in range(0, n, batch_size):
        nodes = perm[start : start + batch_size]
        local_train_mask = train_mask[nodes]
        if not np.any(local_train_mask):
            continue
        sub_adj = normalized_subgraph(graph.adj, nodes)
        sub_adj_t = scipy_to_torch_sparse(sub_adj, device)
        xb = torch.from_numpy(graph.features[nodes]).to(device)
        yb = labels_t[nodes]
        train_local_idx = torch.as_tensor(np.where(local_train_mask)[0], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, sub_adj_t)
        loss = criterion(logits[train_local_idx], yb[train_local_idx], problem_type)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        b_train = int(local_train_mask.sum())
        total_loss += float(loss.item()) * b_train
        total_train += b_train
    if total_train == 0:
        return float("nan"), 0.0
    return total_loss / total_train, float(total_train)


def choose_metric(dataset_name: str, explicit_metric: str) -> str:
    if explicit_metric and explicit_metric.lower() != "auto":
        return explicit_metric.lower()
    name = dataset_name.lower().replace("-", "_")
    if name in {"tolokers", "minesweeper", "questions", "ogbn_proteins", "ogbn-proteins"}:
        return "rocauc"
    return "acc"


def main() -> None:
    parser = argparse.ArgumentParser("SGFormer node-classification baseline for BiLGT-format data")
    parser.add_argument("--data_dir", required=True, help="Directory containing adj.npz, feat.npy, label.npy, train.npy, val.npy, test.npy")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100, help="Patience counted by evaluation rounds, not raw epochs")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--trans_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for GCN branch in Eq. (4)")
    parser.add_argument("--beta", type=float, default=0.5, help="Residual weight in SGFormer global attention Eq. (3)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Transformer-side dropout (official ours_dropout / trans_dropout)")
    parser.add_argument("--gnn_dropout", type=float, default=None, help="GCN-branch dropout (official dropout / gnn_dropout); default: --dropout")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Transformer-side weight decay (official ours_weight_decay / trans_weight_decay)")
    parser.add_argument("--gnn_weight_decay", type=float, default=None, help="GCN-branch weight decay (official weight_decay / gnn_weight_decay); default: --weight_decay")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--metric", default="auto", choices=["auto", "acc", "accuracy", "rocauc", "auc", "roc_auc"])
    parser.add_argument("--to_undirected", type=int, default=1)
    parser.add_argument("--feature_norm", default="none", choices=["none", "row", "standard"])
    parser.add_argument("--train_mode", default="full", choices=["full", "batch"])
    parser.add_argument("--batch_size", type=int, default=20000, help="Mini-batch partition size when --train_mode=batch")
    parser.add_argument("--eval_mode", default="full", choices=["full", "batch", "full_cpu"],
                        help="full_cpu: whole-graph inference on the CPU (official main-batch.py evaluate_large)")
    parser.add_argument("--eval_batch_size", type=int, default=50000)
    parser.add_argument("--arch", default="adapter", choices=["adapter", "official_large"],
                        help="adapter: SGFormerNode; official_large: large/ours.py of the official repository "
                             "(--alpha = graph_weight, --dropout / --weight_decay = trans_*, --gnn_* = gnn_*)")
    parser.add_argument("--trans_heads", type=int, default=1, help="official_large: trans_num_heads")
    for flag, default in [("gnn_use_bn", 1), ("gnn_use_residual", 1), ("gnn_use_weight", 1), ("gnn_use_init", 0),
                          ("gnn_use_act", 1), ("trans_use_bn", 1), ("trans_use_residual", 1), ("trans_use_weight", 1),
                          ("trans_use_act", 0)]:
        parser.add_argument(f"--{flag}", type=int, default=default, help="official_large only (1/0)")
    parser.add_argument("--amp", type=int, default=0, help="Reserved; full sparse training keeps fp32 for stability")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.provenance import record_config
    record_config(args, __file__)

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    metric = choose_metric(args.dataset, args.metric)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"resolved_metric": metric}, f, indent=2, ensure_ascii=False)

    graph = load_bilgt_graph(args.data_dir, to_undirected=bool(args.to_undirected), feature_norm=args.feature_norm)
    print(f"[Data] dataset={args.dataset} N={graph.num_nodes} E={graph.adj.nnz} F={graph.num_features} C={graph.num_classes} problem={graph.problem_type}")
    print(f"[Split] train={len(graph.train_idx)} val={len(graph.val_idx)} test={len(graph.test_idx)}")
    print(f"[Metric] {metric}; train_mode={args.train_mode}; eval_mode={args.eval_mode}; seed={args.seed}")

    gnn_wd = args.weight_decay if args.gnn_weight_decay is None else args.gnn_weight_decay
    if args.arch == "official_large":
        model = SGFormerOfficialLarge(
            graph.num_features, args.hidden_dim, graph.num_classes, graph_weight=args.alpha,
            trans_layers=args.trans_layers, trans_heads=args.trans_heads, trans_dropout=args.dropout,
            trans_use_bn=bool(args.trans_use_bn), trans_use_residual=bool(args.trans_use_residual),
            trans_use_weight=bool(args.trans_use_weight), trans_use_act=bool(args.trans_use_act),
            gcn_layers=args.gcn_layers, gnn_dropout=args.dropout if args.gnn_dropout is None else args.gnn_dropout,
            gnn_use_bn=bool(args.gnn_use_bn), gnn_use_residual=bool(args.gnn_use_residual),
            gnn_use_weight=bool(args.gnn_use_weight), gnn_use_init=bool(args.gnn_use_init),
            gnn_use_act=bool(args.gnn_use_act),
        ).to(device)
        # main.py / main-batch.py: Adam([{params1, trans_weight_decay}, {params2, gnn_weight_decay}])
        trans_params, gnn_params = model.param_groups()
        optimizer = torch.optim.Adam(
            [{"params": trans_params, "weight_decay": args.weight_decay},
             {"params": gnn_params, "weight_decay": gnn_wd}],
            lr=args.lr,
        )
    else:
        model = SGFormerNode(
            graph.num_features,
            args.hidden_dim,
            graph.num_classes,
            alpha=args.alpha,
            beta=args.beta,
            gcn_layers=args.gcn_layers,
            trans_layers=args.trans_layers,
            dropout=args.dropout,
            gnn_dropout=args.gnn_dropout,
        ).to(device)
        # Official SGFormer uses separate weight decays for the GNN branch and the Transformer part
        # (weight_decay / ours_weight_decay, gnn_weight_decay / trans_weight_decay); both default to
        # --weight_decay, which reproduces the previous single-group optimizer.
        gnn_params = list(model.gcn.parameters())
        gnn_ids = {id(p) for p in gnn_params}
        other_params = [p for p in model.parameters() if id(p) not in gnn_ids]
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "weight_decay": args.weight_decay},
             {"params": gnn_params, "weight_decay": gnn_wd}],
            lr=args.lr,
        )
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[Model] arch={args.arch} params={num_params}")

    labels_t = tensor_labels(graph.labels, device, graph.problem_type)
    labels_t_cpu = tensor_labels(graph.labels, torch.device("cpu"), graph.problem_type)
    train_mask = np.zeros(graph.num_nodes, dtype=bool)
    train_mask[graph.train_idx] = True

    if args.train_mode == "full" or args.eval_mode == "full":
        x_full = torch.from_numpy(graph.features).to(device)
        adj_norm_t = scipy_to_torch_sparse(graph.adj_norm, device)
    else:
        x_full = None
        adj_norm_t = None
    cpu_cache: Dict[str, torch.Tensor] = {}   # eval_mode=full_cpu: features and adjacency on the CPU

    best_val = -float("inf")
    best_test = -float("inf")
    best_epoch = 0
    bad_eval_rounds = 0
    history: List[Dict[str, float]] = []

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
                train_loss, train_seen = train_one_epoch_full(
                    model, optimizer, x_full, adj_norm_t, labels_t, graph.train_idx,
                    graph.problem_type, args.grad_clip,
                )
            else:
                train_loss, train_seen = train_one_epoch_batch(
                    model, optimizer, graph, labels_t, train_mask, device, args.batch_size,
                    graph.problem_type, args.grad_clip,
                )
            cuda_sync()
            train_epoch_s = time.perf_counter() - train_start

            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            val_loss = test_loss = val_score = test_score = float("nan")
            val_eval_s = 0.0
            if do_eval:
                val_start = time.perf_counter()   # the metrics are Python floats (synchronised)
                if args.eval_mode == "full":
                    val_loss, val_score = evaluate_full(model, x_full, adj_norm_t, labels_t, graph.val_idx, metric, graph.problem_type)
                    val_eval_s = time.perf_counter() - val_start
                    test_loss, test_score = evaluate_full(model, x_full, adj_norm_t, labels_t, graph.test_idx, metric, graph.problem_type)
                elif args.eval_mode == "full_cpu":
                    # one whole-graph forward gives both splits (as evaluate_large); it is counted as validation time
                    logits_cpu = predict_full_cpu(model, graph, cpu_cache)
                    val_loss, val_score = scores_from_logits(logits_cpu, labels_t_cpu, graph.val_idx, metric, graph.problem_type)
                    val_eval_s = time.perf_counter() - val_start
                    test_loss, test_score = scores_from_logits(logits_cpu, labels_t_cpu, graph.test_idx, metric, graph.problem_type)
                    del logits_cpu
                else:
                    val_loss, val_score = evaluate_batch(model, graph, labels_t_cpu, graph.val_idx, metric, graph.problem_type, device, args.eval_batch_size)
                    val_eval_s = time.perf_counter() - val_start
                    test_loss, test_score = evaluate_batch(model, graph, labels_t_cpu, graph.test_idx, metric, graph.problem_type, device, args.eval_batch_size)
                if val_score > best_val:
                    best_val = val_score
                    best_test = test_score
                    best_epoch = epoch
                    bad_eval_rounds = 0
                    torch.save(model.state_dict(), out_dir / "best_model.pt")
                else:
                    bad_eval_rounds += 1
            cuda_sync()
            complete_epoch_s = time.perf_counter() - epoch_start
            mem = peak_memory_mb(device)

            row = {
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
            writer.writerow(row)
            f.flush()

            if do_eval:
                print(
                    f"[Epoch {epoch:04d}] train_loss={train_loss:.6f} "
                    f"val_{metric}={val_score:.6f} test_{metric}={test_score:.6f} "
                    f"best_val={best_val:.6f} best_test={best_test:.6f} "
                    f"train_s={train_epoch_s:.4f} complete_s={complete_epoch_s:.4f} mem_mb={mem:.2f}"
                )
            else:
                print(
                    f"[Epoch {epoch:04d}] train_loss={train_loss:.6f} "
                    f"train_s={train_epoch_s:.4f} complete_s={complete_epoch_s:.4f} mem_mb={mem:.2f}"
                )

            if args.patience > 0 and bad_eval_rounds >= args.patience:
                print(f"[EarlyStop] no improvement for {bad_eval_rounds} eval rounds; stop at epoch={epoch}")
                break

    train_times = [r["train_epoch_s"] for r in history]
    complete_times = [r["complete_epoch_s"] for r in history]
    mems = [r["peak_mem_mb"] for r in history]
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "metric": metric,
        "problem_type": graph.problem_type,
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "best_test_at_best_val": best_test,
        "best_val_score_percent": best_val * 100.0,
        "best_test_at_best_val_percent": best_test * 100.0,
        "epochs_run": len(history),
        "avg_train_epoch_s": float(np.mean(train_times)) if train_times else float("nan"),
        "avg_complete_epoch_s": float(np.mean(complete_times)) if complete_times else float("nan"),
        "avg_peak_mem_mb": float(np.mean(mems)) if mems else float("nan"),
        "max_peak_mem_mb": float(np.max(mems)) if mems else float("nan"),
        "params": num_params,
        # training + validation time of epochs 1..best_epoch (test evaluation / checkpoint saving excluded)
        "time_to_best_s": float(sum(r["train_epoch_s"] + r["val_eval_s"] for r in history if r["epoch"] <= best_epoch)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[Summary] " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
