"""Run VCR-Graphormer as a BILGT-compatible baseline.

This entry point keeps the official VCR model architecture while replacing the
original dataset loaders with a BILGT-style loader and a reproducible 5-run
baseline protocol.  It reports:
  Test Acc(%) | training average Tepoch(s) | training max Mem.(MB)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from lr import PolynomialDecayLR
from model import TransformerModel
from vcr_bilgt_data import load_bilgt_dataset
from vcr_tokenization import TokenizerConfig, build_or_load_tokens


class TokenDataset(Data.Dataset):
    def __init__(self, token_array: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.tokens = token_array
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        # np.asarray keeps RAM arrays zero-copy and materializes memmap slices cheaply.
        x = np.asarray(self.tokens[idx], dtype=np.float32)
        y = np.int64(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BILGT-compatible VCR-Graphormer baseline")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing BILGT-format graph files")
    p.add_argument("--output_dir", type=str, default="results_vcr_bilgt")
    p.add_argument("--cache_dir", type=str, default=None, help="Token cache dir. Default: <data_dir>/VCRGraphormer_cache")
    p.add_argument("--device", type=str, default="0", help="CUDA id, cuda:0, or cpu")
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--to_undirected", type=int, default=1)

    # VCR tokenization parameters.
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--num_structure_supernodes", type=int, default=20)
    p.add_argument("--num_structure_tokens", type=int, default=5)
    p.add_argument("--num_content_tokens", type=int, default=5)
    p.add_argument("--partition_method", type=str, default="auto", choices=["auto", "metis", "bfs", "hash"])
    p.add_argument("--content_label_source", type=str, default="train", choices=["train", "trainval", "non_test", "all"])
    p.add_argument("--ppr_alpha", type=float, default=0.85)
    p.add_argument("--structure_ppr_eps", type=float, default=1e-7)
    p.add_argument("--content_ppr_eps", type=float, default=1e-2)
    p.add_argument("--structure_ppr_max_itr", type=int, default=10000)
    p.add_argument("--content_ppr_max_itr", type=int, default=100)
    p.add_argument("--force_retokenize", action="store_true")
    p.add_argument("--include_preprocess_in_time", action="store_true", help="Add one-time tokenization time to run-1 epoch timing. Off by default.")
    p.add_argument("--load_tokens_to_ram", type=str, default="auto", choices=["auto", "0", "1"], help="auto loads token cache to RAM if <= --ram_token_limit_gb")
    p.add_argument("--ram_token_limit_gb", type=float, default=4.0)

    # Model parameters, matching official implementation names.
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ffn_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attention_dropout", type=float, default=0.1)

    # Training parameters.
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--eval_batch_size", type=int, default=0, help="Default: batch_size")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--peak_lr", type=float, default=0.001)
    p.add_argument("--end_lr", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_updates", type=int, default=400)
    p.add_argument("--tot_updates", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "auc", "roc_auc"],
                   help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    args = p.parse_args()
    metric = args.metric.lower()
    if metric == "auto":
        metric = "rocauc" if args.dataset.lower().replace("-", "_") in AUC_DATASETS else "acc"
    args.metric = "rocauc" if metric in ("auc", "roc_auc", "rocauc") else "acc"
    return args


def resolve_device(device_arg: str) -> torch.device:
    s = str(device_arg).lower()
    if s == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if s.startswith("cuda"):
        return torch.device(s)
    return torch.device(f"cuda:{int(s)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    y = labels.reshape(-1).astype(np.int64, copy=False)
    valid = np.unique(y[y >= 0])
    mapping = {int(old): int(new) for new, old in enumerate(valid.tolist())}
    out = np.full_like(y, -1)
    for old, new in mapping.items():
        out[y == old] = new
    return out, mapping


def accuracy_count(log_probs: torch.Tensor, labels: torch.Tensor) -> int:
    pred = log_probs.argmax(dim=1)
    return int((pred == labels).sum().item())


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def metric_label(metric: str) -> str:
    return "ROC_AUC" if metric == "rocauc" else "Acc"


def print_efficiency(params, train_epoch_s, time_to_best_s, peak_mem_mb, pre_s=None, best_epoch=None) -> None:
    """Efficiency block with the same labels/definitions in every baseline (see common/protocol.py):
    Params = trainable parameters; Train time/epoch = training only (no evaluation), mean over epochs
    and runs; Time to best = training + validation time from epoch 1 to the end of the best-validation
    epoch (test evaluation and one-off preprocessing excluded), mean ± std over runs; Peak GPU mem =
    torch.cuda.max_memory_allocated, max over runs."""
    ttb = np.asarray(time_to_best_s, dtype=np.float64)
    print("==== Efficiency ====")
    print(f"Params: {int(params):,} ({int(params) / 1e6:.3f}M)")
    print(f"Train time/epoch (s): {float(np.mean(train_epoch_s)):.4f}")
    at = f" (best epoch {float(np.mean(best_epoch)):.1f})" if best_epoch is not None else ""
    print(f"Time to best (s): {ttb.mean():.2f} ± {(ttb.std(ddof=1) if ttb.size > 1 else 0.0):.2f}{at}")
    if pre_s is not None:
        pre = np.asarray(pre_s, dtype=np.float64)
        print(f"Preprocessing (s): {pre.mean():.2f}")
        print(f"Time to best incl. preprocessing (s): {(ttb + pre).mean():.2f}")
    print(f"Peak GPU mem (MB): {peak_mem_mb:.2f}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Data.DataLoader, device: torch.device,
             metric: str = "acc") -> Tuple[float, float]:
    """Returns (loss, score); score is accuracy, or ROC-AUC of P(class 1) when metric='rocauc'."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    probs, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)  # log-probabilities
        loss = F.nll_loss(out, yb, reduction="sum")
        total_loss += float(loss.item())
        total_correct += accuracy_count(out, yb)
        total += int(yb.numel())
        if metric == "rocauc":
            probs.append(out.detach().float().exp().cpu())
            targets.append(yb.detach().cpu())
    if metric == "rocauc":
        from sklearn.metrics import roc_auc_score
        prob = torch.cat(probs).numpy()
        true = torch.cat(targets).numpy()
        auc = roc_auc_score(true, prob[:, 1]) if prob.shape[1] == 2 else roc_auc_score(true, prob, multi_class="ovr")
        return total_loss / max(total, 1), float(auc)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_run(args: argparse.Namespace, tokens: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, seed: int, device: torch.device) -> Dict[str, float]:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    eval_bs = args.eval_batch_size if args.eval_batch_size and args.eval_batch_size > 0 else args.batch_size
    train_ds = TokenDataset(tokens, labels, train_idx)
    val_ds = TokenDataset(tokens, labels, val_idx)
    test_ds = TokenDataset(tokens, labels, test_idx)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), generator=generator)
    val_loader = Data.DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = Data.DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    token_length = int(tokens.shape[1] - 1)
    num_classes = int(labels[labels >= 0].max() + 1)
    model = TransformerModel(
        token_length=token_length,
        n_class=num_classes,
        input_dim=int(tokens.shape[2]),
        n_layers=args.n_layers,
        num_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        dropout_rate=args.dropout,
        attention_dropout_rate=args.attention_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    epoch_times: List[float] = []   # training + validation (no test): prefix sums give the time to best
    train_times: List[float] = []   # training only

    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[Run seed={seed}] model params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.nll_loss(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += float(loss.item()) * int(yb.numel())
            train_correct += accuracy_count(out.detach(), yb)
            train_total += int(yb.numel())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_times.append(time.perf_counter() - t0)
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == 0) or (epoch + 1 == args.epochs)
        if do_eval:
            # val_acc holds args.metric (accuracy, or ROC-AUC on the AUC datasets)
            val_loss, val_acc = evaluate(model, val_loader, device, args.metric)
        else:
            val_loss, val_acc = float("nan"), float("nan")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)
        train_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        if do_eval:
            print(
                f"[Run seed={seed}] Epoch {epoch+1:04d} | "
                f"loss_train={train_loss:.4f} acc_train={train_acc:.4f} | "
                f"loss_val={val_loss:.4f} {metric_label(args.metric)}_val={val_acc:.4f} | "
                f"Tepoch={epoch_time:.4f}s"
            )
        else:
            print(
                f"[Run seed={seed}] Epoch {epoch+1:04d} | "
                f"loss_train={train_loss:.4f} acc_train={train_acc:.4f} | "
                f"Tepoch={epoch_time:.4f}s"
            )

        if do_eval:
            improved = (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss)
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"[Run seed={seed}] Early stopped at epoch {epoch+1}; best epoch={best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device, args.metric)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = 0.0

    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    result = {
        "seed": seed,
        "test_acc_pct": test_acc * 100.0,
        "best_val_acc_pct": best_val_acc * 100.0,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch + 1,
        "stop_epoch": len(epoch_times),
        "train_avg_epoch_s": avg_epoch_s,
        "train_max_mem_mb": float(peak_mem_mb),
        "test_loss": test_loss,
        "params": num_params,
        "train_time_per_epoch_s": float(np.mean(train_times)) if train_times else 0.0,
        # epochs are counted from 0 here: the best epoch e is the (e + 1)-th entry
        "time_to_best_s": float(sum(epoch_times[:best_epoch + 1])) if best_epoch >= 0 else 0.0,
    }
    print(
        f"[Run seed={seed}] Test {metric_label(args.metric)}(%)={result['test_acc_pct']:.4f} | "
        f"训练平均Tepoch(s)={result['train_avg_epoch_s']:.4f} | "
        f"训练最大Mem.(MB)={result['train_max_mem_mb']:.2f} | "
        f"到最佳模型时间(s)={result['time_to_best_s']:.2f}"
    )
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_token_array(token_path: Path, shape: Tuple[int, int, int], args: argparse.Namespace) -> np.ndarray:
    size_gb = np.prod(shape) * np.dtype(np.float32).itemsize / (1024 ** 3)
    if args.load_tokens_to_ram == "1" or (args.load_tokens_to_ram == "auto" and size_gb <= args.ram_token_limit_gb):
        print(f"[Cache] Loading token cache into RAM ({size_gb:.2f} GB): {token_path}")
        return np.load(token_path, mmap_mode=None).astype(np.float32, copy=False)
    print(f"[Cache] Using memory-mapped token cache ({size_gb:.2f} GB): {token_path}")
    return np.load(token_path, mmap_mode="r")


def main() -> None:
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else Path(args.data_dir).expanduser().resolve() / "VCRGraphormer_cache"

    print("[Config]", json.dumps(vars(args), indent=2, sort_keys=True))
    print(f"[Device] {device}")
    data = load_bilgt_dataset(args.data_dir, dataset_name=args.dataset, to_undirected=bool(args.to_undirected))
    labels, label_mapping = remap_labels(data.labels)
    print(
        f"[Data] dataset={data.name} nodes={data.num_nodes} edges={data.adj.nnz} "
        f"features={data.num_features} classes={int(labels.max()+1)} "
        f"split=({len(data.train_idx)},{len(data.val_idx)},{len(data.test_idx)})"
    )
    if label_mapping and sorted(label_mapping.keys()) != list(range(len(label_mapping))):
        print(f"[Data] Remapped labels: {label_mapping}")

    cfg = TokenizerConfig(
        hops=args.hops,
        num_structure_supernodes=args.num_structure_supernodes,
        num_structure_tokens=args.num_structure_tokens,
        num_content_tokens=args.num_content_tokens,
        ppr_alpha=args.ppr_alpha,
        structure_ppr_eps=args.structure_ppr_eps,
        content_ppr_eps=args.content_ppr_eps,
        structure_ppr_max_itr=args.structure_ppr_max_itr,
        content_ppr_max_itr=args.content_ppr_max_itr,
        partition_method=args.partition_method,
        content_label_source=args.content_label_source,
        dtype="float32",
        token_seed=args.seed_start,
    )
    preprocess_t0 = time.perf_counter()
    token_path, shape, meta = build_or_load_tokens(
        dataset_name=args.dataset,
        adj=data.adj,
        features=data.features,
        labels=labels,
        train_idx=data.train_idx,
        val_idx=data.val_idx,
        test_idx=data.test_idx,
        cfg=cfg,
        cache_dir=cache_dir,
        force=args.force_retokenize,
    )
    preprocess_s = time.perf_counter() - preprocess_t0
    print(f"[Tokenize] token_path={token_path} shape={shape} preprocess_s={preprocess_s:.3f}")
    tokens = _load_token_array(token_path, shape, args)

    results: List[Dict[str, float]] = []
    for run in range(args.num_runs):
        seed = args.seed_start + run
        res = train_one_run(args, tokens, labels, data.train_idx, data.val_idx, data.test_idx, seed, device)
        if args.include_preprocess_in_time and run == 0 and res["stop_epoch"] > 0:
            res["train_avg_epoch_s"] += preprocess_s / res["stop_epoch"]
        results.append(res)

    test_acc = np.asarray([r["test_acc_pct"] for r in results], dtype=np.float64)
    epoch_s = np.asarray([r["train_avg_epoch_s"] for r in results], dtype=np.float64)
    mem_mb = np.asarray([r["train_max_mem_mb"] for r in results], dtype=np.float64)
    summary = {
        "dataset": args.dataset,
        "metric": args.metric,  # test_acc_* hold this metric
        "num_runs": args.num_runs,
        "seed_start": args.seed_start,
        "test_acc_mean_pct": float(test_acc.mean()),
        "test_acc_std_pct": float(test_acc.std(ddof=1) if len(test_acc) > 1 else 0.0),
        "train_avg_epoch_s_mean": float(epoch_s.mean()),
        "train_avg_epoch_s_std": float(epoch_s.std(ddof=1) if len(epoch_s) > 1 else 0.0),
        "train_max_mem_mb_mean": float(mem_mb.mean()),
        "train_max_mem_mb_max": float(mem_mb.max()),
        "token_path": str(token_path),
        "preprocess_s": float(preprocess_s),
        "config": vars(args),
        "tokenizer_config": asdict(cfg),
        "runs": results,
    }

    print("\n==== Summary ====")
    print(f"Test {metric_label(args.metric)}: {summary['test_acc_mean_pct']:.2f} \u00b1 {summary['test_acc_std_pct']:.2f}")
    print(f"Tepoch (s): {summary['train_avg_epoch_s_mean']:.2f}")
    print(f"Mem. (MB): {summary['train_max_mem_mb_max']:.2f}")
    # Tepoch above includes the validation check; the efficiency block uses training-only time.
    # Preprocessing = VCR tokenization (structure/content virtual connections), done once for all runs.
    print_efficiency(results[0]["params"], [r["train_time_per_epoch_s"] for r in results],
                     [r["time_to_best_s"] for r in results], summary["train_max_mem_mb_max"],
                     pre_s=[preprocess_s] * len(results), best_epoch=[r["best_epoch"] for r in results])
    return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.dataset}_vcr_bilgt_{stamp}.json"
    csv_path = output_dir / f"{args.dataset}_vcr_bilgt_{stamp}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    with csv_path.open("w", newline="") as f:
        fieldnames = ["dataset", "seed", "test_acc_pct", "best_val_acc_pct", "best_epoch", "stop_epoch", "train_avg_epoch_s", "train_max_mem_mb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["dataset"] = args.dataset
            writer.writerow(row)
        writer.writerow({
            "dataset": args.dataset,
            "seed": "mean/std",
            "test_acc_pct": f"{summary['test_acc_mean_pct']:.4f} ± {summary['test_acc_std_pct']:.4f}",
            "train_avg_epoch_s": f"{summary['train_avg_epoch_s_mean']:.4f} ± {summary['train_avg_epoch_s_std']:.4f}",
            "train_max_mem_mb": f"mean {summary['train_max_mem_mb_mean']:.2f}; max {summary['train_max_mem_mb_max']:.2f}",
        })

    print("\n================ VCR-Graphormer BILGT Final Summary ================")
    print("Test Acc(%)\t训练平均Tepoch(s)\t训练最大Mem.(MB)")
    print(
        f"{summary['test_acc_mean_pct']:.4f} ± {summary['test_acc_std_pct']:.4f}\t"
        f"{summary['train_avg_epoch_s_mean']:.4f} ± {summary['train_avg_epoch_s_std']:.4f}\t"
        f"{summary['train_max_mem_mb_max']:.2f}"
    )
    print(f"[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
