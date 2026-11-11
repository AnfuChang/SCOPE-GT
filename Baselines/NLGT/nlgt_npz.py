#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def contiguous_remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = labels.astype(np.int64, copy=True)
    valid = labels >= 0
    uniq = np.unique(labels[valid])
    mapping = {int(old): int(new) for new, old in enumerate(uniq.tolist())}
    remapped = np.full_like(labels, fill_value=-1)
    for old, new in mapping.items():
        remapped[labels == old] = new
    return remapped, mapping



def load_graph_npz(data_root: Path):
    adj = sp.load_npz(data_root / "adj.npz").tocsr().astype(np.float32)
    feat = np.load(data_root / "feat.npy")
    labels = np.load(data_root / "label.npy")

    if feat.dtype != np.float32:
        feat = feat.astype(np.float32)
    if labels.dtype not in (np.int64, np.int32):
        labels = labels.astype(np.int64)
    else:
        labels = labels.astype(np.int64, copy=False)

    if adj.shape[0] != feat.shape[0] or feat.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch: adj={adj.shape}, feat={feat.shape}, label={labels.shape}"
        )

    # Sampling uses explicit neighbor lists; self-loops are unnecessary here.
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    labels, label_mapping = contiguous_remap_labels(labels)
    return adj, feat, labels, label_mapping



def stratified_622_split(labels: np.ndarray, seed: int):
    labeled = np.where(labels >= 0)[0]
    y = labels[labeled]

    train_idx, temp_idx = train_test_split(
        labeled, test_size=0.4, random_state=seed, stratify=y
    )
    y_temp = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)



def load_or_make_splits(data_root: Path, labels: np.ndarray, seed: int, resplit_622: bool):
    train_p = data_root / "train.npy"
    val_p = data_root / "val.npy"
    test_p = data_root / "test.npy"

    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raise FileNotFoundError(
            "This script now requires a pre-generated fixed split: train.npy / val.npy / test.npy"
        )

    train_idx = np.load(train_p).astype(np.int64)
    val_idx = np.load(val_p).astype(np.int64)
    test_idx = np.load(test_p).astype(np.int64)
    return train_idx, val_idx, test_idx



def maybe_save_splits(data_root: Path, train_idx, val_idx, test_idx):
    return None


# -----------------------------
# Sampling
# -----------------------------


class GraphSampler:
    def __init__(self, adj: sp.csr_matrix, fanouts: Sequence[int]):
        self.adj = adj.tocsr()
        self.fanouts = list(fanouts)
        self.neighbors = [
            self.adj.indices[self.adj.indptr[i] : self.adj.indptr[i + 1]].tolist()
            for i in range(self.adj.shape[0])
        ]

    def sample_subgraph(self, center: int, rng: np.random.Generator) -> List[int]:
        seq = [center]
        used = {center}
        frontier = [center]

        for fanout in self.fanouts:
            next_frontier = []
            for parent in frontier:
                cand = [v for v in self.neighbors[parent] if v not in used]
                if not cand:
                    continue
                if len(cand) > fanout:
                    chosen = rng.choice(np.asarray(cand, dtype=np.int64), size=fanout, replace=False).tolist()
                else:
                    chosen = cand
                for v in chosen:
                    if v not in used:
                        used.add(v)
                        seq.append(v)
                        next_frontier.append(v)
            frontier = next_frontier
            if not frontier:
                break
        return seq


class SampledNodeDataset(Dataset):
    def __init__(
        self,
        centers: np.ndarray,
        feat: np.ndarray,
        labels: np.ndarray,
        sampler: GraphSampler,
        max_nodes: int,
        num_classes: int,
        seed: int,
        strict_no_neighbor_labels: bool = False,
        train_mask: Optional[np.ndarray] = None,
    ):
        self.centers = np.asarray(centers, dtype=np.int64)
        self.feat = feat
        self.labels = labels
        self.sampler = sampler
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.seed = seed
        self.strict_no_neighbor_labels = strict_no_neighbor_labels
        if train_mask is None:
            train_mask = np.zeros(labels.shape[0], dtype=np.bool_)
        self.train_mask = np.asarray(train_mask, dtype=np.bool_)
        if self.train_mask.shape[0] != labels.shape[0]:
            raise ValueError("train_mask must have the same length as labels")
        self.cls_token_id = num_classes
        self.pad_token_id = num_classes + 1

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx: int):
        center = int(self.centers[idx])
        rng = np.random.default_rng(self.seed + idx * 10007 + center * 97)
        seq = self.sampler.sample_subgraph(center, rng)
        seq = seq[: self.max_nodes]
        n = len(seq)

        x = np.zeros((self.max_nodes, self.feat.shape[1]), dtype=np.float32)
        x[:n] = self.feat[seq]

        valid_tokens = np.zeros((self.max_nodes,), dtype=np.bool_)
        valid_tokens[:n] = True

        label_ids = np.full((self.max_nodes,), self.pad_token_id, dtype=np.int64)
        if not self.strict_no_neighbor_labels:
            sampled_labels = self.labels[seq].astype(np.int64, copy=False)
            # Leakage guard: label-as-token may expose labels only for official
            # training nodes. Validation/test/unlabeled sampled nodes are forced
            # to PAD, and the center label is always PAD even during training.
            is_train = self.train_mask[seq]
            sampled_labels = np.where(
                is_train & (sampled_labels >= 0) & (sampled_labels < self.num_classes),
                sampled_labels,
                self.pad_token_id,
            )
            label_ids[:n] = sampled_labels
        label_ids[0] = self.pad_token_id

        target = int(self.labels[center])
        return x, label_ids, valid_tokens, target, center


# -----------------------------
# Model
# -----------------------------


class NeighborMaskedMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, token_valid: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], token_valid: [B, N] bool
        bsz, num_tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        key_valid = token_valid[:, None, None, :]     # [B,1,1,N]
        query_valid = token_valid[:, None, :, None]   # [B,1,N,1]
        attn_valid = query_valid & key_valid

        # Use a large negative number instead of -inf to avoid softmax(all -inf) -> NaN.
        scores = scores.masked_fill(~attn_valid, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn * query_valid.to(attn.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Clear outputs on PAD queries so residual/LN do not re-inject junk.
        out = out * token_valid.unsqueeze(-1).to(out.dtype)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class NLGTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeighborMaskedMHA(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, token_valid: torch.Tensor) -> torch.Tensor:
        token_valid_f = token_valid.unsqueeze(-1).to(x.dtype)
        x = x + self.attn(self.norm1(x), token_valid)
        x = x * token_valid_f
        ff = self.ffn(self.norm2(x))
        ff = torch.nan_to_num(ff, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + ff
        x = x * token_valid_f
        return x


class NLGTModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hid_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_token_id = num_classes
        self.pad_token_id = num_classes + 1

        self.input_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.label_embed = nn.Embedding(num_classes + 2, hid_dim)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [NLGTBlock(hid_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hid_dim)
        self.head = nn.Linear(hid_dim, num_classes, bias=True)

        nn.init.trunc_normal_(self.label_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, label_ids: torch.Tensor, token_valid: torch.Tensor) -> torch.Tensor:
        safe_label_ids = torch.where(
            (label_ids >= 0) & (label_ids <= self.pad_token_id),
            label_ids,
            torch.full_like(label_ids, self.pad_token_id),
        )

        h = self.input_proj(x) + self.label_embed(safe_label_ids)
        h = self.in_drop(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h = h * token_valid.unsqueeze(-1).to(h.dtype)

        for block in self.blocks:
            h = block(h, token_valid)

        h = self.norm(h)
        h = h * token_valid.unsqueeze(-1).to(h.dtype)
        logits = self.head(h[:, 0, :])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return logits


# -----------------------------
# Train / Eval
# -----------------------------


@torch.no_grad()
def predict_with_votes(
    model: nn.Module,
    centers: np.ndarray,
    feat: np.ndarray,
    labels: np.ndarray,
    sampler: GraphSampler,
    max_nodes: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    votes: int,
    base_seed: int,
    strict_no_neighbor_labels: bool,
    train_mask: np.ndarray,
) -> np.ndarray:
    model.eval()
    logits_sum = np.zeros((len(centers), num_classes), dtype=np.float64)

    for vote_id in range(votes):
        ds = SampledNodeDataset(
            centers=centers,
            feat=feat,
            labels=labels,
            sampler=sampler,
            max_nodes=max_nodes,
            num_classes=num_classes,
            seed=base_seed + vote_id * 1000003,
            strict_no_neighbor_labels=strict_no_neighbor_labels,
            train_mask=train_mask,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        start = 0
        for batch in loader:
            x, label_ids, token_valid, _, _ = batch
            x = x.to(device, non_blocking=True)
            label_ids = label_ids.to(device, non_blocking=True)
            token_valid = token_valid.to(device, non_blocking=True)
            logits = model(x, label_ids, token_valid).detach().cpu().numpy()
            logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            bs = logits.shape[0]
            logits_sum[start : start + bs] += logits
            start += bs

    return logits_sum / float(votes)



def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float((pred == targets).mean())


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def resolve_metric(metric: str, dataset: str) -> str:
    metric = metric.lower()
    if metric == "auto":
        return "rocauc" if dataset.lower().replace("-", "_") in AUC_DATASETS else "acc"
    return "rocauc" if metric in ("auc", "roc_auc", "rocauc") else "acc"


def metric_name(metric: str) -> str:
    return "ROC_AUC" if metric == "rocauc" else "Acc"


def score_from_logits(logits: np.ndarray, targets: np.ndarray, metric: str) -> float:
    """logits: vote-averaged logits [n, C]; accuracy or ROC-AUC of softmax(logits)."""
    if metric == "acc":
        return accuracy_from_logits(logits, targets)
    from sklearn.metrics import roc_auc_score
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    if prob.shape[1] == 2:
        return float(roc_auc_score(targets, prob[:, 1]))  # score of the positive class
    return float(roc_auc_score(targets, prob, multi_class="ovr"))


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    best_val_acc: float
    best_test_acc: float
    best_epoch_train_time_s: float
    best_epoch_eval_time_s: float
    best_epoch_total_time_s: float
    best_epoch_train_peak_mem_mb: float
    best_epoch_eval_peak_mem_mb: float
    best_epoch_total_peak_mem_mb: float
    avg_epoch_time_s: float
    max_peak_mem_mb: float
    params: int = 0                      # trainable parameters
    train_time_per_epoch_s: float = 0.0  # training only, mean over epochs
    time_to_best_s: float = 0.0          # training + validation time of epochs 1..best_epoch


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



def build_optimizer(args, model: nn.Module):
    name = args.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")



def maybe_sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def run_one(
    run_seed: int,
    args,
    adj: sp.csr_matrix,
    feat: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> RunResult:
    set_seed(run_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sampler = GraphSampler(adj, args.fanouts)
    num_classes = int(labels[labels >= 0].max()) + 1

    model = NLGTModel(
        in_dim=feat.shape[1],
        num_classes=num_classes,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * args.lrf, 1e-8)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    best_epoch_train_time = 0.0
    best_epoch_eval_time = 0.0
    best_epoch_total_time = 0.0
    best_epoch_train_mem_mb = 0.0
    best_epoch_eval_mem_mb = 0.0
    best_epoch_total_mem_mb = 0.0
    best_state = None
    bad_epochs = 0
    epoch_times: List[float] = []
    epoch_peak_mems: List[float] = []
    train_times: List[float] = []
    train_val_times: List[float] = []   # per epoch: training + validation (no test) -> time to best

    val_targets = labels[val_idx]
    test_targets = labels[test_idx]
    train_mask = np.zeros(labels.shape[0], dtype=np.bool_)
    train_mask[train_idx] = True

    for epoch in range(1, args.epochs + 1):
        ds = SampledNodeDataset(
            centers=train_idx,
            feat=feat,
            labels=labels,
            sampler=sampler,
            max_nodes=args.max_nodes,
            num_classes=num_classes,
            seed=run_seed + epoch * 7919,
            strict_no_neighbor_labels=args.strict_no_neighbor_labels,
            train_mask=train_mask,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )

        model.train()
        epoch_loss = 0.0
        seen = 0
        correct = 0
        skipped_batches = 0
        processed_batches = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        maybe_sync_cuda(device)
        t0 = time.perf_counter()

        for batch_id, batch in enumerate(loader, start=1):
            x, label_ids, token_valid, target, _ = batch
            x = x.to(device, non_blocking=True)
            label_ids = label_ids.to(device, non_blocking=True)
            token_valid = token_valid.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, label_ids, token_valid)

            if not torch.isfinite(logits).all():
                skipped_batches += 1
                if skipped_batches <= 3:
                    print(f"[seed={run_seed}] epoch={epoch:03d} batch={batch_id:04d} non-finite logits detected; batch skipped")
                continue

            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                skipped_batches += 1
                if skipped_batches <= 3:
                    print(f"[seed={run_seed}] epoch={epoch:03d} batch={batch_id:04d} non-finite loss detected; batch skipped")
                continue

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            bs = target.size(0)
            epoch_loss += float(loss.item()) * bs
            seen += bs
            correct += int((logits.argmax(dim=1) == target).sum().item())
            processed_batches += 1

        scheduler.step()
        maybe_sync_cuda(device)
        train_time = time.perf_counter() - t0
        train_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )

        if processed_batches == 0:
            raise RuntimeError(
                "All training batches were skipped due to non-finite activations/loss. "
                "Try lowering --lr or inspecting the input data."
            )

        train_acc = correct / max(seen, 1)
        train_loss = epoch_loss / max(seen, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        eval_time = 0.0
        val_time = 0.0
        eval_peak_mem_mb = 0.0
        if epoch % args.eval_every == 0:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            maybe_sync_cuda(device)
            eval_t0 = time.perf_counter()
            val_logits = predict_with_votes(
                model=model,
                centers=val_idx,
                feat=feat,
                labels=labels,
                sampler=sampler,
                max_nodes=args.max_nodes,
                num_classes=num_classes,
                batch_size=args.eval_batch_size,
                device=device,
                votes=args.eval_votes,
                base_seed=run_seed + epoch * 104729,
                strict_no_neighbor_labels=args.strict_no_neighbor_labels,
                train_mask=train_mask,
            )
            val_time = time.perf_counter() - eval_t0   # predict_with_votes returns numpy (synchronised)
            test_logits = predict_with_votes(
                model=model,
                centers=test_idx,
                feat=feat,
                labels=labels,
                sampler=sampler,
                max_nodes=args.max_nodes,
                num_classes=num_classes,
                batch_size=args.eval_batch_size,
                device=device,
                votes=args.eval_votes,
                base_seed=run_seed + epoch * 15485863,
                strict_no_neighbor_labels=args.strict_no_neighbor_labels,
                train_mask=train_mask,
            )
            maybe_sync_cuda(device)
            eval_time = time.perf_counter() - eval_t0
            eval_peak_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            )
            # val_acc / test_acc hold args.metric (accuracy, or ROC-AUC on the AUC datasets)
            val_acc = score_from_logits(val_logits, val_targets, args.metric)
            test_acc = score_from_logits(test_logits, test_targets, args.metric)
        else:
            val_acc = float("nan")
            test_acc = float("nan")
        total_time = train_time + eval_time
        total_peak_mem_mb = max(train_peak_mem_mb, eval_peak_mem_mb)
        epoch_times.append(float(total_time))
        epoch_peak_mems.append(float(total_peak_mem_mb))
        train_times.append(float(train_time))
        train_val_times.append(float(train_time + val_time))

        best_str = f"best_val={best_val:.4f}" if best_epoch > 0 else "best_val=NA"
        name = metric_name(args.metric)
        print(
            f"[seed={run_seed}] epoch={epoch:03d} lr={current_lr:.3e} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_{name}={val_acc if not math.isnan(val_acc) else -1:.4f} "
            f"test_{name}={test_acc if not math.isnan(test_acc) else -1:.4f} batches={processed_batches} "
            f"skipped={skipped_batches} total_time={total_time:.3f}s(train={train_time:.3f},eval={eval_time:.3f}) "
            f"peak_mem={total_peak_mem_mb:.1f}MB(train={train_peak_mem_mb:.1f},eval={eval_peak_mem_mb:.1f}) {best_str}"
        )

        improved = (not math.isnan(val_acc)) and (val_acc > best_val)
        if improved:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            best_epoch_train_time = train_time
            best_epoch_eval_time = eval_time
            best_epoch_total_time = total_time
            best_epoch_train_mem_mb = train_peak_mem_mb
            best_epoch_eval_mem_mb = eval_peak_mem_mb
            best_epoch_total_mem_mb = total_peak_mem_mb
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            if not math.isnan(val_acc):
                bad_epochs += args.eval_every

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return RunResult(
        seed=run_seed,
        best_epoch=best_epoch,
        best_val_acc=best_val,
        best_test_acc=best_test,
        best_epoch_train_time_s=best_epoch_train_time,
        best_epoch_eval_time_s=best_epoch_eval_time,
        best_epoch_total_time_s=best_epoch_total_time,
        best_epoch_train_peak_mem_mb=best_epoch_train_mem_mb,
        best_epoch_eval_peak_mem_mb=best_epoch_eval_mem_mb,
        best_epoch_total_peak_mem_mb=best_epoch_total_mem_mb,
        avg_epoch_time_s=float(np.mean(epoch_times)) if epoch_times else 0.0,
        max_peak_mem_mb=float(max(epoch_peak_mems, default=0.0)),
        params=num_params,
        train_time_per_epoch_s=float(np.mean(train_times)) if train_times else 0.0,
        time_to_best_s=float(sum(train_val_times[:max(best_epoch, 0)])),
    )


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Stable NLGT adaptation for unified npz/npy graph datasets")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="dataset name (default: basename of --data-root)")
    parser.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "auc", "roc_auc"],
                        help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    parser.add_argument("--out-dir", type=str, default="./results/nlgt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--resplit-622", action="store_true", help="Deprecated and ignored. This script always reads train.npy/val.npy/test.npy.")
    parser.add_argument("--save-splits", action="store_true", help="Deprecated and ignored.")

    parser.add_argument("--fanouts", type=int, nargs="+", default=[10, 5, 2])
    parser.add_argument("--max-nodes", type=int, default=168)
    parser.add_argument(
        "--strict-no-neighbor-labels",
        action="store_true",
        help="Disable NLGT label-as-token input. By default label-as-token is enabled, but nlgt_npz.py masks center, validation, test and unlabeled node labels to prevent leakage.",
    )

    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-votes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)

    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)

    data_root = Path(args.data_root)
    if args.dataset is None:
        args.dataset = data_root.name
    args.metric = resolve_metric(args.metric, args.dataset)
    name = metric_name(args.metric)
    adj, feat, labels, label_mapping = load_graph_npz(data_root)
    results = []
    num_classes = int(labels[labels >= 0].max()) + 1

    print("==== NLGT adaptation (stable) ====")
    print(f"data_root={data_root} dataset={args.dataset} metric={name}")
    print(f"nodes={feat.shape[0]} edges={adj.nnz} feat_dim={feat.shape[1]} classes={num_classes}")
    print(f"optimizer={args.optimizer} lr={args.lr} dropout={args.dropout} grad_clip={args.grad_clip}")
    print(f"official-style neighbor-label input={'off' if args.strict_no_neighbor_labels else 'on'}")
    if args.strict_no_neighbor_labels:
        print("neighbor labels are fully masked as PAD tokens")
    print(f"label remap={label_mapping}")

    for run_id in range(args.runs):
        seed = args.base_seed + run_id
        train_idx, val_idx, test_idx = load_or_make_splits(data_root, labels, seed, args.resplit_622)
        print(f"\n[Run {run_id + 1}/{args.runs}] seed={seed}")
        print(f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        run_result = run_one(
            run_seed=seed,
            args=args,
            adj=adj,
            feat=feat,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        results.append(run_result)
        print(
            f"[Run {run_id + 1}] best_epoch={run_result.best_epoch} best_val_{name}={run_result.best_val_acc:.4f} "
            f"best_test_{name}={run_result.best_test_acc:.4f} best_epoch_total_time={run_result.best_epoch_total_time_s:.3f}s "
            f"best_epoch_total_peak_mem={run_result.best_epoch_total_peak_mem_mb:.1f}MB "
            f"time_to_best={run_result.time_to_best_s:.2f}s"
        )

    summary = {
        "runs": [r.__dict__ for r in results],
        "metric": args.metric,  # the *_acc fields below hold this metric
        "mean_best_test_acc": float(np.mean([r.best_test_acc for r in results])),
        "std_best_test_acc": float(np.std([r.best_test_acc for r in results], ddof=0)),
        "mean_best_epoch": float(np.mean([r.best_epoch for r in results])),
        "mean_best_val_acc": float(np.mean([r.best_val_acc for r in results])),
        "mean_best_epoch_train_time_s": float(np.mean([r.best_epoch_train_time_s for r in results])),
        "mean_best_epoch_eval_time_s": float(np.mean([r.best_epoch_eval_time_s for r in results])),
        "mean_best_epoch_total_time_s": float(np.mean([r.best_epoch_total_time_s for r in results])),
        "mean_best_epoch_train_peak_mem_mb": float(np.mean([r.best_epoch_train_peak_mem_mb for r in results])),
        "mean_best_epoch_eval_peak_mem_mb": float(np.mean([r.best_epoch_eval_peak_mem_mb for r in results])),
        "mean_best_epoch_total_peak_mem_mb": float(np.mean([r.best_epoch_total_peak_mem_mb for r in results])),
        "mean_avg_epoch_time_s": float(np.mean([r.avg_epoch_time_s for r in results])),
        "max_peak_mem_mb": float(max(r.max_peak_mem_mb for r in results)),
        "config": vars(args),
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
    }

    print("\n==== Summary ====")
    print(f"Test {name}: {summary['mean_best_test_acc'] * 100:.2f} \u00b1 {summary['std_best_test_acc'] * 100:.2f}")
    print(f"Tepoch (s): {summary['mean_avg_epoch_time_s']:.2f}")
    print(f"Mem. (MB): {summary['max_peak_mem_mb']:.2f}")
    print_efficiency(results[0].params, [r.train_time_per_epoch_s for r in results],
                     [r.time_to_best_s for r in results], summary["max_peak_mem_mb"],
                     best_epoch=[r.best_epoch for r in results])
    return

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n==== Summary ====")
    print(f"Test Acc @ convergence: {summary['mean_best_test_acc']:.4f} ± {summary['std_best_test_acc']:.4f}")
    print(f"Mean best epoch: {summary['mean_best_epoch']:.2f}")
    print(f"Mean best-epoch total time: {summary['mean_best_epoch_total_time_s']:.3f}s")
    print(f"  train/eval time: {summary['mean_best_epoch_train_time_s']:.3f}s / {summary['mean_best_epoch_eval_time_s']:.3f}s")
    print(f"Mean best-epoch total peak mem: {summary['mean_best_epoch_total_peak_mem_mb']:.2f}MB")
    print(f"  train/eval peak mem: {summary['mean_best_epoch_train_peak_mem_mb']:.2f}MB / {summary['mean_best_epoch_eval_peak_mem_mb']:.2f}MB")
    print(f"Saved to: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
