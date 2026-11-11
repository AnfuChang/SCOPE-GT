#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified node-classification baseline suite for datasets stored as:
    adj.npz feat.npy label.npy [train.npy val.npy test.npy]

Supported models:
    - gcn
    - gat
    - graphsage
    - nagphormer       (Hop2Token + Transformer; NrAug explicitly disabled by default)
    - nagphormer_plus  (same backbone + Neighborhood Augmentation / NrAug)

Design goals
------------
1) Be easy to adapt to an existing project dataset format.
2) Use modern PyTorch instead of the very old official TensorFlow codebases of
   GCN / GAT / GraphSAGE.
3) Provide a comparable experiment protocol:
   - default 6:2:2 stratified split
   - 5 runs with different seeds
   - early stopping to approximate convergence
   - report mean/std of test accuracy, convergence epoch, peak GPU memory at
     convergence epoch, and convergence-epoch runtime.

Notes
-----
- This file intentionally focuses on reproducible node-classification baselines.
- FairGT and CoBFormer are not included in this first code drop because they
  introduce extra task assumptions: FairGT needs a meaningful sensitive feature,
  and CoBFormer needs graph partitioning plus collaborative training.
- NAGphormer here is a faithful reimplementation of the paper idea rather than
  a line-by-line port of the official repository.
- NAGphormer and NAGphormer+ are now separated at the CLI level:
  --model nagphormer disables NrAug by default; --model nagphormer_plus
  enables NrAug by default. You can still override this with --nr-aug.
- For NAGphormer, epoch time includes mini-batch token gathering / sampling
  overhead. Offline Hop2Token preprocessing time is also recorded separately.
- For GraphSAGE, training follows neighbor-sampling mini-batches, and the epoch
  time includes seed permutation, neighbor sampling, subgraph materialization,
  CPU->GPU feature transfer, and the train step itself. Validation/test still
  use exact full-batch inference on the whole graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Reproducibility helpers
# =========================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


# =========================================================
# Data loading and split utilities
# =========================================================

def load_npz_adj(path: str | Path) -> sp.csr_matrix:
    adj = sp.load_npz(path).tocsr().astype(np.float32)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")
    if (adj != adj.T).nnz != 0:
        adj = adj.maximum(adj.T)
    return adj


def load_features(path: str | Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 2:
        raise ValueError(f"feat.npy must be 2D, got shape {x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def load_labels(path: str | Path) -> np.ndarray:
    y = np.load(path)
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            row_sums = y.sum(axis=1)
            y = y.argmax(axis=1)
            unlabeled = row_sums <= 0
            if unlabeled.any():
                y = y.astype(np.int64)
                y[unlabeled] = -1
    elif y.ndim != 1:
        raise ValueError(f"label.npy must be 1D or 2D, got shape {y.shape}")
    return y.astype(np.int64, copy=False)


def row_normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.abs(x).sum(axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32, copy=False)


def build_622_splits(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labeled = np.where(labels >= 0)[0]
    y = labels[labeled]
    if len(labeled) == 0:
        raise ValueError("No labeled nodes found in label.npy")

    unique, counts = np.unique(y, return_counts=True)
    can_stratify = np.all(counts >= 5) and len(unique) >= 2

    if can_stratify:
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
        train_rel, tmp_rel = next(sss1.split(labeled, y))
        train_idx = labeled[train_rel]
        tmp_idx = labeled[tmp_rel]
        tmp_y = labels[tmp_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
        val_rel, test_rel = next(sss2.split(tmp_idx, tmp_y))
        val_idx = tmp_idx[val_rel]
        test_idx = tmp_idx[test_rel]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(labeled)
        n = len(perm)
        n_train = int(round(0.6 * n))
        n_val = int(round(0.2 * n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def validate_split_indices(labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if idx.ndim != 1:
            raise ValueError(f"{name}.npy must be 1D, got shape {idx.shape}")
        if len(idx) == 0:
            raise ValueError(f"{name} split is empty")
        if idx.min() < 0 or idx.max() >= len(labels):
            raise ValueError(f"{name} split contains out-of-range indices")
        if np.any(labels[idx] < 0):
            raise ValueError(f"{name} split contains unlabeled nodes (label < 0)")

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train/val/test splits overlap")


def load_or_resplit(root: str | Path, labels: np.ndarray, seed: int, resplit_622: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    train_p = root / "train.npy"
    val_p = root / "val.npy"
    test_p = root / "test.npy"

    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raise FileNotFoundError(
            "This script now requires a pre-generated fixed split: train.npy / val.npy / test.npy"
        )

    train_idx = np.load(train_p)
    val_idx = np.load(val_p)
    test_idx = np.load(test_p)

    validate_split_indices(labels, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


# =========================================================
# Sparse operators and graph preprocessing
# =========================================================

def add_self_loops(adj: sp.csr_matrix) -> sp.csr_matrix:
    n = adj.shape[0]
    return (adj + sp.eye(n, dtype=np.float32, format="csr")).tocsr()


def normalize_adj_sym(adj: sp.csr_matrix, add_self: bool = True) -> sp.csr_matrix:
    a = add_self_loops(adj) if add_self else adj.tocsr()
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5).astype(np.float32)
    d = sp.diags(deg_inv_sqrt)
    return (d @ a @ d).tocsr().astype(np.float32)


def normalize_adj_row(adj: sp.csr_matrix, add_self: bool = False) -> sp.csr_matrix:
    a = add_self_loops(adj) if add_self else adj.tocsr()
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    deg_inv = np.power(np.maximum(deg, 1.0), -1.0).astype(np.float32)
    d = sp.diags(deg_inv)
    return (d @ a).tocsr().astype(np.float32)


def scipy_csr_to_torch_sparse(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = mat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def csr_to_edge_index(adj: sp.csr_matrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coo = adj.tocoo()
    src = torch.tensor(coo.row, dtype=torch.long, device=device)
    dst = torch.tensor(coo.col, dtype=torch.long, device=device)
    return src, dst


def compute_laplacian_pe(adj: sp.csr_matrix, pe_dim: int, dense_fallback_max_n: int = 10000) -> np.ndarray:
    n = adj.shape[0]
    if pe_dim <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    if n <= 2:
        return np.zeros((n, pe_dim), dtype=np.float32)

    a = add_self_loops(adj)
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    d = sp.diags(deg_inv_sqrt)
    lap = sp.eye(n, dtype=np.float32, format="csr") - (d @ a @ d).tocsr()

    k = min(pe_dim + 1, n - 1)
    try:
        vals, vecs = spla.eigsh(lap, k=k, which="SM")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        pe = vecs[:, 1:1 + pe_dim]
    except Exception as exc:
        if n > dense_fallback_max_n:
            raise RuntimeError(
                "Laplacian PE eigsh failed, and dense fallback is disabled for large graphs. "
                f"n={n} exceeds --pe-dense-fallback-max-n={dense_fallback_max_n}. "
                "Please reduce --pe-dim, use a smaller graph, or set --pe-dim 0."
            ) from exc
        warnings.warn(
            "eigsh failed when computing Laplacian PE; falling back to dense eigh. "
            f"This is only allowed because n={n} <= {dense_fallback_max_n}.",
            RuntimeWarning,
        )
        dense = lap.toarray()
        vals, vecs = np.linalg.eigh(dense)
        order = np.argsort(vals)
        vecs = vecs[:, order]
        pe = vecs[:, 1:1 + pe_dim]

    if pe.shape[1] < pe_dim:
        pad = np.zeros((n, pe_dim - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe.astype(np.float32), pad], axis=1)
    return pe.astype(np.float32)


def precompute_hop_tokens(a_hat: sp.csr_matrix, x: np.ndarray, hops: int) -> np.ndarray:
    seq = [x.astype(np.float32)]
    cur = x.astype(np.float32)
    for _ in range(hops):
        cur = a_hat @ cur
        cur = np.asarray(cur, dtype=np.float32)
        seq.append(cur)
    return np.stack(seq, axis=1).astype(np.float32)


@dataclass
class GraphSAGESampledBlock:
    src_nodes: np.ndarray
    num_dst: int
    edge_src: np.ndarray
    edge_dst: np.ndarray


def parse_fanouts(spec: str, num_layers: int) -> List[int]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        raise ValueError("--sage-fanouts must not be empty")
    fanouts = [int(x) for x in items]
    if len(fanouts) == 1 and num_layers > 1:
        fanouts = fanouts * num_layers
    if len(fanouts) != num_layers:
        raise ValueError(
            f"--sage-fanouts expects exactly num_layers={num_layers} integers, got {fanouts}"
        )
    return fanouts


def sample_graphsage_blocks(
    indptr: np.ndarray,
    indices: np.ndarray,
    seed_nodes: np.ndarray,
    fanouts: List[int],
    rng: np.random.Generator,
) -> List[GraphSAGESampledBlock]:
    """Sample GraphSAGE blocks.

    fanouts are ordered from near-to-target hop outward. For a 2-layer GraphSAGE,
    the default "25,10" means: sample up to 25 one-hop neighbors for the target
    batch first, then sample up to 10 neighbors for the expanded frontier.
    The returned blocks are ordered for forward propagation, i.e. outermost block
    first and innermost block last.
    """
    frontier = np.asarray(seed_nodes, dtype=np.int64)
    sampled_blocks: List[GraphSAGESampledBlock] = []

    for fanout in fanouts:
        dst_nodes = frontier
        neigh_samples: List[np.ndarray] = []
        sampled_any = False

        for node in dst_nodes:
            start, end = int(indptr[int(node)]), int(indptr[int(node) + 1])
            neigh = indices[start:end]
            if neigh.size == 0:
                picked = np.empty(0, dtype=np.int64)
            elif fanout < 0 or neigh.size <= fanout:
                picked = neigh.astype(np.int64, copy=False)
            else:
                picked = rng.choice(neigh, size=fanout, replace=False).astype(np.int64, copy=False)
            if picked.size > 0:
                sampled_any = True
            neigh_samples.append(picked)

        if sampled_any:
            sampled_unique = np.unique(np.concatenate([arr for arr in neigh_samples if arr.size > 0]))
        else:
            sampled_unique = np.empty(0, dtype=np.int64)

        extra_mask = ~np.isin(sampled_unique, dst_nodes, assume_unique=False)
        extra_nodes = sampled_unique[extra_mask]
        src_nodes = np.concatenate([dst_nodes, extra_nodes]).astype(np.int64, copy=False)
        local_map = {int(n): i for i, n in enumerate(src_nodes.tolist())}

        total_edges = int(sum(arr.size for arr in neigh_samples))
        edge_src = np.empty(total_edges, dtype=np.int64)
        edge_dst = np.empty(total_edges, dtype=np.int64)
        cursor = 0
        for dst_local, picked in enumerate(neigh_samples):
            n_picked = int(picked.size)
            if n_picked == 0:
                continue
            edge_dst[cursor:cursor + n_picked] = dst_local
            edge_src[cursor:cursor + n_picked] = np.fromiter(
                (local_map[int(v)] for v in picked.tolist()),
                dtype=np.int64,
                count=n_picked,
            )
            cursor += n_picked

        sampled_blocks.append(
            GraphSAGESampledBlock(
                src_nodes=src_nodes,
                num_dst=int(dst_nodes.size),
                edge_src=edge_src[:cursor],
                edge_dst=edge_dst[:cursor],
            )
        )
        frontier = src_nodes

    sampled_blocks.reverse()
    return sampled_blocks


# =========================================================
# Metrics
# =========================================================

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def resolve_metric(metric: str, dataset: str) -> str:
    metric = metric.lower()
    if metric == "auto":
        return "rocauc" if dataset.lower().replace("-", "_") in AUC_DATASETS else "acc"
    return "rocauc" if metric in ("auc", "roc_auc", "rocauc") else "acc"


def metric_label(metric: str) -> str:
    return "ROC_AUC" if metric == "rocauc" else "Acc"


def score(logits: torch.Tensor, y: torch.Tensor, metric: str) -> float:
    if metric == "acc":
        return accuracy(logits, y)
    from sklearn.metrics import roc_auc_score
    prob = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
    true = y.detach().cpu().numpy()
    if prob.shape[1] == 2:
        return float(roc_auc_score(true, prob[:, 1]))  # score of the positive class
    return float(roc_auc_score(true, prob, multi_class="ovr"))


def is_nag_model_name(model_name: str) -> bool:
    return model_name in {"nagphormer", "nagphormer_plus"}


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy for Mixup-style soft labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def apply_nr_aug(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply NAGphormer+ Neighborhood Augmentation on one training mini-batch.

    The implementation follows the paper-level idea: Global Neighborhood
    Augmentation (GNA) mixes two node token sequences and interpolates their
    labels; Local Neighborhood Augmentation (LNA) masks a ratio of hop-token
    columns. The function returns augmented tokens plus optional soft labels.
    If GNA is disabled or skipped, the second return value is None and callers
    should use ordinary hard-label cross entropy.
    """
    if getattr(args, "nr_aug", "off") != "on":
        return tokens, None
    if tokens.numel() == 0:
        return tokens, None
    p_aug = float(getattr(args, "nr_aug_paug", 0.5))
    if p_aug <= 0.0 or torch.rand((), device=tokens.device).item() > p_aug:
        return tokens, None

    aug_tokens = tokens
    soft_labels: Optional[torch.Tensor] = None

    use_gna = bool(getattr(args, "nr_aug_use_gna", True))
    use_lna = bool(getattr(args, "nr_aug_use_lna", True))

    if use_gna and tokens.size(0) >= 2:
        alpha = float(getattr(args, "nr_aug_alpha", 1.0))
        beta = float(getattr(args, "nr_aug_beta", 1.0))
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("--nr-aug-alpha and --nr-aug-beta must be positive")
        beta_dist = torch.distributions.Beta(
            torch.tensor(alpha, dtype=tokens.dtype, device=tokens.device),
            torch.tensor(beta, dtype=tokens.dtype, device=tokens.device),
        )
        lam = beta_dist.sample()
        perm = torch.randperm(tokens.size(0), device=tokens.device)
        aug_tokens = lam * tokens + (1.0 - lam) * tokens[perm]
        hard = F.one_hot(labels, num_classes=num_classes).to(dtype=tokens.dtype, device=tokens.device)
        soft_labels = lam * hard + (1.0 - lam) * hard[perm]

    if use_lna:
        tau = float(getattr(args, "nr_aug_tau", 0.25))
        if tau > 0.0 and aug_tokens.size(1) > 1:
            seq_len = int(aug_tokens.size(1))
            # Paper masks floor((K+1)*tau) columns while ensuring at least one
            # masked column. We additionally avoid masking all hop tokens.
            n_mask = int(seq_len * tau)
            n_mask = max(1, min(n_mask, seq_len - 1))
            scores = torch.rand((aug_tokens.size(0), seq_len), device=aug_tokens.device)
            mask_idx = scores.topk(k=n_mask, dim=1, largest=False).indices
            mask = torch.ones((aug_tokens.size(0), seq_len, 1), dtype=aug_tokens.dtype, device=aug_tokens.device)
            mask.scatter_(1, mask_idx.unsqueeze(-1), 0.0)
            aug_tokens = aug_tokens * mask

    return aug_tokens, soft_labels


# =========================================================
# Model definitions
# =========================================================

class GCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim] + [hid_dim] * max(0, num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = torch.sparse.mm(a_hat, h)
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, a_row: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(a_row, x)
        return self.lin_self(x) + self.lin_neigh(neigh)

    def forward_sampled(
        self,
        x_src: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        x_dst = x_src[:num_dst]
        neigh = torch.zeros((num_dst, x_src.size(1)), dtype=x_src.dtype, device=x_src.device)
        if edge_src.numel() > 0:
            neigh.index_add_(0, edge_dst, x_src[edge_src])
            deg = torch.zeros((num_dst, 1), dtype=x_src.dtype, device=x_src.device)
            deg.index_add_(0, edge_dst, torch.ones((edge_dst.numel(), 1), dtype=x_src.dtype, device=x_src.device))
            neigh = neigh / deg.clamp_min_(1.0)
        return self.lin_self(x_dst) + self.lin_neigh(neigh)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim] + [hid_dim] * max(0, num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([GraphSAGELayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a_row: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, a_row)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_sampled(
        self,
        x_outer: torch.Tensor,
        blocks: List[Tuple[torch.Tensor, torch.Tensor, int]],
    ) -> torch.Tensor:
        h = x_outer
        for i, layer in enumerate(self.layers):
            edge_src, edge_dst, num_dst = blocks[i]
            h = layer.forward_sampled(h, edge_src, edge_dst, num_dst)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.empty(num_heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim * num_heads if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.zeros_(self.bias)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        e, h = scores.shape
        max_init = torch.full((n_nodes, h), -1e30, dtype=scores.dtype, device=scores.device)
        max_per_dst = max_init.scatter_reduce(0, index[:, None].expand(e, h), scores, reduce="amax", include_self=True)
        stable = scores - max_per_dst[index]
        exp_scores = torch.exp(stable)
        denom = torch.zeros((n_nodes, h), dtype=scores.dtype, device=scores.device)
        denom.index_add_(0, index, exp_scores)
        alpha = exp_scores / (denom[index] + 1e-12)
        return alpha

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
        h = self.lin(x).view(-1, self.num_heads, self.out_dim)
        alpha_l = (h * self.attn_l.unsqueeze(0)).sum(-1)
        alpha_r = (h * self.attn_r.unsqueeze(0)).sum(-1)
        e_ij = F.leaky_relu(alpha_l[src] + alpha_r[dst], negative_slope=0.2)
        alpha = self._segment_softmax(e_ij, dst, n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((n_nodes, self.num_heads, self.out_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        if self.concat:
            out = out.reshape(n_nodes, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.hidden = nn.ModuleList()

        if num_layers == 1:
            self.out_layer = GATLayer(in_dim, out_dim, num_heads=1, dropout=dropout, concat=False)
        else:
            self.hidden.append(GATLayer(in_dim, hid_dim, num_heads=num_heads, dropout=dropout, concat=True))
            for _ in range(num_layers - 2):
                self.hidden.append(GATLayer(hid_dim * num_heads, hid_dim, num_heads=num_heads, dropout=dropout, concat=True))
            self.out_layer = GATLayer(hid_dim * num_heads, out_dim, num_heads=1, dropout=dropout, concat=False)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        h = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.hidden:
            h = layer(h, src, dst, n_nodes)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.out_layer(h, src, dst, n_nodes)
        return h


class AttentionReadout(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.scorer = nn.Linear(2 * hid_dim, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z0 = z[:, 0:1, :].expand_as(z)
        score = self.scorer(torch.cat([z0, z], dim=-1)).squeeze(-1)
        attn = torch.softmax(score, dim=1)
        return (attn.unsqueeze(-1) * z).sum(dim=1)


class NAGphormer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_proj = nn.Linear(in_dim, hid_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.readout = AttentionReadout(hid_dim)
        self.cls = nn.Linear(hid_dim, out_dim)

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        z = self.token_proj(token_seq)
        z = self.encoder(z)
        g = self.readout(z)
        return self.cls(g)


# =========================================================
# Training/eval helpers
# =========================================================

@dataclass
class RunMetrics:
    seed: int
    convergence_epoch: int
    best_val_acc: float
    test_acc: float
    peak_mem_mb_at_convergence_epoch: float
    epoch_time_sec_at_convergence_epoch: float
    max_peak_mem_mb: float = 0.0
    avg_epoch_time_sec: float = 0.0
    precompute_time_sec: float = 0.0
    params: int = 0                 # trainable parameters
    time_to_best_sec: float = 0.0   # training + validation time of epochs 1..convergence_epoch


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


@dataclass
class SummaryMetrics:
    model: str
    dataset_root: str
    num_runs: int
    seeds: List[int]
    test_acc_mean: float
    test_acc_std: float
    convergence_epoch_mean: float
    convergence_epoch_std: float
    best_val_acc_mean: float
    peak_mem_mb_mean: float
    epoch_time_sec_mean: float
    precompute_time_sec_mean: float
    runs: List[Dict]


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, metric: float, epoch: int, model: nn.Module) -> bool:
        improved = metric > self.best_metric if self.mode == "max" else metric < self.best_metric
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# =========================================================
# Experiment class
# =========================================================

class Experiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.data_root)
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def load_data(self, seed: int) -> Dict[str, object]:
        adj = load_npz_adj(self.root / "adj.npz")
        feat = load_features(self.root / "feat.npy")
        labels = load_labels(self.root / "label.npy")
        feat = row_normalize_features(feat)

        if adj.shape[0] != feat.shape[0] or feat.shape[0] != labels.shape[0]:
            raise ValueError(f"Shape mismatch: adj={adj.shape}, feat={feat.shape}, labels={labels.shape}")

        train_idx, val_idx, test_idx = load_or_resplit(self.root, labels, seed, self.args.resplit_622)
        n_classes = int(labels[labels >= 0].max()) + 1

        data: Dict[str, object] = {
            "adj": adj,
            "feat": feat,
            "labels": labels,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "n_classes": n_classes,
        }
        return data

    def build_model_and_cache(self, data: Dict[str, object]) -> Tuple[nn.Module, Dict[str, object], float]:
        adj: sp.csr_matrix = data["adj"]  # type: ignore[assignment]
        feat: np.ndarray = data["feat"]  # type: ignore[assignment]
        n_classes: int = data["n_classes"]  # type: ignore[assignment]

        cache: Dict[str, object] = {}
        precompute_time_sec = 0.0

        if self.args.model == "gcn":
            x = torch.tensor(feat, dtype=torch.float32, device=self.device)
            cache["x"] = x
            a_hat = scipy_csr_to_torch_sparse(normalize_adj_sym(adj, add_self=True), self.device)
            cache["a_hat"] = a_hat
            model = GCN(x.size(1), self.args.hid_dim, n_classes, self.args.num_layers, self.args.dropout)
        elif self.args.model == "graphsage":
            x_cpu = torch.from_numpy(feat.astype(np.float32, copy=False)).contiguous()
            cache["x_cpu"] = x_cpu
            cache["x"] = x_cpu.to(self.device)
            a_row = scipy_csr_to_torch_sparse(normalize_adj_row(adj, add_self=False), self.device)
            cache["a_row"] = a_row
            cache["sage_indptr"] = adj.indptr.astype(np.int64, copy=False)
            cache["sage_indices"] = adj.indices.astype(np.int64, copy=False)
            model = GraphSAGE(feat.shape[1], self.args.hid_dim, n_classes, self.args.num_layers, self.args.dropout)
        elif self.args.model == "gat":
            x = torch.tensor(feat, dtype=torch.float32, device=self.device)
            cache["x"] = x
            adj_loop = add_self_loops(adj)
            src, dst = csr_to_edge_index(adj_loop, self.device)
            cache["src"] = src
            cache["dst"] = dst
            model = GAT(x.size(1), self.args.hid_dim, n_classes, self.args.num_layers, self.args.num_heads, self.args.dropout)
        elif is_nag_model_name(self.args.model):
            t0 = time.perf_counter()
            x_np = feat
            if self.args.pe_dim > 0:
                pe = compute_laplacian_pe(adj, self.args.pe_dim, dense_fallback_max_n=self.args.pe_dense_fallback_max_n)
                x_np = np.concatenate([x_np, pe], axis=1).astype(np.float32)
            a_hat_np = normalize_adj_sym(adj, add_self=True)
            token_seq_np = precompute_hop_tokens(a_hat_np, x_np, self.args.hops)
            precompute_time_sec = time.perf_counter() - t0

            token_seq = torch.tensor(token_seq_np, dtype=torch.float32)
            if self.args.nag_cache_on_device:
                token_seq = token_seq.to(self.device)
            cache["token_seq"] = token_seq
            model = NAGphormer(
                in_dim=token_seq_np.shape[-1],
                hid_dim=self.args.hid_dim,
                out_dim=n_classes,
                num_layers=self.args.num_layers,
                num_heads=self.args.num_heads,
                dim_feedforward=self.args.dim_feedforward,
                dropout=self.args.dropout,
                seq_len=token_seq_np.shape[1],
            )
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        return model.to(self.device), cache, precompute_time_sec

    def _nag_batch_to_device(self, token_seq: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if batch_idx is None:
            tokens = token_seq
        else:
            batch_idx_cpu = batch_idx.detach().cpu() if batch_idx.device.type != "cpu" else batch_idx
            tokens = token_seq[batch_idx_cpu]
        if tokens.device != self.device:
            tokens = tokens.to(self.device, non_blocking=(self.device.type == "cuda"))
        return tokens

    def _sample_graphsage_batch(
        self,
        cache: Dict[str, object],
        batch_idx: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, int]]]:
        batch_cpu = batch_idx.detach().cpu() if batch_idx.device.type != "cpu" else batch_idx
        blocks_np = sample_graphsage_blocks(
            cache["sage_indptr"],  # type: ignore[arg-type]
            cache["sage_indices"],  # type: ignore[arg-type]
            batch_cpu.numpy().astype(np.int64, copy=False),
            self.args.sage_fanouts_list,
            rng,
        )
        outer_src_nodes = torch.from_numpy(blocks_np[0].src_nodes)
        x_outer = cache["x_cpu"][outer_src_nodes]  # type: ignore[index]
        x_outer = x_outer.to(self.device, non_blocking=(self.device.type == "cuda"))

        blocks_t: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        for blk in blocks_np:
            edge_src = torch.from_numpy(blk.edge_src).to(self.device, non_blocking=(self.device.type == "cuda"))
            edge_dst = torch.from_numpy(blk.edge_dst).to(self.device, non_blocking=(self.device.type == "cuda"))
            blocks_t.append((edge_src, edge_dst, blk.num_dst))
        return x_outer, blocks_t

    def forward_logits(self, model: nn.Module, cache: Dict[str, object], batch_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.args.model == "gcn":
            return model(cache["x"], cache["a_hat"])  # type: ignore[misc]
        if self.args.model == "graphsage":
            return model(cache["x"], cache["a_row"])  # type: ignore[misc]
        if self.args.model == "gat":
            return model(cache["x"], cache["src"], cache["dst"])  # type: ignore[misc]
        if is_nag_model_name(self.args.model):
            token_seq = cache["token_seq"]
            tokens = self._nag_batch_to_device(token_seq, batch_idx)
            return model(tokens)  # type: ignore[misc]
        raise RuntimeError("unreachable")

    def evaluate_subset(self, model: nn.Module, cache: Dict[str, object], labels_t: torch.Tensor, idx_t: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            if is_nag_model_name(self.args.model):
                bs = self.args.eval_batch_size
                logits_chunks: List[torch.Tensor] = []
                label_chunks: List[torch.Tensor] = []
                for st in range(0, idx_t.numel(), bs):
                    batch = idx_t[st:st + bs]
                    batch_device = batch.to(self.device, non_blocking=(self.device.type == "cuda"))
                    logits_chunks.append(self.forward_logits(model, cache, batch))
                    label_chunks.append(labels_t[batch_device])
                logits = torch.cat(logits_chunks, dim=0)
                labels = torch.cat(label_chunks, dim=0)
                return score(logits, labels, self.args.metric)
            logits = self.forward_logits(model, cache)
            return score(logits[idx_t], labels_t[idx_t], self.args.metric)

    def run_once(self, seed: int) -> RunMetrics:
        set_seed(seed, deterministic=self.args.deterministic)
        data = self.load_data(seed)
        model, cache, precompute_time_sec = self.build_model_and_cache(data)
        num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        labels_np: np.ndarray = data["labels"]  # type: ignore[assignment]
        train_idx: np.ndarray = data["train_idx"]  # type: ignore[assignment]
        val_idx: np.ndarray = data["val_idx"]  # type: ignore[assignment]
        test_idx: np.ndarray = data["test_idx"]  # type: ignore[assignment]

        labels_t = torch.tensor(labels_np, dtype=torch.long, device=self.device)
        if is_nag_model_name(self.args.model):
            train_t = torch.tensor(train_idx, dtype=torch.long)
            val_t = torch.tensor(val_idx, dtype=torch.long)
            test_t = torch.tensor(test_idx, dtype=torch.long)
        elif self.args.model == "graphsage":
            train_t = torch.tensor(train_idx, dtype=torch.long)
            val_t = torch.tensor(val_idx, dtype=torch.long, device=self.device)
            test_t = torch.tensor(test_idx, dtype=torch.long, device=self.device)
        else:
            train_t = torch.tensor(train_idx, dtype=torch.long, device=self.device)
            val_t = torch.tensor(val_idx, dtype=torch.long, device=self.device)
            test_t = torch.tensor(test_idx, dtype=torch.long, device=self.device)

        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        stopper = EarlyStopper(patience=self.args.patience, mode="max")

        best_epoch_time = 0.0
        best_epoch_peak_mem = 0.0
        epoch_times: List[float] = []
        epoch_peak_mems: List[float] = []
        train_val_times: List[float] = []   # per epoch: training + validation (no test) -> time to best

        for epoch in range(1, self.args.epochs + 1):
            model.train()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.perf_counter()

            if is_nag_model_name(self.args.model):
                perm = train_t[torch.randperm(train_t.numel())]
                optimizer.zero_grad(set_to_none=True)
                for st in range(0, perm.numel(), self.args.batch_size):
                    batch = perm[st:st + self.args.batch_size]
                    batch_device = batch.to(self.device, non_blocking=(self.device.type == "cuda"))
                    tokens = self._nag_batch_to_device(cache["token_seq"], batch)
                    hard_labels = labels_t[batch_device]
                    tokens, soft_labels = apply_nr_aug(tokens, hard_labels, num_classes=int(data["n_classes"]), args=self.args)
                    logits = model(tokens)  # type: ignore[misc]
                    if soft_labels is None:
                        loss = F.cross_entropy(logits, hard_labels)
                    else:
                        loss = soft_cross_entropy(logits, soft_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            elif self.args.model == "graphsage":
                perm = train_t[torch.randperm(train_t.numel())]
                rng = np.random.default_rng(seed * 1000003 + epoch)
                optimizer.zero_grad(set_to_none=True)
                for st in range(0, perm.numel(), self.args.sage_batch_size):
                    batch = perm[st:st + self.args.sage_batch_size]
                    batch_device = batch.to(self.device, non_blocking=(self.device.type == "cuda"))
                    x_outer, blocks_t = self._sample_graphsage_batch(cache, batch, rng)
                    logits = model.forward_sampled(x_outer, blocks_t)  # type: ignore[attr-defined]
                    loss = F.cross_entropy(logits, labels_t[batch_device])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward_logits(model, cache)
                loss = F.cross_entropy(logits[train_t], labels_t[train_t])
                loss.backward()
                optimizer.step()

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)   # CUDA kernels are asynchronous: wait before reading the clock
            epoch_time = time.perf_counter() - t0
            if self.device.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            else:
                peak_mem_mb = 0.0
            epoch_times.append(float(epoch_time))
            epoch_peak_mems.append(float(peak_mem_mb))
            train_val_times.append(float(epoch_time))

            do_eval = (epoch % self.args.eval_every == 0) or (epoch == 1) or (epoch == self.args.epochs)
            if do_eval:
                val_t0 = time.perf_counter()
                val_acc = self.evaluate_subset(model, cache, labels_t, val_t)   # Python float (synchronised)
                train_val_times[-1] += time.perf_counter() - val_t0
                should_stop = stopper.step(val_acc, epoch, model)
                if stopper.best_epoch == epoch:
                    best_epoch_time = epoch_time
                    best_epoch_peak_mem = peak_mem_mb
                if should_stop:
                    break

        if stopper.best_state is None:
            raise RuntimeError("Training failed to record a best checkpoint")

        model.load_state_dict(stopper.best_state)
        test_acc = self.evaluate_subset(model, cache, labels_t, test_t)
        return RunMetrics(
            seed=seed,
            convergence_epoch=stopper.best_epoch,
            best_val_acc=float(stopper.best_metric),
            test_acc=float(test_acc),
            peak_mem_mb_at_convergence_epoch=float(best_epoch_peak_mem),
            epoch_time_sec_at_convergence_epoch=float(best_epoch_time),
            max_peak_mem_mb=float(max(epoch_peak_mems, default=0.0)),
            avg_epoch_time_sec=float(np.mean(epoch_times)) if epoch_times else 0.0,
            precompute_time_sec=float(precompute_time_sec),
            params=num_params,
            time_to_best_sec=float(sum(train_val_times[:max(stopper.best_epoch, 0)])),
        )

    def summarize(self, runs: List[RunMetrics]) -> SummaryMetrics:
        test_accs = np.array([r.test_acc for r in runs], dtype=np.float64)
        convergence_epochs = np.array([r.convergence_epoch for r in runs], dtype=np.float64)
        val_accs = np.array([r.best_val_acc for r in runs], dtype=np.float64)
        mems = np.array([r.max_peak_mem_mb for r in runs], dtype=np.float64)
        times = np.array([r.avg_epoch_time_sec for r in runs], dtype=np.float64)
        precompute_times = np.array([r.precompute_time_sec for r in runs], dtype=np.float64)

        return SummaryMetrics(
            model=self.args.model,
            dataset_root=str(self.root),
            num_runs=len(runs),
            seeds=[r.seed for r in runs],
            test_acc_mean=float(test_accs.mean()),
            test_acc_std=float(test_accs.std(ddof=1) if len(runs) > 1 else 0.0),
            convergence_epoch_mean=float(convergence_epochs.mean()),
            convergence_epoch_std=float(convergence_epochs.std(ddof=1) if len(runs) > 1 else 0.0),
            best_val_acc_mean=float(val_accs.mean()),
            peak_mem_mb_mean=float(mems.mean()),
            epoch_time_sec_mean=float(times.mean()),
            precompute_time_sec_mean=float(precompute_times.mean()),
            runs=[asdict(r) for r in runs],
        )

    def save_summary(self, summary: SummaryMetrics) -> None:
        return

        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.args.model}_{self.root.name}"

        json_path = out_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

        csv_path = out_dir / f"{stem}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "dataset_root", "num_runs", "seeds",
                "test_acc_mean", "test_acc_std", "convergence_epoch_mean", "convergence_epoch_std",
                "best_val_acc_mean", "peak_mem_mb_mean", "epoch_time_sec_mean", "precompute_time_sec_mean",
            ])
            writer.writerow([
                summary.model, summary.dataset_root, summary.num_runs,
                " ".join(map(str, summary.seeds)),
                summary.test_acc_mean, summary.test_acc_std,
                summary.convergence_epoch_mean, summary.convergence_epoch_std,
                summary.best_val_acc_mean, summary.peak_mem_mb_mean,
                summary.epoch_time_sec_mean, summary.precompute_time_sec_mean,
            ])

    def run(self) -> SummaryMetrics:
        seeds = [self.args.seed + i for i in range(self.args.runs)]
        runs = [self.run_once(seed) for seed in seeds]
        summary = self.summarize(runs)
        self.save_summary(summary)
        return summary


# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified node-classification baseline suite")
    ap.add_argument("--data-root", type=str, required=True, help="Directory containing adj.npz feat.npy label.npy")
    ap.add_argument("--dataset", type=str, default=None, help="dataset name (default: basename of --data-root)")
    ap.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "auc", "roc_auc"],
                    help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    ap.add_argument("--model", type=str, required=True, choices=["gcn", "gat", "graphsage", "nagphormer", "nagphormer_plus"])
    ap.add_argument("--device", type=str, default="", help="e.g. cuda:0 or cpu; default: auto")
    ap.add_argument("--out-dir", type=str, default="./results")

    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer used for all models; run_nagphormer.sh passes adamw for official-style NAG runs.")

    ap.add_argument(
        "--nr-aug",
        type=str,
        default="auto",
        choices=["auto", "off", "on"],
        help="Neighborhood Augmentation switch: auto=on for nagphormer_plus, off for nagphormer.",
    )
    ap.add_argument("--nr-aug-paug", type=float, default=0.5, help="Probability of applying NrAug to a NAG mini-batch.")
    ap.add_argument("--nr-aug-tau", type=float, default=0.25, help="LNA mask ratio over hop-token columns.")
    ap.add_argument("--nr-aug-alpha", type=float, default=1.0, help="Beta distribution alpha for GNA mixing.")
    ap.add_argument("--nr-aug-beta", type=float, default=1.0, help="Beta distribution beta for GNA mixing.")
    ap.add_argument("--nr-aug-use-gna", action="store_true", default=True, help="Enable Global Neighborhood Augmentation component.")
    ap.add_argument("--no-nr-aug-use-gna", action="store_false", dest="nr_aug_use_gna", help="Disable GNA and keep only LNA if NrAug is on.")
    ap.add_argument("--nr-aug-use-lna", action="store_true", default=True, help="Enable Local Neighborhood Augmentation component.")
    ap.add_argument("--no-nr-aug-use-lna", action="store_false", dest="nr_aug_use_lna", help="Disable LNA and keep only GNA if NrAug is on.")

    split_group = ap.add_mutually_exclusive_group()
    split_group.add_argument("--resplit-622", action="store_true", dest="resplit_622", help="Deprecated and ignored. This script always reads train.npy/val.npy/test.npy.")
    split_group.add_argument("--keep-existing-split", action="store_false", dest="resplit_622", help="Deprecated. Fixed split files are always used.")
    ap.set_defaults(resplit_622=False)

    det_group = ap.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", action="store_true", dest="deterministic", help="Enable deterministic settings for paper experiments")
    det_group.add_argument("--non-deterministic", action="store_false", dest="deterministic", help="Allow faster but potentially non-deterministic execution")
    ap.set_defaults(deterministic=True)

    ap.add_argument("--hid-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=256)

    ap.add_argument(
        "--sage-fanouts",
        type=str,
        default="25,10",
        help="Comma-separated neighbor fanouts from near-to-target hop outward; length must equal num_layers",
    )
    ap.add_argument("--sage-batch-size", type=int, default=1024)

    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--pe-dim", type=int, default=8)
    ap.add_argument(
        "--pe-dense-fallback-max-n",
        type=int,
        default=10000,
        help="Allow dense eigh fallback for Laplacian PE only when num_nodes <= this threshold",
    )
    ap.add_argument(
        "--nag-cache-on-device",
        action="store_true",
        default=False,
        help="Store NAGphormer token_seq on the target device instead of CPU",
    )
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.nr_aug == "auto":
        args.nr_aug = "on" if args.model == "nagphormer_plus" else "off"
    if args.nr_aug == "on" and not is_nag_model_name(args.model):
        raise ValueError("--nr-aug on is only meaningful for nagphormer/nagphormer_plus")
    args.sage_fanouts_list = parse_fanouts(args.sage_fanouts, args.num_layers) if args.model == "graphsage" else []
    if args.dataset is None:
        args.dataset = Path(args.data_root).name
    args.metric = resolve_metric(args.metric, args.dataset)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    label = metric_label(args.metric)
    print(f"[{args.model}] dataset={args.dataset} metric={label}", flush=True)
    exp = Experiment(args)
    summary = exp.run()  # test_acc_* / best_val_acc_* hold args.metric

    peak_mem = max((float(r["max_peak_mem_mb"]) for r in summary.runs), default=0.0)
    print("\n==== Summary ====")
    print(f"Test {label}: {summary.test_acc_mean * 100:.2f} \u00b1 {summary.test_acc_std * 100:.2f}")
    print(f"Tepoch (s): {summary.epoch_time_sec_mean:.2f}")
    print(f"Mem. (MB): {peak_mem:.2f}")
    # Tepoch of this script is already training-only time per epoch; preprocessing = Hop2Token (+ LapPE)
    pre = [float(r["precompute_time_sec"]) for r in summary.runs]
    print_efficiency(summary.runs[0]["params"], [float(r["avg_epoch_time_sec"]) for r in summary.runs],
                     [float(r["time_to_best_sec"]) for r in summary.runs], peak_mem,
                     pre_s=pre if any(p > 0 for p in pre) else None,
                     best_epoch=[float(r["convergence_epoch"]) for r in summary.runs])
    return

    print("=" * 80)
    print(f"Model: {summary.model}")
    if is_nag_model_name(summary.model):
        print(f"NrAug: {args.nr_aug} | paug={args.nr_aug_paug} tau={args.nr_aug_tau} alpha={args.nr_aug_alpha} beta={args.nr_aug_beta} GNA={args.nr_aug_use_gna} LNA={args.nr_aug_use_lna}")
    print(f"Dataset root: {summary.dataset_root}")
    print(f"Runs: {summary.num_runs} | Seeds: {summary.seeds}")
    print(f"Test Acc @ convergence: {summary.test_acc_mean * 100:.2f} ± {summary.test_acc_std * 100:.2f}")
    print(f"Convergence epoch: {summary.convergence_epoch_mean:.2f} ± {summary.convergence_epoch_std:.2f}")
    print(f"Best val acc (mean): {summary.best_val_acc_mean * 100:.2f}")
    print(f"Peak GPU mem at convergence epoch (MB, mean): {summary.peak_mem_mb_mean:.2f}")
    print(f"Epoch time at convergence epoch (s, mean): {summary.epoch_time_sec_mean:.4f}")
    if is_nag_model_name(summary.model):
        print(f"Hop2Token precompute time (s, mean): {summary.precompute_time_sec_mean:.4f}")
        print(f"All epoch time (s, mean): {summary.precompute_time_sec_mean + summary.epoch_time_sec_mean:.4f}")
    print(f"Results written to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
