#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BILGT-compatible Exphormer baseline.

This file keeps the core Exphormer idea from the official implementation:
  sparse attention over local graph edges + random regular expander edges
  (+ optional virtual/global nodes), used together with a GCN branch in a
  GraphGPS-style layer.

It deliberately avoids torch_geometric/torch_scatter so that it can run in the
same project conda environment used by BILGT-style baselines as long as PyTorch,
NumPy and SciPy are available.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
try:
    import resource
except ImportError:  # Windows: model APIs remain usable by the portable measurement runner.
    resource = None
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility and utility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def current_memory_mb(device: torch.device) -> float:
    """Return peak GPU memory when CUDA is used, else process RSS in MB."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def canonical_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    if index.numel() == 0:
        return float("nan")
    pred = logits[index].argmax(dim=-1)
    true = labels[index]
    return float((pred == true).float().mean().item())


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def rocauc_from_logits(logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    if index.numel() == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    prob = torch.softmax(logits[index].detach().float(), dim=-1).cpu().numpy()
    true = labels[index].detach().cpu().numpy()
    if prob.shape[1] == 2:
        return float(roc_auc_score(true, prob[:, 1]))  # score of the positive class
    return float(roc_auc_score(true, prob, multi_class="ovr"))


def micro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    """Small no-sklearn fallback for multi-label data."""
    pred = (torch.sigmoid(logits[index]) > 0.5).to(torch.int64)
    true = labels[index].to(torch.int64)
    tp = (pred & true).sum().float()
    fp = (pred & (1 - true)).sum().float()
    fn = ((1 - pred) & true).sum().float()
    denom = 2 * tp + fp + fn
    if denom.item() == 0:
        return 0.0
    return float((2 * tp / denom).item())


# ---------------------------------------------------------------------------
# BILGT-style data loader
# ---------------------------------------------------------------------------


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _load_array(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        # Try scipy sparse first, then ordinary numpy npz.
        try:
            return sp.load_npz(path)
        except Exception:
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported file format: {path}")


def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _dense_features(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.tocsr().astype(np.float32)
        # Most citation/product features are sparse but still moderate enough for
        # the baselines used in this project. This script is full-batch, so dense
        # node features are expected by the model.
        return x.toarray().astype(np.float32)
    if isinstance(x, dict):
        for key in ["x", "feat", "features", "node_feat", "node_features"]:
            if key in x:
                return _dense_features(x[key])
        raise KeyError(f"Feature npz/dict has no known key. Keys={list(x.keys())}")
    x = _as_numpy(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D node features, got shape {x.shape}")
    return x.astype(np.float32, copy=False)


def _labels_from_raw(y) -> Tuple[np.ndarray, bool]:
    if isinstance(y, dict):
        for key in ["y", "label", "labels", "node_label", "node_labels"]:
            if key in y:
                return _labels_from_raw(y[key])
        raise KeyError(f"Label npz/dict has no known key. Keys={list(y.keys())}")
    y = _as_numpy(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim == 2:
        # Treat exact one-hot labels as ordinary multi-class labels.
        row_sums = y.sum(axis=1)
        if np.all((y == 0) | (y == 1)) and np.allclose(row_sums, 1):
            return y.argmax(axis=1).astype(np.int64), False
        return y.astype(np.float32), True
    return y.astype(np.int64, copy=False), False


def _split_to_index(arr, n: int) -> np.ndarray:
    arr = _as_numpy(arr)
    if arr.dtype == np.bool_:
        if arr.shape[0] != n:
            raise ValueError(f"Boolean split mask length {arr.shape[0]} != num_nodes {n}")
        return np.where(arr)[0].astype(np.int64)
    arr = arr.reshape(-1)
    return arr.astype(np.int64, copy=False)


def _load_splits(data_dir: Path, n: int) -> Dict[str, np.ndarray]:
    split_file = _first_existing(data_dir, [
        "splits.npz", "split.npz", "split_idx.npz", "indices.npz",
        "masks.npz", "mask.npz",
    ])
    if split_file is not None:
        raw = _load_array(split_file)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected split file {split_file} to contain named arrays")
        key_map = {
            "train": ["train", "train_idx", "idx_train", "train_mask"],
            "val": ["val", "valid", "valid_idx", "val_idx", "idx_val", "idx_valid", "valid_mask", "val_mask"],
            "test": ["test", "test_idx", "idx_test", "test_mask"],
        }
        out = {}
        for split, keys in key_map.items():
            key = next((k for k in keys if k in raw), None)
            if key is not None:
                out[split] = _split_to_index(raw[key], n)
        if {"train", "val", "test"}.issubset(out.keys()):
            return out

    candidates = {
        "train": ["train.npy", "train_idx.npy", "idx_train.npy", "train_mask.npy"],
        "val": ["val.npy", "valid.npy", "val_idx.npy", "valid_idx.npy", "idx_val.npy", "idx_valid.npy", "val_mask.npy", "valid_mask.npy"],
        "test": ["test.npy", "test_idx.npy", "idx_test.npy", "test_mask.npy"],
    }
    splits = {}
    for split, names in candidates.items():
        path = _first_existing(data_dir, names)
        if path is None:
            raise FileNotFoundError(
                f"Cannot find {split} split in {data_dir}. Expected one of {names} "
                "or a split/splits .npz file."
            )
        splits[split] = _split_to_index(_load_array(path), n)
    return splits


def _edge_index_from_adj(raw_adj, n: int) -> np.ndarray:
    if sp.issparse(raw_adj):
        coo = raw_adj.tocoo()
        edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
        return edge_index
    if isinstance(raw_adj, dict):
        # Common cases: scipy saved manually, npz with edge_index, or arrays row/col.
        for key in ["edge_index", "edges", "edge", "adj"]:
            if key in raw_adj:
                val = raw_adj[key]
                if key == "adj" and sp.issparse(val):
                    return _edge_index_from_adj(val, n)
                arr = _as_numpy(val)
                if arr.ndim == 2 and arr.shape[0] == 2:
                    return arr.astype(np.int64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr.T.astype(np.int64)
        if {"row", "col"}.issubset(raw_adj.keys()):
            return np.vstack([raw_adj["row"], raw_adj["col"]]).astype(np.int64)
        if {"src", "dst"}.issubset(raw_adj.keys()):
            return np.vstack([raw_adj["src"], raw_adj["dst"]]).astype(np.int64)
        raise KeyError(f"Adjacency npz/dict has no known edge keys. Keys={list(raw_adj.keys())}")
    arr = _as_numpy(raw_adj)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.astype(np.int64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.T.astype(np.int64)
    if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == n:
        coo = sp.coo_matrix(arr)
        return np.vstack([coo.row, coo.col]).astype(np.int64)
    raise ValueError(f"Cannot interpret adjacency/edge_index with shape {arr.shape}")


def row_normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.maximum(rowsum, eps)
    return x / rowsum


@dataclass
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    num_classes: int
    is_multilabel: bool


def load_bilgt_graph(data_dir: Path, feature_norm: str = "none") -> GraphData:
    feature_path = _first_existing(data_dir, [
        "x.npy", "feat.npy", "features.npy", "node_feat.npy", "node_features.npy",
        "x.npz", "feat.npz", "features.npz", "node_feat.npz", "x.pt", "features.pt",
    ])
    label_path = _first_existing(data_dir, [
        "y.npy", "label.npy", "labels.npy", "node_label.npy", "node_labels.npy",
        "y.npz", "labels.npz", "y.pt", "labels.pt",
    ])
    adj_path = _first_existing(data_dir, [
        "adj.npz", "adjacency.npz", "graph.npz", "edge_index.npy", "edges.npy",
        "edge_index.npz", "edge_index.pt", "edges.pt",
    ])
    if feature_path is None:
        raise FileNotFoundError(f"Cannot find node feature file in {data_dir}")
    if label_path is None:
        raise FileNotFoundError(f"Cannot find label file in {data_dir}")
    if adj_path is None:
        raise FileNotFoundError(f"Cannot find adjacency/edge_index file in {data_dir}")

    x = _dense_features(_load_array(feature_path))
    if feature_norm == "row":
        x = row_normalize_features(x).astype(np.float32)
    elif feature_norm == "standard":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x = ((x - mean) / np.maximum(std, 1e-12)).astype(np.float32)
    elif feature_norm != "none":
        raise ValueError(f"Unknown feature_norm={feature_norm}")

    y_np, is_multilabel = _labels_from_raw(_load_array(label_path))
    n = x.shape[0]
    if y_np.shape[0] != n:
        raise ValueError(f"Features have {n} nodes, labels have {y_np.shape[0]}")
    edge_index_np = _edge_index_from_adj(_load_array(adj_path), n)
    if edge_index_np.min(initial=0) < 0 or edge_index_np.max(initial=0) >= n:
        raise ValueError("edge_index contains node ids outside [0, num_nodes)")
    splits = _load_splits(data_dir, n)

    if is_multilabel:
        num_classes = int(y_np.shape[1])
        y = torch.tensor(y_np, dtype=torch.float32)
    else:
        num_classes = int(np.max(y_np)) + 1
        y = torch.tensor(y_np, dtype=torch.long)
    return GraphData(
        x=torch.tensor(x, dtype=torch.float32),
        y=y,
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        train_idx=torch.tensor(splits["train"], dtype=torch.long),
        val_idx=torch.tensor(splits["val"], dtype=torch.long),
        test_idx=torch.tensor(splits["test"], dtype=torch.long),
        num_classes=num_classes,
        is_multilabel=is_multilabel,
    )


# ---------------------------------------------------------------------------
# Edge construction: local graph + random regular expander + optional globals
# ---------------------------------------------------------------------------


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    return edge_index[:, edge_index[0] != edge_index[1]]


def add_reverse_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def coalesce_edges(edge_index: torch.Tensor, edge_type: Optional[torch.Tensor], num_nodes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Coalesce duplicate directed edges. Keep the minimum edge type id."""
    if edge_index.numel() == 0:
        return edge_index, edge_type
    key = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(key)
    edge_index = edge_index[:, order]
    key = key[order]
    if edge_type is not None:
        edge_type = edge_type[order]
    keep = torch.ones(key.numel(), dtype=torch.bool)
    keep[1:] = key[1:] != key[:-1]
    # If there are duplicate edges with different types, the first after sorting is
    # arbitrary. Local edges are concatenated before expander edges, and coalescing
    # mainly reduces exact duplicates; edge type is only a light structural cue.
    return edge_index[:, keep], None if edge_type is None else edge_type[keep]


def generate_expander_edges(num_nodes: int, actual_degree: int, seed: int, algorithm: str = "random-d") -> torch.Tensor:
    """Generate directed, symmetric expander edges.

    The Exphormer paper/configs use d/2 random permutations and then add both
    directions, so `actual_degree=6` means three random permutations.
    """
    if actual_degree <= 0 or num_nodes <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    if actual_degree % 2 != 0:
        raise ValueError("actual expander degree must be even; use 6 by default")
    half = min(actual_degree // 2, max(1, num_nodes - 1))
    rng = np.random.default_rng(seed)
    senders: List[int] = []
    receivers: List[int] = []
    nodes = np.arange(num_nodes, dtype=np.int64)

    if num_nodes <= 10:
        src, dst = np.where(np.ones((num_nodes, num_nodes), dtype=bool))
        mask = src != dst
        return torch.tensor(np.vstack([src[mask], dst[mask]]), dtype=torch.long)

    if algorithm == "random-d":
        for _ in range(half):
            perm = rng.permutation(nodes)
            senders.extend(nodes.tolist())
            receivers.extend(perm.tolist())
            senders.extend(perm.tolist())
            receivers.extend(nodes.tolist())
    elif algorithm in {"random-d2", "random-d-2"}:
        stubs = np.repeat(nodes, half)
        perm = rng.permutation(stubs)
        senders.extend(stubs.tolist())
        receivers.extend(perm.tolist())
        senders.extend(perm.tolist())
        receivers.extend(stubs.tolist())
    elif algorithm == "hamiltonian":
        for _ in range(half):
            perm = rng.permutation(nodes)
            prev = np.roll(perm, 1)
            senders.extend(perm.tolist())
            receivers.extend(prev.tolist())
            senders.extend(prev.tolist())
            receivers.extend(perm.tolist())
    else:
        raise ValueError("expander_algorithm must be random-d, random-d2, or hamiltonian")

    edge_index = torch.tensor(np.vstack([senders, receivers]), dtype=torch.long)
    return remove_self_loops(edge_index)


def build_attention_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    expander_degree: int = 6,
    use_local_edges: bool = True,
    use_expander_edges: bool = True,
    add_loops: bool = True,
    undirected: bool = True,
    coalesce: bool = True,
    expander_algorithm: str = "random-d",
) -> Tuple[torch.Tensor, torch.Tensor]:
    parts: List[torch.Tensor] = []
    types: List[torch.Tensor] = []
    if use_local_edges:
        local = remove_self_loops(edge_index.cpu())
        if undirected:
            local = add_reverse_edges(local)
        parts.append(local)
        types.append(torch.zeros(local.size(1), dtype=torch.long))  # 0: input/local
    if use_expander_edges:
        exp = generate_expander_edges(num_nodes, expander_degree, seed, expander_algorithm)
        parts.append(exp)
        types.append(torch.ones(exp.size(1), dtype=torch.long))  # 1: expander
    if add_loops:
        loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
        parts.append(loops)
        types.append(torch.full((num_nodes,), 2, dtype=torch.long))  # 2: self-loop
    if len(parts) == 0:
        raise ValueError("At least one of local edges, expander edges, or self loops must be used")
    out_edge = torch.cat(parts, dim=1)
    out_type = torch.cat(types, dim=0)
    if coalesce:
        out_edge, out_type = coalesce_edges(out_edge, out_type, num_nodes)
    return out_edge.contiguous(), out_type.contiguous()


def gcn_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
    return (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).float()


def build_virtual_edges(num_nodes: int, num_virtual_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional edges between each real node and each virtual node."""
    if num_virtual_nodes <= 0:
        return (torch.empty(2, 0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    real = torch.arange(num_nodes, dtype=torch.long, device=device)
    edge_parts = []
    type_parts = []
    for i in range(num_virtual_nodes):
        vid = num_nodes + i
        v = torch.full((num_nodes,), vid, dtype=torch.long, device=device)
        edge_parts.append(torch.stack([real, v], dim=0))
        edge_parts.append(torch.stack([v, real], dim=0))
        type_parts.append(torch.full((num_nodes,), 3, dtype=torch.long, device=device))
        type_parts.append(torch.full((num_nodes,), 3, dtype=torch.long, device=device))
    return torch.cat(edge_parts, dim=1), torch.cat(type_parts, dim=0)


# ---------------------------------------------------------------------------
# Pure PyTorch Exphormer / GraphGPS-style model
# ---------------------------------------------------------------------------


class SparseExphormerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, num_edge_types: int = 4, use_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.Q = nn.Linear(dim, dim, bias=use_bias)
        self.K = nn.Linear(dim, dim, bias=use_bias)
        self.V = nn.Linear(dim, dim, bias=use_bias)
        self.O = nn.Linear(dim, dim, bias=use_bias)
        self.edge_emb = nn.Embedding(num_edge_types, dim)
        nn.init.ones_(self.edge_emb.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        q = self.Q(x).view(n, self.num_heads, self.head_dim)
        k = self.K(x).view(n, self.num_heads, self.head_dim)
        v = self.V(x).view(n, self.num_heads, self.head_dim)
        e = self.edge_emb(edge_type).view(-1, self.num_heads, self.head_dim)

        # Official Exphormer uses exp(clamp(sum((K*Q*E)/sqrt(d)))) and sums by destination.
        scores = ((k[src] * q[dst]) * e).sum(dim=-1, keepdim=True) / math.sqrt(self.head_dim)
        scores = torch.exp(scores.clamp(-5.0, 5.0))
        if self.training:
            scores = self.dropout(scores)
        msg = v[src] * scores
        out = torch.zeros_like(v)
        z = torch.zeros(n, self.num_heads, 1, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        z.index_add_(0, dst, scores)
        out = out / (z + 1e-6)
        out = out.reshape(n, self.dim)
        return self.O(out)


class SparseGCNConv(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        xw = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        msg = xw[src] * edge_weight.view(-1, 1)
        out = torch.zeros_like(xw)
        out.index_add_(0, dst, msg)
        return out


class GPSExphormerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        ffn_dim: int,
        norm: str = "batch",
        use_gcn: bool = True,
        num_virtual_nodes: int = 0,
    ):
        super().__init__()
        self.use_gcn = use_gcn
        self.num_virtual_nodes = num_virtual_nodes
        self.attn = SparseExphormerAttention(dim, heads, dropout=attn_dropout)
        self.gcn = SparseGCNConv(dim) if use_gcn else None
        self.virtual_node_emb = nn.Parameter(torch.zeros(num_virtual_nodes, dim)) if num_virtual_nodes > 0 else None
        if self.virtual_node_emb is not None:
            nn.init.normal_(self.virtual_node_emb, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        def make_norm():
            if norm == "batch":
                return nn.BatchNorm1d(dim)
            if norm == "layer":
                return nn.LayerNorm(dim)
            if norm == "none":
                return nn.Identity()
            raise ValueError("norm must be batch, layer, or none")

        self.norm_local = make_norm()
        self.norm_attn = make_norm()
        self.norm_ffn = make_norm()

    def forward(
        self,
        x: torch.Tensor,
        attn_edge_index: torch.Tensor,
        attn_edge_type: torch.Tensor,
        gcn_edge_index: torch.Tensor,
        gcn_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h_parts = []
        if self.num_virtual_nodes > 0:
            virt = self.virtual_node_emb
            x_ext = torch.cat([x, virt], dim=0)
            virt_edge_index, virt_edge_type = build_virtual_edges(x.size(0), self.num_virtual_nodes, x.device)
            full_edge_index = torch.cat([attn_edge_index, virt_edge_index], dim=1)
            full_edge_type = torch.cat([attn_edge_type, virt_edge_type], dim=0)
            h_attn_ext = self.attn(x_ext, full_edge_index, full_edge_type)
            h_attn_raw = h_attn_ext[:x.size(0)]
        else:
            h_attn_raw = self.attn(x, attn_edge_index, attn_edge_type)
        h_attn = x + self.dropout(h_attn_raw)
        h_attn = self.norm_attn(h_attn)
        h_parts.append(h_attn)
        if self.use_gcn:
            h_local = x + self.dropout(self.gcn(x, gcn_edge_index, gcn_edge_weight))
            h_local = self.norm_local(h_local)
            h_parts.append(h_local)
        h = sum(h_parts)
        h = h + self.ffn(h)
        h = self.norm_ffn(h)
        return h


class BILGTExphormer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        ffn_dim: int,
        norm: str,
        use_gcn: bool,
        num_virtual_nodes: int = 0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            GPSExphormerLayer(
                hidden_dim, num_heads, dropout, attn_dropout,
                ffn_dim=ffn_dim, norm=norm, use_gcn=use_gcn,
                num_virtual_nodes=num_virtual_nodes,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_edge_index: torch.Tensor,
        attn_edge_type: torch.Tensor,
        gcn_edge_index: torch.Tensor,
        gcn_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, attn_edge_index, attn_edge_type, gcn_edge_index, gcn_edge_weight)
        return self.head(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    best_val_pct: float
    test_acc_pct: float
    tepoch_s: float
    mem_mb: float
    stopped_epoch: int
    params: int = 0              # trainable parameters
    time_to_best_s: float = 0.0  # training + validation time of epochs 1..best_epoch
    pre_time_s: float = 0.0      # expander / GCN edge construction


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


def make_scheduler(optimizer: torch.optim.Optimizer, max_epoch: int, warmup_epochs: int, scheduler: str):
    if scheduler == "none":
        return None
    if scheduler != "cosine":
        raise ValueError("scheduler must be cosine or none")

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, max_epoch - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_run(args: argparse.Namespace, data: GraphData, run_seed: int, run_no: int) -> RunResult:
    set_seed(run_seed)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    x = data.x.to(device)
    y = data.y.to(device)
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    test_idx = data.test_idx.to(device)

    pre_t0 = time.perf_counter()   # one-off preprocessing: expander + GCN edges
    attn_edge_index, attn_edge_type = build_attention_edges(
        data.edge_index,
        num_nodes=data.x.size(0),
        seed=run_seed,
        expander_degree=args.expander_degree,
        use_local_edges=not args.no_local_edges,
        use_expander_edges=not args.no_expander_edges,
        add_loops=args.add_self_loops,
        undirected=not args.keep_directed,
        coalesce=not args.no_coalesce_edges,
        expander_algorithm=args.expander_algorithm,
    )
    # GCN branch always uses real graph + reverse + self loops.
    gcn_edge_index = data.edge_index.cpu()
    if not args.keep_directed:
        gcn_edge_index = add_reverse_edges(remove_self_loops(gcn_edge_index))
    gcn_edge_index = add_self_loops(remove_self_loops(gcn_edge_index), data.x.size(0))
    gcn_edge_index, _ = coalesce_edges(gcn_edge_index, None, data.x.size(0))
    gcn_edge_weight = gcn_norm(gcn_edge_index, data.x.size(0))

    attn_edge_index = attn_edge_index.to(device)
    attn_edge_type = attn_edge_type.to(device)
    gcn_edge_index = gcn_edge_index.to(device)
    gcn_edge_weight = gcn_edge_weight.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pre_time_s = time.perf_counter() - pre_t0

    model = BILGTExphormer(
        in_dim=data.x.size(1),
        out_dim=data.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        ffn_dim=args.ffn_dim,
        norm=args.norm,
        use_gcn=not args.no_gcn_branch,
        num_virtual_nodes=args.num_virtual_nodes,
    ).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("optimizer must be AdamW or Adam")
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.scheduler)

    if data.is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
        metric_fn = micro_f1_from_logits
        metric_name = "micro-F1"
    elif args.metric == "rocauc":
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = rocauc_from_logits
        metric_name = "ROC_AUC"
    else:
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = accuracy_from_logits
        metric_name = "accuracy"

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    patience_ctr = 0
    epoch_times: List[float] = []
    train_val_times: List[float] = []   # per epoch: training + validation (no test) -> time to best

    print(
        f"[Run {run_no}] seed={run_seed} nodes={data.x.size(0)} edges(attn)={attn_edge_index.size(1)} "
        f"features={data.x.size(1)} classes={data.num_classes} metric={metric_name}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, attn_edge_index, attn_edge_type, gcn_edge_index, gcn_edge_weight)
        if data.is_multilabel:
            loss = loss_fn(logits[train_idx], y[train_idx].float())
        else:
            loss = loss_fn(logits[train_idx], y[train_idx])
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)   # CUDA kernels are asynchronous: wait before reading the clock
        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)

        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        val_time = 0.0
        if do_eval:
            val_t0 = time.perf_counter()
            model.eval()
            with torch.no_grad():
                logits = model(x, attn_edge_index, attn_edge_type, gcn_edge_index, gcn_edge_weight)
                val_score = metric_fn(logits, y, val_idx)   # returns a Python float (synchronised)
                val_time = time.perf_counter() - val_t0
                test_score = metric_fn(logits, y, test_idx)
            if val_score > best_val:
                best_val = val_score
                best_test = test_score
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
        else:
            val_score = best_val
            test_score = best_test
        train_val_times.append(epoch_time + val_time)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs or do_eval:
            print(
                f"[Run {run_no}] Epoch {epoch:04d} loss={loss.item():.4f} "
                f"val={val_score * 100:.2f} test={test_score * 100:.2f} "
                f"best_test={best_test * 100:.2f}@{best_epoch} "
                f"time={epoch_time:.3f}s",
                flush=True,
            )
        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"[Run {run_no}] Early stop at epoch {epoch}; best_epoch={best_epoch}", flush=True)
            break

    mem_mb = current_memory_mb(device)
    tepoch_s = float(np.mean(epoch_times)) if epoch_times else float("nan")
    result = RunResult(
        seed=run_seed,
        best_epoch=int(best_epoch),
        best_val_pct=float(best_val * 100.0),
        test_acc_pct=float(best_test * 100.0),
        tepoch_s=tepoch_s,
        mem_mb=mem_mb,
        stopped_epoch=len(epoch_times),
        params=num_params,
        time_to_best_s=float(sum(train_val_times[:max(best_epoch, 0)])),
        pre_time_s=pre_time_s,
    )
    label = metric_label(args.metric)
    print(
        f"[Run {run_no}] RESULT seed={run_seed} best_epoch={result.best_epoch} "
        f"Best Val {label}(%)={result.best_val_pct:.2f} Test {label}(%)={result.test_acc_pct:.2f} "
        f"Tepoch(s)={result.tepoch_s:.4f} Mem.(MB)={result.mem_mb:.2f} TimeToBest(s)={result.time_to_best_s:.2f}",
        flush=True,
    )

    del model, optimizer, scheduler, x, y, train_idx, val_idx, test_idx
    del attn_edge_index, attn_edge_type, gcn_edge_index, gcn_edge_weight
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_label(metric: str) -> str:
    return "ROC_AUC" if metric == "rocauc" else "Acc"


def summarize(results: Sequence[RunResult], dataset: str, out_dir: Path, metric: str = "acc") -> None:
    # test_acc_pct holds the selected metric (accuracy, or ROC-AUC on the AUC datasets)
    arr_acc = np.array([r.test_acc_pct for r in results], dtype=float)
    arr_time = np.array([r.tepoch_s for r in results], dtype=float)
    arr_mem = np.array([r.mem_mb for r in results], dtype=float)
    std_ddof = 1 if len(results) > 1 else 0
    print("\n==== Summary ====")
    print(f"Test {metric_label(metric)}: {arr_acc.mean():.2f} \u00b1 {arr_acc.std(ddof=std_ddof):.2f}")
    print(f"Tepoch (s): {arr_time.mean():.2f}")
    print(f"Mem. (MB): {arr_mem.max():.2f}")
    # Tepoch of this script is already training-only time per epoch
    print_efficiency(results[0].params, [r.tepoch_s for r in results], [r.time_to_best_s for r in results],
                     float(arr_mem.max()), pre_s=[r.pre_time_s for r in results],
                     best_epoch=[r.best_epoch for r in results])
    return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    arr_acc = np.array([r.test_acc_pct for r in results], dtype=float)
    arr_time = np.array([r.tepoch_s for r in results], dtype=float)
    arr_mem = np.array([r.mem_mb for r in results], dtype=float)
    arr_epoch = np.array([r.best_epoch for r in results], dtype=float)
    std_ddof = 1 if len(results) > 1 else 0
    print("\n========== Exphormer BILGT Summary ==========")
    print(f"Dataset: {dataset}")
    print(f"Runs: {len(results)}")
    print(f"Test Acc(%): {arr_acc.mean():.2f} ± {arr_acc.std(ddof=std_ddof):.2f}")
    print(f"Tepoch(s): {arr_time.mean():.4f} ± {arr_time.std(ddof=std_ddof):.4f}")
    print(f"Mem.(MB): {arr_mem.mean():.2f} ± {arr_mem.std(ddof=std_ddof):.2f}")
    print(f"Best Epoch: {arr_epoch.mean():.1f} ± {arr_epoch.std(ddof=std_ddof):.1f}")
    print(f"CSV: {csv_path}")
    print(
        f"FINAL_RESULT dataset={dataset} runs={len(results)} "
        f"Test Acc(%)={arr_acc.mean():.2f}±{arr_acc.std(ddof=std_ddof):.2f} "
        f"Tepoch(s)={arr_time.mean():.4f}±{arr_time.std(ddof=std_ddof):.4f} "
        f"Mem.(MB)={arr_mem.mean():.2f}±{arr_mem.std(ddof=std_ddof):.2f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BILGT-compatible Exphormer baseline")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., flickr, pubmed, github, ogbn_arxiv")
    p.add_argument("--data_root", type=str, default="/mnt/sdb/changaf25/sdust/datasets", help="Root folder containing dataset folders")
    p.add_argument("--data_dir", type=str, default=None, help="Explicit dataset folder. Overrides data_root/dataset")
    p.add_argument("--out_dir", type=str, default="results_bilgt_exphormer", help="Output directory")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=5)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--attn_dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "adamw", "adam"])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--norm", type=str, default="batch", choices=["batch", "layer", "none"])

    p.add_argument("--expander_degree", type=int, default=6, help="Actual directed/symmetric expander degree; paper's large datasets use 6")
    p.add_argument("--num_virtual_nodes", type=int, default=0, help="Optional Exphormer global/virtual nodes; large transductive paper settings use 0")
    p.add_argument("--expander_algorithm", type=str, default="random-d", choices=["random-d", "random-d2", "random-d-2", "hamiltonian"])
    p.add_argument("--no_expander_edges", action="store_true")
    p.add_argument("--no_local_edges", action="store_true")
    p.add_argument("--no_gcn_branch", action="store_true")
    p.add_argument("--add_self_loops", action="store_true", default=True)
    p.add_argument("--no_add_self_loops", action="store_false", dest="add_self_loops")
    p.add_argument("--keep_directed", action="store_true", help="Do not add reverse edges")
    p.add_argument("--no_coalesce_edges", action="store_true")
    p.add_argument("--feature_norm", type=str, default="none", choices=["none", "row", "standard"])
    p.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "accuracy", "rocauc", "auc", "roc_auc"],
                   help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_threads", type=int, default=4, help="CPU torch threads; ignored by CUDA kernels but avoids CPU thread overhead")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.num_threads, 4)))
        except RuntimeError:
            pass
    ds_name = canonical_dataset_name(args.dataset)
    metric = args.metric.lower()
    if metric == "auto":
        metric = "rocauc" if ds_name in AUC_DATASETS else "acc"
    args.metric = "rocauc" if metric in ("auc", "roc_auc", "rocauc") else "acc"
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    if not data_dir.exists():
        alt = Path(args.data_root) / ds_name
        if alt.exists():
            data_dir = alt
        else:
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    out_dir = Path(args.out_dir) / ds_name

    print(f"[Data] Loading {args.dataset} from {data_dir}", flush=True)
    data = load_bilgt_graph(data_dir, feature_norm=args.feature_norm)
    print(
        f"[Data] x={tuple(data.x.shape)} y={tuple(data.y.shape)} edge_index={tuple(data.edge_index.shape)} "
        f"train/val/test={data.train_idx.numel()}/{data.val_idx.numel()}/{data.test_idx.numel()} "
        f"classes={data.num_classes} multilabel={data.is_multilabel}",
        flush=True,
    )

    results = []
    for i in range(args.runs):
        run_seed = args.seed + i
        results.append(train_one_run(args, data, run_seed, i + 1))
    summarize(results, args.dataset, out_dir, args.metric)


if __name__ == "__main__":
    main()
