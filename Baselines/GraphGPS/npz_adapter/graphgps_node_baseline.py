#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BILGT-compatible GraphGPS baseline for single-graph transductive node classification.

This adapter keeps the core GraphGPS recipe used in the official repository/paper:
interleave a local real-edge MPNN branch with a global attention branch, then mix the
sum by an FFN.  It is intentionally dependency-light so that it can run in the same
project conda environment as the other baselines: only PyTorch, NumPy and SciPy are
required.  torch_geometric / torch_scatter / performer-pytorch are not required.

Supported BILGT-style input formats under --data_dir:
  feat.npy | features.npy | x.npy
  labels.npy | label.npy | y.npy
  adj.npz                      (scipy sparse square matrix)
  or edge_index.npy | edges.npy  (shape [2, E] or [E, 2])
  train.npy | train_idx.npy | train_mask.npy
  val.npy   | val_idx.npy   | valid.npy | valid_idx.npy | val_mask.npy
  test.npy  | test_idx.npy  | test_mask.npy

Also supports a single data.npz / graph.npz containing keys x/feat/features, y/labels,
edge_index or adj, train/val/test.
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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.sparse as sp
except Exception as exc:  # pragma: no cover
    raise ImportError("This BILGT GraphGPS adapter requires scipy for adj.npz loading.") from exc


# -----------------------------
# Reproducibility and utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can make sparse ops very slow or unsupported in
    # some PyTorch builds, so we keep cudnn deterministic only.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        # max_memory_allocated is usually the cleanest measurement for model-side
        # peak memory. It is lower than nvidia-smi because CUDA context memory is
        # not counted; this matches most PyTorch baseline scripts.
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    # ru_maxrss: Linux reports KB, macOS reports bytes. The cluster is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def safe_float(x: float) -> float:
    return float("nan") if x is None else float(x)


# -----------------------------
# Data loading
# -----------------------------

def _candidate_dirs(args: argparse.Namespace) -> List[Path]:
    candidates: List[Path] = []
    if args.data_dir:
        candidates.append(Path(args.data_dir).expanduser())
    if args.data_root and args.dataset:
        candidates.append(Path(args.data_root).expanduser() / args.dataset)
    if args.dataset:
        p = Path(args.dataset).expanduser()
        if p.exists():
            candidates.append(p)
    # Deduplicate while preserving order.
    out: List[Path] = []
    seen = set()
    for p in candidates:
        p = p.resolve() if p.exists() else p
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def _find_file(base: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def _load_numpy_or_sparse(path: Path):
    if path.suffix == ".npz":
        # Try SciPy sparse first. If it fails, fall back to numpy savez format.
        try:
            return sp.load_npz(path)
        except Exception:
            return np.load(path, allow_pickle=True)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported file extension: {path}")


def _npz_get(npz, keys: Iterable[str]):
    for k in keys:
        if k in npz:
            return npz[k]
    return None


def _as_tensor_idx(idx, n: int, name: str) -> torch.Tensor:
    if isinstance(idx, torch.Tensor):
        arr = idx.detach().cpu()
    else:
        arr = torch.as_tensor(np.asarray(idx))
    if arr.dtype == torch.bool:
        if arr.numel() != n:
            raise ValueError(f"{name} boolean mask has {arr.numel()} entries, expected {n}.")
        return arr.nonzero(as_tuple=False).view(-1).long()
    arr = arr.view(-1).long()
    # Support occasional 1-based indices by detecting out-of-range N exactly.
    if arr.numel() > 0 and int(arr.max()) == n and int(arr.min()) >= 1:
        arr = arr - 1
    return arr


def _to_edge_index(edge_obj, n: Optional[int] = None) -> torch.Tensor:
    if isinstance(edge_obj, np.lib.npyio.NpzFile):
        maybe = _npz_get(edge_obj, ["edge_index", "edges", "arr_0", "adj", "A"])
        if maybe is None:
            raise ValueError("Could not find edge_index/edges/adj in npz edge file.")
        edge_obj = maybe
    if sp.issparse(edge_obj):
        coo = edge_obj.tocoo()
        edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
        return torch.from_numpy(edge_index)
    if isinstance(edge_obj, torch.Tensor):
        arr0 = edge_obj.detach().cpu()
    else:
        arr0 = torch.as_tensor(np.asarray(edge_obj))
    if arr0.ndim != 2:
        raise ValueError(f"edge_index/edges/adj must be 2-D, got shape {tuple(arr0.shape)}")
    # Dense square adjacency from a bundled npz.
    if n is not None and arr0.shape[0] == n and arr0.shape[1] == n and arr0.shape[0] != 2 and arr0.shape[1] != 2:
        nz = arr0.nonzero(as_tuple=False)
        return nz.t().contiguous().long()
    arr = arr0.long()
    if arr.shape[0] == 2:
        return arr.contiguous()
    if arr.shape[1] == 2:
        return arr.t().contiguous()
    raise ValueError(f"edge_index/edges must have shape [2,E] or [E,2], got {tuple(arr.shape)}")


def _load_from_npz_bundle(path: Path):
    npz = np.load(path, allow_pickle=True)
    x = _npz_get(npz, ["x", "feat", "feats", "feature", "features"])
    y = _npz_get(npz, ["y", "label", "labels"])
    edge = _npz_get(npz, ["edge_index", "edges"])
    adj = _npz_get(npz, ["adj", "adjacency", "A"])
    train = _npz_get(npz, ["train", "train_idx", "train_index", "train_mask"])
    val = _npz_get(npz, ["val", "valid", "val_idx", "valid_idx", "val_index", "valid_index", "val_mask"])
    test = _npz_get(npz, ["test", "test_idx", "test_index", "test_mask"])
    return x, y, edge, adj, train, val, test


def _split_fallback(y: torch.Tensor, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    all_idx = np.arange(n)
    # Do a simple global split. The official comparison path should use the
    # project-provided split files; fallback is only for debugging.
    rng.shuffle(all_idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = torch.from_numpy(all_idx[:n_train]).long()
    val = torch.from_numpy(all_idx[n_train:n_train + n_val]).long()
    test = torch.from_numpy(all_idx[n_train + n_val:]).long()
    return train, val, test


@dataclass
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    is_multilabel: bool
    num_classes: int
    data_dir: str


def load_bilgt_graph(args: argparse.Namespace) -> GraphData:
    dirs = _candidate_dirs(args)
    if not dirs:
        raise ValueError("Please provide --data_dir or --data_root plus --dataset.")

    last_error: Optional[Exception] = None
    for base in dirs:
        try:
            if not base.exists():
                continue

            x = y = edge = adj = train = val = test = None
            bundle = _find_file(base, ["data.npz", "graph.npz", "dataset.npz"])
            if bundle is not None:
                x, y, edge, adj, train, val, test = _load_from_npz_bundle(bundle)

            if x is None:
                xp = _find_file(base, ["feat.npy", "feats.npy", "features.npy", "feature.npy", "x.npy", "feat.npz", "features.npz", "x.npz"])
                if xp is None:
                    raise FileNotFoundError(f"Cannot find feature file under {base}")
                x_obj = _load_numpy_or_sparse(xp)
                if isinstance(x_obj, np.lib.npyio.NpzFile):
                    x_obj = _npz_get(x_obj, ["x", "feat", "feats", "feature", "features", "arr_0"])
                x = x_obj.toarray() if sp.issparse(x_obj) else x_obj
            if y is None:
                yp = _find_file(base, ["labels.npy", "label.npy", "y.npy", "labels.npz", "label.npz", "y.npz"])
                if yp is None:
                    raise FileNotFoundError(f"Cannot find label file under {base}")
                y_obj = _load_numpy_or_sparse(yp)
                if isinstance(y_obj, np.lib.npyio.NpzFile):
                    y = _npz_get(y_obj, ["y", "label", "labels", "arr_0"])
                else:
                    y = y_obj
            if edge is None and adj is None:
                ep = _find_file(base, ["edge_index.npy", "edges.npy", "edge_index.pt", "edges.pt"])
                ap = _find_file(base, ["adj.npz", "adjacency.npz", "A.npz", "graph_adj.npz"])
                if ep is not None:
                    edge = _load_numpy_or_sparse(ep)
                elif ap is not None:
                    adj = _load_numpy_or_sparse(ap)
                else:
                    raise FileNotFoundError(f"Cannot find adj.npz or edge_index/edges under {base}")

            x_np = x.toarray() if sp.issparse(x) else np.asarray(x)
            if x_np.ndim != 2:
                raise ValueError(f"Feature matrix must be 2-D, got {x_np.shape}")
            x_t = torch.as_tensor(x_np, dtype=torch.float32)
            n = x_t.shape[0]

            y_np = np.asarray(y)
            if y_np.ndim == 2 and y_np.shape[1] == 1:
                y_np = y_np.reshape(-1)
            if y_np.shape[0] != n:
                raise ValueError(f"Labels first dimension {y_np.shape[0]} != num_nodes {n}")
            # Multi-label when labels are a dense 2-D binary/float matrix.
            is_multilabel = False
            if y_np.ndim == 2 and y_np.shape[1] > 1:
                unique_vals = np.unique(y_np[~np.isnan(y_np)] if np.issubdtype(y_np.dtype, np.floating) else y_np)
                is_binary_matrix = unique_vals.size <= 2 and set(unique_vals.tolist()).issubset({0, 1, 0.0, 1.0})
                row_sums = y_np.sum(axis=1)
                looks_onehot = is_binary_matrix and np.all(row_sums == 1)
                # Float probability / binary matrix with multi-positive rows is treated as multi-label.
                is_multilabel = bool(not looks_onehot and (not np.issubdtype(y_np.dtype, np.integer) or is_binary_matrix))
            if is_multilabel:
                y_t = torch.as_tensor(y_np, dtype=torch.float32)
                num_classes = y_t.shape[1]
            else:
                if y_np.ndim == 2 and y_np.shape[1] > 1:
                    # One-hot or class-probability labels.
                    y_np = y_np.argmax(axis=1)
                y_t = torch.as_tensor(y_np, dtype=torch.long).view(-1)
                valid_labels = y_t[y_t >= 0]
                num_classes = int(valid_labels.max().item()) + 1 if valid_labels.numel() else 0

            edge_source = adj if edge is None else edge
            edge_index = _to_edge_index(edge_source, n=n).long()

            if train is None:
                p = _find_file(base, ["train.npy", "train_idx.npy", "train_index.npy", "idx_train.npy", "train_mask.npy"])
                train = _load_numpy_or_sparse(p) if p is not None else None
            if val is None:
                p = _find_file(base, ["val.npy", "valid.npy", "val_idx.npy", "valid_idx.npy", "val_index.npy", "idx_val.npy", "val_mask.npy", "valid_mask.npy"])
                val = _load_numpy_or_sparse(p) if p is not None else None
            if test is None:
                p = _find_file(base, ["test.npy", "test_idx.npy", "test_index.npy", "idx_test.npy", "test_mask.npy"])
                test = _load_numpy_or_sparse(p) if p is not None else None

            if train is None or val is None or test is None:
                if args.allow_random_split:
                    train_idx, val_idx, test_idx = _split_fallback(y_t, args.seed, args.train_ratio, args.val_ratio)
                    print("[WARN] Split files not found. Generated a random split for debugging.", flush=True)
                else:
                    raise FileNotFoundError("Missing train/val/test split files. Set --allow_random_split only for debugging.")
            else:
                train_idx = _as_tensor_idx(train, n, "train")
                val_idx = _as_tensor_idx(val, n, "val")
                test_idx = _as_tensor_idx(test, n, "test")

            # Remove ignored labels if present.
            if not is_multilabel:
                train_idx = train_idx[y_t[train_idx] >= 0]
                val_idx = val_idx[y_t[val_idx] >= 0]
                test_idx = test_idx[y_t[test_idx] >= 0]

            return GraphData(x=x_t, y=y_t, edge_index=edge_index,
                             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                             is_multilabel=is_multilabel, num_classes=num_classes,
                             data_dir=str(base))
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to load BILGT graph from candidates {dirs}. Last error: {last_error}")


# -----------------------------
# Preprocessing: features, PE, adjacency
# -----------------------------

def normalize_features(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "row":
        denom = x.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        return x / denom
    if mode == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-12)
        return (x - mean) / std
    raise ValueError(f"Unknown feature normalization: {mode}")


def add_reverse_and_self_loops(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    parts = [edge_index]
    if undirected:
        parts.append(torch.stack([dst, src], dim=0))
    loop = torch.arange(num_nodes, dtype=torch.long)
    parts.append(torch.stack([loop, loop], dim=0))
    ei = torch.cat(parts, dim=1)
    # Remove duplicates cheaply by using linearized unique indices on CPU.
    lin = ei[0] * num_nodes + ei[1]
    uniq = torch.unique(lin)
    ei = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0)
    return ei.long()


def build_norm_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    # edge_index stores src -> dst; sparse matrix stores [dst, src] for A @ X aggregation.
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), dtype=torch.float32)
    return adj.coalesce().to(device)


def degree_pe(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    deg_in = torch.zeros(num_nodes, dtype=torch.float32)
    deg_out = torch.zeros(num_nodes, dtype=torch.float32)
    deg_in.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg_out.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    pe = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], dim=1)
    pe = (pe - pe.mean(0, keepdim=True)) / pe.std(0, keepdim=True).clamp_min(1e-12)
    return pe


def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int, cache_path: Optional[Path] = None) -> torch.Tensor:
    if k <= 0:
        return torch.empty(num_nodes, 0)
    if cache_path is not None and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (num_nodes, k):
            return torch.from_numpy(arr).float()
    try:
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        data = np.ones(src.shape[0], dtype=np.float32)
        adj = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        deg = np.asarray(adj.sum(1)).reshape(-1)
        inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
        D = sp.diags(inv_sqrt)
        L = sp.eye(num_nodes, dtype=np.float32) - D @ adj @ D
        # k+1 because the first eigenvector is the trivial constant component.
        kk = min(k + 1, max(2, num_nodes - 1))
        vals, vecs = sp.linalg.eigsh(L, k=kk, which="SM", tol=1e-3)
        order = np.argsort(vals)
        vecs = vecs[:, order]
        vecs = vecs[:, 1:k + 1]
        if vecs.shape[1] < k:
            pad = np.zeros((num_nodes, k - vecs.shape[1]), dtype=np.float32)
            vecs = np.concatenate([vecs, pad], axis=1)
        # Deterministic sign convention: largest-magnitude entry positive.
        for j in range(vecs.shape[1]):
            idx = np.argmax(np.abs(vecs[:, j]))
            if vecs[idx, j] < 0:
                vecs[:, j] *= -1
        vecs = vecs.astype(np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vecs)
        return torch.from_numpy(vecs).float()
    except Exception as exc:
        print(f"[WARN] LapPE computation failed ({exc}); falling back to zero PE.", flush=True)
        return torch.zeros(num_nodes, k)


def attach_pe(data: GraphData, args: argparse.Namespace) -> torch.Tensor:
    pes = []
    if args.use_degree_pe:
        pes.append(degree_pe(data.edge_index, data.x.shape[0]))
    if args.lap_pe_k > 0:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path(data.data_dir) / ".graphgps_cache"
        cache_name = f"lap_pe_k{args.lap_pe_k}_{data.x.shape[0]}n.npy"
        pes.append(laplacian_pe(data.edge_index, data.x.shape[0], args.lap_pe_k, cache_dir / cache_name))
    if not pes:
        return data.x
    return torch.cat([data.x] + pes, dim=1)


# -----------------------------
# GraphGPS-style model
# -----------------------------

class SparseGCNBranch(nn.Module):
    """One real-edge local MPNN branch used inside each GPS layer."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(adj, self.lin(x))
        return self.dropout(h)


class DenseMultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single graph as one sequence.
        out = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), need_weights=False)[0]
        return out.squeeze(0)


class LinearGlobalAttention(nn.Module):
    """Dependency-light Performer-style linear global attention.

    It uses positive feature map phi(x)=elu(x)+1. This is not a bitwise copy of
    performer-pytorch FAVOR+, but it implements the same GraphGPS design choice:
    global attention without materializing the full N x N attention matrix.
    """
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        qkv = self.qkv(x).view(n, 3, self.heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [H, N, Dh]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        # Stabilize by sequence length; improves numeric behavior for large N.
        k = k / math.sqrt(max(n, 1))
        v = v / math.sqrt(max(n, 1))
        kv = torch.einsum("hnd,hne->hde", k, v)  # [H, Dh, Dh]
        z = 1.0 / (torch.einsum("hnd,hd->hn", q, k.sum(dim=1)) + 1e-6)
        out = torch.einsum("hnd,hde,hn->hne", q, kv, z)
        out = out.transpose(0, 1).contiguous().view(n, self.dim)
        return self.dropout(self.out_proj(out))


class GPSBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, attn_dropout: float,
                 global_model: str, norm: str, dense_attention_limit: int):
        super().__init__()
        self.local = SparseGCNBranch(dim, dropout)
        self.global_model = global_model.lower()
        if self.global_model == "transformer":
            self.global_attn = DenseMultiheadSelfAttention(dim, heads, attn_dropout)
        elif self.global_model in {"performer", "linear", "linear_attention"}:
            self.global_attn = LinearGlobalAttention(dim, heads, attn_dropout)
        else:
            raise ValueError(f"Unsupported global model: {global_model}")
        self.dense_attention_limit = dense_attention_limit
        self.linear_fallback = LinearGlobalAttention(dim, heads, attn_dropout) if self.global_model == "transformer" else None

        if norm == "batch":
            self.norm_local = nn.BatchNorm1d(dim)
            self.norm_attn = nn.BatchNorm1d(dim)
            self.norm_ff = nn.BatchNorm1d(dim)
        elif norm == "layer":
            self.norm_local = nn.LayerNorm(dim)
            self.norm_attn = nn.LayerNorm(dim)
            self.norm_ff = nn.LayerNorm(dim)
        elif norm == "none":
            self.norm_local = nn.Identity()
            self.norm_attn = nn.Identity()
            self.norm_ff = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Eq. (2) and Eq. (3) in the GPS paper: local MPNN and global attention
        # are applied in parallel on the incoming features, then summed.
        local_out = self.norm_local(x + self.local(x, adj))
        if self.global_model == "transformer" and x.shape[0] > self.dense_attention_limit:
            # Avoid accidental OOM on large transductive graphs. The run log makes
            # this explicit so it is not hidden from the experiment record.
            attn_raw = self.linear_fallback(x)
        else:
            attn_raw = self.global_attn(x)
        attn_out = self.norm_attn(x + attn_raw)
        h = local_out + attn_out
        h = self.norm_ff(h + self.ff(h))
        return h


class GraphGPSNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, heads: int,
                 dropout: float, attn_dropout: float, global_model: str,
                 norm: str, dense_attention_limit: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            GPSBlock(hidden, heads, dropout, attn_dropout, global_model, norm, dense_attention_limit)
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, adj)
        return self.head(h)


# -----------------------------
# Metrics and training
# -----------------------------

@torch.no_grad()
def classification_metric(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                          is_multilabel: bool, metric: str) -> float:
    metric = metric.lower()
    if idx.numel() == 0:
        return 0.0
    logits_i = logits[idx]
    y_i = y[idx]
    if is_multilabel:
        if metric in {"acc", "accuracy", "auto"}:
            pred = (torch.sigmoid(logits_i) > 0.5).to(y_i.dtype)
            return float((pred == y_i).float().mean().item())
        if metric in {"micro_f1", "f1"}:
            pred = (torch.sigmoid(logits_i) > 0.5).float()
            true = y_i.float()
            tp = (pred * true).sum()
            fp = (pred * (1.0 - true)).sum()
            fn = ((1.0 - pred) * true).sum()
            return float((2 * tp / (2 * tp + fp + fn + 1e-12)).item())
        raise ValueError(f"Unsupported multi-label metric: {metric}")
    else:
        if metric in {"acc", "accuracy", "auto"}:
            pred = logits_i.argmax(dim=-1)
            return float((pred == y_i).float().mean().item())
        if metric in {"rocauc", "auc"}:
            try:
                from sklearn.metrics import roc_auc_score
                if logits_i.shape[1] == 2:
                    score = torch.softmax(logits_i, dim=-1)[:, 1].detach().cpu().numpy()
                    return float(roc_auc_score(y_i.detach().cpu().numpy(), score))
                probs = torch.softmax(logits_i, dim=-1).detach().cpu().numpy()
                return float(roc_auc_score(y_i.detach().cpu().numpy(), probs, multi_class="ovr"))
            except Exception as exc:
                raise RuntimeError("ROC-AUC requires scikit-learn and valid class distribution.") from exc
        raise ValueError(f"Unsupported metric: {metric}")


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def resolve_metric(metric: str, dataset: str) -> str:
    metric = metric.lower()
    if metric == "auto":
        return "rocauc" if dataset.lower().replace("-", "_") in AUC_DATASETS else "accuracy"
    return {"acc": "accuracy", "auc": "rocauc", "f1": "micro_f1"}.get(metric, metric)


def metric_label(metric: str) -> str:
    return {"rocauc": "ROC_AUC", "micro_f1": "Micro-F1"}.get(metric, "Acc")


def loss_fn(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, is_multilabel: bool) -> torch.Tensor:
    if is_multilabel:
        return F.binary_cross_entropy_with_logits(logits[idx], y[idx].float())
    return F.cross_entropy(logits[idx], y[idx].long())


@dataclass
class RunResult:
    run: int
    seed: int
    best_epoch: int
    val_score: float
    test_score: float
    avg_epoch_time: float
    mem_mb: float
    params: int = 0                      # trainable parameters
    train_time_per_epoch: float = 0.0    # training only (avg_epoch_time also contains the evaluation)
    time_to_best: float = 0.0            # training + validation time of epochs 1..best_epoch
    pre_time: float = 0.0                # PE (LapPE / degree), feature normalisation, adjacency


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


def train_one_run(base_args: argparse.Namespace, data: GraphData, run: int, seed: int) -> RunResult:
    set_seed(seed)
    device = torch.device(base_args.device if torch.cuda.is_available() or not base_args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pre_t0 = time.perf_counter()   # one-off preprocessing: PE, feature normalisation, adjacency
    x = normalize_features(attach_pe(data, base_args), base_args.feature_norm).to(device)
    y = data.y.to(device)
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    test_idx = data.test_idx.to(device)
    edge_index = add_reverse_and_self_loops(data.edge_index, data.x.shape[0], undirected=base_args.undirected)
    adj = build_norm_adj(edge_index, data.x.shape[0], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pre_time = time.perf_counter() - pre_t0

    model = GraphGPSNodeClassifier(
        in_dim=x.shape[1], out_dim=data.num_classes, hidden=base_args.hidden,
        layers=base_args.layers, heads=base_args.heads, dropout=base_args.dropout,
        attn_dropout=base_args.attn_dropout, global_model=base_args.global_model,
        norm=base_args.norm, dense_attention_limit=base_args.dense_attention_limit,
    ).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if base_args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_args.lr, weight_decay=base_args.weight_decay)
    elif base_args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=base_args.lr, weight_decay=base_args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {base_args.optimizer}")

    if base_args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(base_args.epochs, 1), eta_min=base_args.min_lr)
    else:
        scheduler = None

    best_val = -float("inf")
    best_test = 0.0
    best_epoch = 0
    bad_epochs = 0
    epoch_times: List[float] = []
    train_times: List[float] = []
    train_val_times: List[float] = []   # per epoch: training + validation (no test) -> time to best

    for epoch in range(1, base_args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, adj)
        loss = loss_fn(logits, y, train_idx, data.is_multilabel)
        loss.backward()
        if base_args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_args.clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_time = time.perf_counter() - t0
        train_times.append(train_time)
        val_time = 0.0

        do_eval = (epoch % base_args.eval_period == 0) or (epoch == 1) or (epoch == base_args.epochs)
        if do_eval:
            val_t0 = time.perf_counter()
            model.eval()
            with torch.no_grad():
                logits_eval = model(x, adj)
                # returns a Python float (synchronised)
                val_score = classification_metric(logits_eval, y, val_idx, data.is_multilabel, base_args.metric)
                val_time = time.perf_counter() - val_t0
                test_score = classification_metric(logits_eval, y, test_idx, data.is_multilabel, base_args.metric)
            if val_score > best_val + base_args.min_delta:
                best_val = val_score
                best_test = test_score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += base_args.eval_period
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - t0)
        train_val_times.append(train_time + val_time)

        if base_args.log_every > 0 and (epoch % base_args.log_every == 0 or epoch == 1):
            msg = (f"Run {run} Seed {seed} | Epoch {epoch:04d} | "
                   f"loss {loss.item():.4f} | best_val {best_val * 100:.2f} | "
                   f"best_test {best_test * 100:.2f} | epoch_time {epoch_times[-1]:.3f}s")
            print(msg, flush=True)
        if base_args.patience > 0 and bad_epochs >= base_args.patience:
            break

    mem_mb = get_peak_memory_mb(device)
    # Clean device tensors before next seed.
    del model, optimizer, adj, x, y, train_idx, val_idx, test_idx
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return RunResult(run=run, seed=seed, best_epoch=best_epoch, val_score=best_val,
                     test_score=best_test, avg_epoch_time=float(np.mean(epoch_times)),
                     mem_mb=mem_mb, params=num_params, train_time_per_epoch=float(np.mean(train_times)),
                     time_to_best=float(sum(train_val_times[:max(best_epoch, 0)])), pre_time=float(pre_time))


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GraphGPS baseline adapted to BILGT single-graph node classification.")
    p.add_argument("--dataset", type=str, default="", help="Dataset name or dataset directory.")
    p.add_argument("--data_root", type=str, default="", help="Root containing <dataset>/feat.npy, labels.npy, adj.npz, splits.")
    p.add_argument("--data_dir", type=str, default="", help="Direct dataset directory. Overrides --data_root/--dataset.")
    p.add_argument("--output_dir", type=str, default="results_graphgps_bilgt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_threads", type=int, default=4, help="CPU torch threads; also limits CPU preprocessing overhead.")

    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="First seed; subsequent runs use seed+i.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--eval_period", type=int, default=5)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--global_model", type=str, default="performer", choices=["transformer", "performer", "linear", "linear_attention"])
    p.add_argument("--dense_attention_limit", type=int, default=12000,
                   help="When global_model=transformer and N exceeds this, use linear fallback to avoid OOM.")
    p.add_argument("--norm", type=str, default="batch", choices=["batch", "layer", "none"])

    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--min_delta", type=float, default=0.0)

    p.add_argument("--feature_norm", type=str, default="row", choices=["row", "standard", "none"])
    p.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_degree_pe", action=argparse.BooleanOptionalAction, default=True,
                   help="Add log-degree local structural encoding as node features.")
    p.add_argument("--lap_pe_k", type=int, default=0, help="Optional Laplacian PE eigenvector count. Use small values only.")
    p.add_argument("--cache_dir", type=str, default="")

    p.add_argument("--metric", type=str, default="auto", choices=["auto", "accuracy", "acc", "rocauc", "auc", "micro_f1", "f1"],
                   help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    p.add_argument("--allow_random_split", action="store_true")
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    return p


def write_results(output_dir: Path, args: argparse.Namespace, data: GraphData, results: List[RunResult]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"graphgps_{args.dataset or Path(data.data_dir).name}_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "seed", "best_epoch", "Val Acc(%)", "Test Acc(%)", "Tepoch(s)", "Mem.(MB)"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run": r.run,
                "seed": r.seed,
                "best_epoch": r.best_epoch,
                "Val Acc(%)": f"{r.val_score * 100:.4f}",
                "Test Acc(%)": f"{r.test_score * 100:.4f}",
                "Tepoch(s)": f"{r.avg_epoch_time:.4f}",
                "Mem.(MB)": f"{r.mem_mb:.2f}",
            })
        vals = np.array([r.val_score * 100 for r in results], dtype=np.float64)
        tests = np.array([r.test_score * 100 for r in results], dtype=np.float64)
        times = np.array([r.avg_epoch_time for r in results], dtype=np.float64)
        mems = np.array([r.mem_mb for r in results], dtype=np.float64)
        writer.writerow({
            "run": "mean", "seed": "", "best_epoch": "",
            "Val Acc(%)": f"{vals.mean():.4f}",
            "Test Acc(%)": f"{tests.mean():.4f}",
            "Tepoch(s)": f"{times.mean():.4f}",
            "Mem.(MB)": f"{mems.mean():.2f}",
        })
        writer.writerow({
            "run": "std", "seed": "", "best_epoch": "",
            "Val Acc(%)": f"{vals.std(ddof=1) if len(vals) > 1 else 0:.4f}",
            "Test Acc(%)": f"{tests.std(ddof=1) if len(tests) > 1 else 0:.4f}",
            "Tepoch(s)": f"{times.std(ddof=1) if len(times) > 1 else 0:.4f}",
            "Mem.(MB)": f"{mems.std(ddof=1) if len(mems) > 1 else 0:.2f}",
        })
    # Save full configuration for reproducibility.
    with (output_dir / f"graphgps_{args.dataset or Path(data.data_dir).name}_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    data = load_bilgt_graph(args)
    ds_name = Path(args.dataset).name if args.dataset else Path(data.data_dir).name
    if args.metric.lower() == "auto" and data.is_multilabel:
        args.metric = "accuracy"
    args.metric = resolve_metric(args.metric, ds_name)
    label = metric_label(args.metric)
    print("=" * 88)
    print("GraphGPS-BILGT baseline")
    print(f"data_dir: {data.data_dir} | metric: {label}")
    print(f"nodes: {data.x.shape[0]} | feat_dim: {data.x.shape[1]} | edges(raw): {data.edge_index.shape[1]} | classes: {data.num_classes}")
    print(f"splits: train={len(data.train_idx)}, val={len(data.val_idx)}, test={len(data.test_idx)}")
    print(f"model: GPS(local=GCN, global={args.global_model}, layers={args.layers}, hidden={args.hidden}, heads={args.heads})")
    print("=" * 88, flush=True)

    results: List[RunResult] = []
    for r in range(args.runs):
        seed = args.seed + r
        result = train_one_run(args, data, run=r + 1, seed=seed)
        results.append(result)
        print(f"RUN\t{r + 1}\tseed={seed}\tbest_epoch={result.best_epoch}\t"
              f"Test {label}(%)\t{result.test_score * 100:.4f}\t"
              f"Tepoch(s)\t{result.avg_epoch_time:.4f}\t"
              f"Mem.(MB)\t{result.mem_mb:.2f}\t"
              f"TimeToBest(s)\t{result.time_to_best:.2f}", flush=True)

    tests = np.array([r.test_score * 100 for r in results], dtype=np.float64)
    times = np.array([r.avg_epoch_time for r in results], dtype=np.float64)
    mems = np.array([r.mem_mb for r in results], dtype=np.float64)
    print("\n==== Summary ====")
    print(f"Test {label}: {tests.mean():.2f} \u00b1 {(tests.std(ddof=1) if len(tests) > 1 else 0.0):.2f}")
    print(f"Tepoch (s): {times.mean():.2f}")
    print(f"Mem. (MB): {mems.max():.2f}")
    # Tepoch above includes the per-epoch evaluation; the efficiency block uses training-only time
    print_efficiency(results[0].params, [r.train_time_per_epoch for r in results],
                     [r.time_to_best for r in results], float(mems.max()),
                     pre_s=[r.pre_time for r in results], best_epoch=[r.best_epoch for r in results])
    return 0

    print("\nFinal averaged result over %d runs (seeds %d-%d):" % (args.runs, args.seed, args.seed + args.runs - 1))
    print("Test Acc(%)\tTepoch(s)\tMem.(MB)")
    print(f"{tests.mean():.4f}±{(tests.std(ddof=1) if len(tests) > 1 else 0):.4f}\t"
          f"{times.mean():.4f}\t{mems.mean():.2f}")

    output_dir = Path(args.output_dir)
    csv_path = write_results(output_dir, args, data, results)
    print(f"Saved CSV summary to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
