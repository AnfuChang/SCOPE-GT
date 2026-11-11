from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class GraphData:
    adj: sp.csr_matrix
    adj_norm: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    num_nodes: int
    num_features: int
    num_classes: int
    problem_type: str  # 'multiclass' or 'multilabel'


def _as_index_array(x: np.ndarray) -> np.ndarray:
    """Accept either a boolean mask or an integer-index array."""
    x = np.asarray(x)
    if x.dtype == np.bool_:
        return np.where(x)[0].astype(np.int64)
    x = x.reshape(-1)
    return x.astype(np.int64)


def _maybe_squeeze_labels(y: np.ndarray) -> Tuple[np.ndarray, str, int]:
    """Infer multiclass vs multilabel labels from a numpy label array."""
    y = np.asarray(y)
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        y = y.reshape(-1).astype(np.int64)
        valid = y[y >= 0]
        num_classes = int(valid.max()) + 1 if valid.size else 0
        return y, "multiclass", num_classes

    # One-hot multiclass labels sometimes appear as [N, C]. Convert them.
    if y.ndim == 2:
        row_sum = y.sum(axis=1)
        finite = np.isfinite(row_sum)
        is_binary = np.all((y[finite] == 0) | (y[finite] == 1)) if finite.any() else False
        is_one_hot = is_binary and np.all((row_sum[finite] == 1) | (row_sum[finite] == 0))
        if is_one_hot and np.nanmax(row_sum) <= 1:
            yy = y.argmax(axis=1).astype(np.int64)
            return yy, "multiclass", y.shape[1]
        return y.astype(np.float32), "multilabel", y.shape[1]

    raise ValueError(f"Unsupported label shape: {y.shape}")


def row_normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rowsum = np.asarray(x.sum(axis=1)).reshape(-1, 1)
    return x / np.maximum(rowsum, eps)


def standardize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def normalize_adjacency(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5).astype(np.float32)
    d_mat = sp.diags(deg_inv_sqrt, dtype=np.float32, format="csr")
    return (d_mat @ adj @ d_mat).tocsr()


def make_undirected(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def scipy_to_torch_sparse(mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def load_bilgt_graph(
    data_dir: str | Path,
    *,
    to_undirected: bool = True,
    feature_norm: str = "none",
) -> GraphData:
    data_dir = Path(data_dir)
    paths = {
        "adj": data_dir / "adj.npz",
        "feat": data_dir / "feat.npy",
        "label": data_dir / "label.npy",
        "train": data_dir / "train.npy",
        "val": data_dir / "val.npy",
        "test": data_dir / "test.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    adj = sp.load_npz(paths["adj"]).tocsr().astype(np.float32)
    if to_undirected:
        adj = make_undirected(adj)

    feat = np.load(paths["feat"]).astype(np.float32)
    if sp.issparse(feat):
        feat = feat.toarray().astype(np.float32)
    if feature_norm == "row":
        feat = row_normalize_features(feat).astype(np.float32)
    elif feature_norm == "standard":
        feat = standardize_features(feat).astype(np.float32)
    elif feature_norm != "none":
        raise ValueError(f"Unknown feature_norm: {feature_norm}")

    labels, problem_type, num_classes = _maybe_squeeze_labels(np.load(paths["label"]))
    train_idx = _as_index_array(np.load(paths["train"]))
    val_idx = _as_index_array(np.load(paths["val"]))
    test_idx = _as_index_array(np.load(paths["test"]))

    if feat.shape[0] != adj.shape[0] or labels.shape[0] != adj.shape[0]:
        raise ValueError(
            f"Shape mismatch: adj={adj.shape}, feat={feat.shape}, labels={labels.shape}"
        )

    adj_norm = normalize_adjacency(adj, add_self_loops=True)
    return GraphData(
        adj=adj,
        adj_norm=adj_norm,
        features=feat,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_nodes=adj.shape[0],
        num_features=feat.shape[1],
        num_classes=num_classes,
        problem_type=problem_type,
    )


def normalized_subgraph(adj: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    """Build a normalized induced subgraph for mini-batch partition training."""
    sub = adj[nodes][:, nodes].tocsr().astype(np.float32)
    return normalize_adjacency(sub, add_self_loops=True)
