"""BILGT-style data loader for VCR-Graphormer baselines.

Expected dataset directory format, matching the common BILGT baseline format:
  - features: feat.npy / features.npy / x.npy / feat.npz / features.npz
  - labels:   labels.npy / label.npy / y.npy
  - graph:    adj.npz OR edge_index.npy OR edges.npy
  - splits:   train.npy, val.npy, test.npy (indices or boolean masks)

The loader deliberately does not create new train/val/test splits.  This keeps
baseline evaluation aligned with the user's own algorithm.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class BilgtGraphData:
    name: str
    data_dir: Path
    adj: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        y = self.labels
        if y.ndim == 2:
            return int(y.shape[1])
        valid = y[y >= 0]
        return int(valid.max() + 1) if valid.size else 0


def _first_existing(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def _load_dense_or_sparse(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        try:
            mx = sp.load_npz(path)
            return mx.toarray().astype(np.float32, copy=False)
        except Exception:
            obj = np.load(path, allow_pickle=True)
            if isinstance(obj, np.lib.npyio.NpzFile):
                # common np.savez keys
                for key in ["features", "feat", "x", "arr_0"]:
                    if key in obj:
                        arr = obj[key]
                        return np.asarray(arr, dtype=np.float32)
                raise KeyError(f"No known feature key found in {path}; keys={list(obj.keys())}")
            return np.asarray(obj, dtype=np.float32)
    return np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)


def _load_labels(path: Path) -> np.ndarray:
    labels = np.load(path, allow_pickle=True)
    labels = np.asarray(labels)
    if labels.ndim == 2:
        if labels.shape[1] == 1:
            labels = labels.reshape(-1)
        else:
            labels = labels.argmax(axis=1)
    if labels.dtype.kind not in "iu":
        labels = labels.astype(np.int64)
    return labels.astype(np.int64, copy=False)


def _indices_from_file(path: Path, n: int) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        arr = np.where(arr.reshape(-1))[0]
    else:
        arr = arr.reshape(-1)
    arr = arr.astype(np.int64, copy=False)
    if arr.size == 0:
        raise ValueError(f"Split file {path} is empty.")
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(f"Split indices in {path} are out of range for n={n}: min={arr.min()}, max={arr.max()}")
    return arr


def _load_adj(root: Path, n: int) -> sp.csr_matrix:
    adj_path = _first_existing(root, ["adj.npz", "adjacency.npz", "graph.npz", "a.npz"])
    if adj_path is not None:
        adj = sp.load_npz(adj_path).tocsr()
    else:
        edge_path = _first_existing(root, ["edge_index.npy", "edges.npy", "edge.npy", "edge_index.npz"])
        if edge_path is None:
            raise FileNotFoundError(
                f"Could not find adj.npz or edge_index.npy/edges.npy under {root}."
            )
        if edge_path.suffix == ".npz":
            obj = np.load(edge_path, allow_pickle=True)
            for key in ["edge_index", "edges", "arr_0"]:
                if key in obj:
                    edge_index = obj[key]
                    break
            else:
                raise KeyError(f"No edge_index-like key found in {edge_path}; keys={list(obj.keys())}")
        else:
            edge_index = np.load(edge_path, allow_pickle=True)
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2:
            raise ValueError(f"edge_index/edges must be 2D, got shape {edge_index.shape}")
        if edge_index.shape[0] == 2:
            row, col = edge_index[0], edge_index[1]
        elif edge_index.shape[1] == 2:
            row, col = edge_index[:, 0], edge_index[:, 1]
        else:
            raise ValueError(f"edge_index/edges must have shape [2,E] or [E,2], got {edge_index.shape}")
        data = np.ones(len(row), dtype=np.float32)
        adj = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    if adj.shape != (n, n):
        raise ValueError(f"Adjacency shape {adj.shape} does not match feature node count {(n, n)}.")
    adj.data = np.ones_like(adj.data, dtype=np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def make_undirected(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj.data = np.ones_like(adj.data, dtype=np.float32)
    adj.eliminate_zeros()
    return adj.tocsr()


def load_bilgt_dataset(data_dir: str | os.PathLike, dataset_name: Optional[str] = None, to_undirected: bool = True) -> BilgtGraphData:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    feat_path = _first_existing(root, ["feat.npy", "features.npy", "x.npy", "node_feat.npy", "feat.npz", "features.npz"])
    label_path = _first_existing(root, ["labels.npy", "label.npy", "y.npy", "node_label.npy"])
    train_path = _first_existing(root, ["train.npy", "train_idx.npy", "idx_train.npy", "train_mask.npy"])
    val_path = _first_existing(root, ["val.npy", "valid.npy", "val_idx.npy", "idx_val.npy", "valid_idx.npy", "val_mask.npy"])
    test_path = _first_existing(root, ["test.npy", "test_idx.npy", "idx_test.npy", "test_mask.npy"])

    missing = [
        ("features", feat_path), ("labels", label_path), ("train split", train_path),
        ("val split", val_path), ("test split", test_path)
    ]
    missing_names = [name for name, path in missing if path is None]
    if missing_names:
        raise FileNotFoundError(f"Missing required files under {root}: {', '.join(missing_names)}")

    features = _load_dense_or_sparse(feat_path)
    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got {features.shape} from {feat_path}")
    n = int(features.shape[0])
    labels = _load_labels(label_path)
    if labels.shape[0] != n:
        raise ValueError(f"Labels length {labels.shape[0]} does not match number of nodes {n}.")
    adj = _load_adj(root, n)
    if to_undirected:
        adj = make_undirected(adj)

    train_idx = _indices_from_file(train_path, n)
    val_idx = _indices_from_file(val_path, n)
    test_idx = _indices_from_file(test_path, n)

    # Check overlap, but do not silently modify official splits.
    sets = [set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        warnings.warn("Train/val/test splits overlap. Keeping them unchanged because baseline must use provided splits.")

    return BilgtGraphData(
        name=dataset_name or root.name,
        data_dir=root,
        adj=adj,
        features=features.astype(np.float32, copy=False),
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
