"""Graph loading.

Vendored unchanged from asymmetric_sampling_experiments/unified_control/data.py so that this
folder runs on its own. Keeping the loader byte-for-byte identical is deliberate: it is what makes
ACE-GT results directly comparable with the unified_control A / center_query / tokenwise runs
(same symmetrisation, same self-loop removal, same L1 row normalisation, same split validation).

Expected directory layout, exactly as produced for the unified_control experiments:
    adj.npz  feat.npy  label.npy  train.npy  val.npy  test.npy
(label_mapping.json and metadata.json may be present; they are not read.)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Graph:
    adj: sparse.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    splits: dict
    fingerprint: dict


def load_graph(root, suffix="", normalize="l1", directed=False):
    root = Path(root)
    adj = sparse.load_npz(root / "adj.npz").tocsr()
    features = np.load(root / "feat.npy", allow_pickle=False).astype(np.float32)
    raw_labels = np.load(root / "label.npy", allow_pickle=False)
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("Labels must be integers; no implicit float-to-int cast")
    labels = raw_labels.astype(np.int64).reshape(-1)
    n = len(labels)
    if adj.shape != (n, n) or features.ndim != 2 or features.shape[0] != n:
        raise ValueError("Graph/features/labels shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(adj.data).all():
        raise ValueError("Nonfinite graph or feature values")
    if not directed:
        adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.sort_indices()
    if normalize == "l1":
        features /= np.maximum(np.abs(features).sum(axis=1, keepdims=True), 1e-12)
    classes = np.unique(labels[labels >= 0])
    if not len(classes) or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Labels must be contiguous 0..C-1, with -1 allowed for unlabeled")
    splits = {}
    for name in ("train", "val", "test"):
        ids = np.load(root / (name + suffix + ".npy"), allow_pickle=False)
        if ids.dtype == np.bool_:
            if ids.shape != (n,):
                raise ValueError("Boolean mask shape mismatch")
            ids = np.flatnonzero(ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Splits must contain integer IDs or boolean masks")
        ids = ids.astype(np.int64).reshape(-1)
        if not len(ids) or ids.min() < 0 or ids.max() >= n or np.any(labels[ids] < 0):
            raise ValueError("Empty, out-of-range, or unlabeled split: " + name)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate split IDs")
        splits[name] = ids
    joined = np.concatenate(list(splits.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Splits overlap")
    names = ["adj.npz", "feat.npy", "label.npy"] + [s + suffix + ".npy" for s in splits]
    return Graph(adj, features, labels, splits, {s: sha256_file(root / s) for s in names})
