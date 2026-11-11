"""Graph loading.

Expected directory layout:
    adj.npz  feat.npy  label.npy  train.npy  val.npy  test.npy
(metadata.json is read when official split provenance is recorded.)
Sampling uses the undirected projection; graph.py reads adj.npz again for edge direction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

OFFICIAL_MASK_DATASETS = ('minesweeper', 'roman_empire', 'amazon_ratings', 'tolokers')


def resolve_split_suffix(dataset, suffix=None, split_index=None):
    """Select a fixed mask independently of a training seed."""
    if split_index is not None:
        if suffix is not None:
            raise ValueError('Use split-index or split-suffix, not both')
        if split_index not in range(10):
            raise ValueError('Official split-index must be in 0..9')
        return f'_split{split_index}'
    if suffix is not None:
        return suffix
    return '_split0' if str(dataset).lower().replace('-', '_') in OFFICIAL_MASK_DATASETS else ''


def resolve_feature_normalize(dataset, mode="auto"):
    """Feature-type starting points, not validation-selected optimal settings.

    Dense embeddings/statistical or mixed attributes keep their input scale.
    Sparse word/count features retain historical L1 normalization. Unknown
    datasets fall back to L1; explicit l1/none always overrides this policy.
    """
    if mode not in ("auto", "l1", "none"):
        raise ValueError("feature-normalize must be auto, l1 or none")
    if mode != "auto":
        return mode
    name = str(dataset).lower().replace("-", "_")
    preserve = {"roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions",
                "wikics", "ogbn_arxiv", "ogbn_products", "pokec", "arxiv_year"}
    return "none" if name in preserve else "l1"


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
    split_metadata: dict | None = None


def load_graph(root, suffix=None, normalize="l1"):
    if normalize not in ("l1", "none"):
        raise ValueError("resolve feature normalization to l1/none before loading")
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
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.sort_indices()
    if normalize == "l1":
        features /= np.maximum(np.abs(features).sum(axis=1, keepdims=True), 1e-12)
    classes = np.unique(labels[labels >= 0])
    if not len(classes) or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Labels must be contiguous 0..C-1, with -1 allowed for unlabeled")
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    dataset = metadata.get('dataset', root.name)
    suffix = resolve_split_suffix(dataset, suffix)
    official = metadata.get('split_policy') in ('platonov_official_10_fixed_50_25_25',
                                                'ogb_official_sales_ranking')
    splits = {}
    for name in ("train", "val", "test"):
        filename = name + suffix + '.npy'
        if official:
            expected = metadata.get('split_sha256', {}).get(filename)
            if not expected or sha256_file(root / filename) != expected:
                raise ValueError('Official split fingerprint mismatch: ' + filename)
        ids = np.load(root / filename, allow_pickle=False)
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
    split_metadata = None
    if official:
        if len(joined) != n:
            raise ValueError('Official split does not cover all nodes')
        names.append('metadata.json')
        split_metadata = {'policy': metadata['split_policy'], 'suffix': suffix,
                          'official_split_id': (int(suffix.removeprefix('_split')) if suffix.startswith('_split')
                                                else metadata.get('default_official_split_id')),
                          'source_url': metadata.get('source_url')}
    return Graph(adj, features, labels, splits, {s: sha256_file(root / s) for s in names}, split_metadata)
