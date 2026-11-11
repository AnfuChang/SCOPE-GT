from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class GraphData:
    adj: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    num_nodes: int
    num_features: int
    num_classes: int
    out_dim: int
    problem_type: str  # multiclass / binary / multilabel


def _as_index_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.bool_:
        return np.where(x.reshape(-1))[0].astype(np.int64)
    return x.reshape(-1).astype(np.int64)


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def infer_labels(
    y: np.ndarray,
    *,
    dataset_name: str = "custom",
    metric: str = "auto",
    force_binary: bool = False,
) -> Tuple[np.ndarray, str, int, int]:
    """Return labels, problem_type, num_classes, output_dim.

    PolyFormer official code treats Minesweeper/Tolokers/Questions as binary
    ROC-AUC tasks with one output logit. For ordinary two-class node
    classification we keep two logits unless binary loss is explicitly required.
    """
    y = np.asarray(y)
    name = normalize_dataset_name(dataset_name)
    metric_l = (metric or "auto").lower()
    known_binary_rocauc = {"minesweeper", "tolokers", "questions"}

    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        yy = y.reshape(-1).astype(np.int64)
        valid = yy[yy >= 0]
        num_classes = int(valid.max()) + 1 if valid.size else 0
        use_binary = force_binary or name in known_binary_rocauc or metric_l in {"rocauc", "roc_auc", "auc"}
        if num_classes <= 2 and use_binary:
            return yy.astype(np.float32), "binary", 2, 1
        return yy, "multiclass", num_classes, num_classes

    if y.ndim == 2:
        # Convert strict one-hot labels to multiclass unless ROC-AUC/multilabel is intended.
        row_sum = y.sum(axis=1)
        finite = np.isfinite(row_sum)
        is_binary_matrix = np.all((y[finite] == 0) | (y[finite] == 1)) if finite.any() else False
        is_one_hot = is_binary_matrix and np.all((row_sum[finite] == 0) | (row_sum[finite] == 1))
        if is_one_hot and metric_l not in {"rocauc", "roc_auc", "auc"}:
            return y.argmax(axis=1).astype(np.int64), "multiclass", y.shape[1], y.shape[1]
        return y.astype(np.float32), "multilabel", y.shape[1], y.shape[1]

    raise ValueError(f"Unsupported label shape: {y.shape}")


def row_normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rowsum = np.asarray(x.sum(axis=1)).reshape(-1, 1)
    return x / np.maximum(rowsum, eps)


def standardize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def make_undirected(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def sym_normalize_adjacency(adj: sp.spmatrix, *, add_self_loops: bool) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5).astype(np.float32)
    dmat = sp.diags(deg_inv_sqrt, dtype=np.float32, format="csr")
    return (dmat @ adj @ dmat).tocsr()


def load_bilgt_graph(
    data_dir: str | Path,
    *,
    dataset_name: str = "custom",
    metric: str = "auto",
    force_binary: bool = False,
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

    feat = np.load(paths["feat"], allow_pickle=False).astype(np.float32)
    if sp.issparse(feat):
        feat = feat.toarray().astype(np.float32)
    if feature_norm == "row":
        feat = row_normalize_features(feat).astype(np.float32)
    elif feature_norm == "standard":
        feat = standardize_features(feat).astype(np.float32)
    elif feature_norm != "none":
        raise ValueError(f"Unknown feature_norm={feature_norm!r}")

    labels, problem_type, num_classes, out_dim = infer_labels(
        np.load(paths["label"], allow_pickle=False),
        dataset_name=dataset_name,
        metric=metric,
        force_binary=force_binary,
    )
    train_idx = _as_index_array(np.load(paths["train"], allow_pickle=False))
    val_idx = _as_index_array(np.load(paths["val"], allow_pickle=False))
    test_idx = _as_index_array(np.load(paths["test"], allow_pickle=False))

    n = adj.shape[0]
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj.npz must be square, got {adj.shape}")
    if feat.shape[0] != n or labels.shape[0] != n:
        raise ValueError(f"Shape mismatch: adj={adj.shape}, feat={feat.shape}, labels={labels.shape}")
    for split_name, idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{split_name}.npy contains node id outside [0, {n})")

    return GraphData(
        adj=adj,
        features=feat,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_nodes=n,
        num_features=int(feat.shape[1]),
        num_classes=int(num_classes),
        out_dim=int(out_dim),
        problem_type=problem_type,
    )


def cache_key(dataset: str, base: str, K: int, graph: GraphData) -> str:
    payload = {
        "dataset": normalize_dataset_name(dataset),
        "base": base,
        "K": int(K),
        "n": int(graph.num_nodes),
        "f": int(graph.num_features),
        "nnz": int(graph.adj.nnz),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{payload['dataset']}_{base}_K{K}_{digest}"


def _base_paths(cache_dir: Path, key: str, K: int) -> List[Path]:
    return [cache_dir / f"{key}_base_{i:02d}.npy" for i in range(K + 1)]


def _load_cached(paths: Sequence[Path], cache_mode: str) -> Optional[List[np.ndarray]]:
    if all(p.exists() for p in paths):
        mmap = "r" if cache_mode == "memmap" else None
        return [np.load(p, mmap_mode=mmap, allow_pickle=False) for p in paths]
    return None


def _save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float32), allow_pickle=False)


def build_polynomial_bases(
    graph: GraphData,
    *,
    dataset: str,
    base: str,
    K: int,
    cache_dir: str | Path | None = None,
    cache_mode: str = "ram",
    force_recompute: bool = False,
) -> List[np.ndarray]:
    r"""Build PolyFormer polynomial tokens H_0...H_K.

    mono: H_0=X, H_k=\hat A H_{k-1}, where \hat A uses symmetric GCN
    normalization with self-loops, matching torch_geometric.gcn_norm default.

    cheb: H_0=X, H_1=\tilde L X=-D^{-1/2}AD^{-1/2}X, and
    H_k=2\tilde L H_{k-1}-H_{k-2}, matching the official code's
    get_laplacian(...)+add_self_loops(fill_value=-1.0) construction.
    """
    base = base.lower()
    if base not in {"mono", "cheb"}:
        raise ValueError("base must be 'mono' or 'cheb'")
    if K < 0:
        raise ValueError("K must be non-negative")
    if cache_mode not in {"ram", "memmap"}:
        raise ValueError("cache_mode must be ram or memmap")

    paths: Optional[List[Path]] = None
    if cache_dir:
        cache_dir_p = Path(cache_dir)
        key = cache_key(dataset, base, K, graph)
        paths = _base_paths(cache_dir_p, key, K)
        if not force_recompute:
            cached = _load_cached(paths, cache_mode)
            if cached is not None:
                return cached

    x = np.asarray(graph.features, dtype=np.float32)
    bases: List[np.ndarray] = []

    def append(i: int, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        if paths is not None:
            _save_array(paths[i], arr)
            if cache_mode == "memmap":
                bases.append(np.load(paths[i], mmap_mode="r", allow_pickle=False))
            else:
                bases.append(arr)
        else:
            bases.append(arr)

    if base == "mono":
        adj_norm = sym_normalize_adjacency(graph.adj, add_self_loops=True)
        cur = x
        append(0, cur)
        for i in range(1, K + 1):
            cur = adj_norm @ cur
            append(i, cur)
    else:
        ltilde = -sym_normalize_adjacency(graph.adj, add_self_loops=False)
        t0 = x
        append(0, t0)
        if K >= 1:
            t1 = ltilde @ x
            append(1, t1)
            for i in range(2, K + 1):
                t2 = 2.0 * (ltilde @ t1) - t0
                append(i, t2)
                t0, t1 = t1, t2

    return bases


def bases_to_tensor(
    bases: Sequence[np.ndarray],
    device: torch.device,
    nodes: Optional[np.ndarray] = None,
) -> torch.Tensor:
    if nodes is None:
        arr = np.stack([np.asarray(b) for b in bases], axis=1)
    else:
        idx = np.asarray(nodes, dtype=np.int64)
        arr = np.stack([np.asarray(b[idx]) for b in bases], axis=1)
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device, non_blocking=True)
