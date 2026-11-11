"""Offline VCR-Graphormer tokenization for BILGT-style datasets.

This module reimplements the VCR token-list construction without requiring DGL
or torch-geometric at runtime.  It follows Eq. (3.3) of the paper:
  target token + L local diffusion tokens + structure-aware PPR tokens +
  content-aware PPR tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

try:
    import numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback for minimal envs
    numba = None
    _HAS_NUMBA = False


@dataclass(frozen=True)
class TokenizerConfig:
    hops: int = 3
    num_structure_supernodes: int = 20
    num_structure_tokens: int = 5
    num_content_tokens: int = 5
    ppr_alpha: float = 0.85
    structure_ppr_eps: float = 1e-7
    content_ppr_eps: float = 1e-2
    structure_ppr_max_itr: int = 10000
    content_ppr_max_itr: int = 100
    partition_method: str = "auto"  # auto/metis/bfs/hash
    content_label_source: str = "train"  # train/trainval/non_test/all
    dtype: str = "float32"
    token_seed: int = 42


def _fingerprint_graph(adj: sp.csr_matrix, features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, cfg: TokenizerConfig) -> str:
    h = hashlib.sha1()
    h.update(str(adj.shape).encode())
    h.update(str(adj.nnz).encode())
    h.update(str(features.shape).encode())
    h.update(str(labels.shape).encode())
    # lightweight but stable enough for cache invalidation
    for arr in [adj.indptr, adj.indices, features.reshape(-1)[: min(features.size, 4096)], labels[: min(labels.size, 4096)], train_idx, val_idx, test_idx]:
        arr = np.asarray(arr)
        h.update(arr.shape.__repr__().encode())
        h.update(str(arr.dtype).encode())
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    h.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    inv[mask] = 1.0 / deg[mask]
    return sp.diags(inv).dot(adj).tocsr()


def _sym_normalize_with_self_loop(adj: sp.csr_matrix) -> sp.csr_matrix:
    n = adj.shape[0]
    a = (adj + sp.eye(n, dtype=np.float32, format="csr")).tocsr().astype(np.float32)
    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    d = sp.diags(inv_sqrt)
    return d.dot(a).dot(d).tocsr()


def _try_dgl_metis(adj: sp.csr_matrix, num_parts: int) -> Optional[List[np.ndarray]]:
    try:
        import dgl  # type: ignore
        g = dgl.from_scipy(adj)
        g = dgl.to_bidirected(g)
        parts = dgl.metis_partition(g=g, k=num_parts, extra_cached_hops=0, reshuffle=False,
                                    balance_ntypes=None, balance_edges=False, mode="k-way")
        clusters: List[np.ndarray] = []
        for i in range(num_parts):
            clusters.append(parts[i].ndata["_ID"].cpu().numpy().astype(np.int64))
        return clusters
    except Exception as exc:
        warnings.warn(f"DGL/METIS partitioning unavailable; falling back to deterministic BFS/hash partition. Reason: {exc}")
        return None


def _hash_partition(n: int, num_parts: int) -> List[np.ndarray]:
    return [np.arange(i, n, num_parts, dtype=np.int64) for i in range(num_parts)]


def _bfs_partition(adj: sp.csr_matrix, num_parts: int, seed: int = 42) -> List[np.ndarray]:
    """A dependency-free balanced BFS fallback.

    It is not meant to replace METIS for strict reproduction, but it keeps the
    baseline executable in environments where DGL/METIS is absent.
    """
    n = adj.shape[0]
    if num_parts <= 1:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.default_rng(seed)
    deg = np.diff(adj.indptr)
    target = int(math.ceil(n / num_parts))
    order = np.argsort(-deg)
    assigned = np.full(n, -1, dtype=np.int32)
    clusters: List[List[int]] = [[] for _ in range(num_parts)]

    seed_ptr = 0
    for part in range(num_parts):
        while seed_ptr < n and assigned[order[seed_ptr]] >= 0:
            seed_ptr += 1
        if seed_ptr >= n:
            break
        src = int(order[seed_ptr])
        q = [src]
        assigned[src] = part
        while q and len(clusters[part]) < target:
            u = q.pop(0)
            clusters[part].append(u)
            neigh = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
            if neigh.size > 0:
                # high-degree first tends to produce more compact components
                neigh = neigh[np.argsort(-deg[neigh])]
            for v in neigh:
                v = int(v)
                if assigned[v] < 0:
                    assigned[v] = part
                    q.append(v)
                if len(clusters[part]) + len(q) >= target:
                    break

    # Fill leftovers into currently smallest clusters.
    leftovers = np.where(assigned < 0)[0]
    if leftovers.size:
        rng.shuffle(leftovers)
    for u in leftovers.tolist():
        part = int(np.argmin([len(c) for c in clusters]))
        assigned[u] = part
        clusters[part].append(int(u))

    return [np.asarray(c, dtype=np.int64) for c in clusters]


def partition_graph(adj: sp.csr_matrix, num_parts: int, method: str = "auto", seed: int = 42) -> List[np.ndarray]:
    n = adj.shape[0]
    num_parts = max(1, min(int(num_parts), n))
    method = (method or "auto").lower()
    if method in {"auto", "metis"}:
        clusters = _try_dgl_metis(adj, num_parts)
        if clusters is not None:
            return clusters
        if method == "metis":
            raise RuntimeError("partition_method=metis requested, but DGL/METIS partitioning failed.")
    if method == "hash":
        return _hash_partition(n, num_parts)
    return _bfs_partition(adj, num_parts, seed=seed)


def add_structure_supernodes(adj: sp.csr_matrix, clusters: Sequence[np.ndarray]) -> sp.csr_matrix:
    n = adj.shape[0]
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for i, members in enumerate(clusters):
        members = np.asarray(members, dtype=np.int64)
        if members.size == 0:
            continue
        s = n + i
        rows.append(members)
        cols.append(np.full(members.size, s, dtype=np.int64))
        rows.append(np.full(members.size, s, dtype=np.int64))
        cols.append(members)
    if not rows:
        return adj.copy().tocsr()
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    sup = sp.coo_matrix((np.ones(row.size, dtype=np.float32), (row, col)), shape=(n + len(clusters), n + len(clusters))).tocsr()
    base = sp.block_diag((adj, sp.csr_matrix((len(clusters), len(clusters)), dtype=np.float32)), format="csr")
    out = (base + sup).tocsr()
    out.data = np.ones_like(out.data, dtype=np.float32)
    out.eliminate_zeros()
    return out


def add_content_supernodes(adj: sp.csr_matrix, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, num_classes: int, source: str = "train") -> sp.csr_matrix:
    n = adj.shape[0]
    source = source.lower()
    if source == "train":
        allowed = train_idx
    elif source == "trainval":
        allowed = np.concatenate([train_idx, val_idx])
    elif source == "non_test":
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        allowed = np.where(mask)[0]
    elif source == "all":
        allowed = np.arange(n)
    else:
        raise ValueError(f"Unknown content_label_source={source}")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    y = labels.reshape(-1)
    for c in range(num_classes):
        members = allowed[y[allowed] == c].astype(np.int64, copy=False)
        if members.size == 0:
            continue
        s = n + c
        rows.append(members)
        cols.append(np.full(members.size, s, dtype=np.int64))
        rows.append(np.full(members.size, s, dtype=np.int64))
        cols.append(members)
    if not rows:
        warnings.warn("No content-aware virtual connections were created; check labels/splits.")
        return sp.block_diag((adj, sp.csr_matrix((num_classes, num_classes), dtype=np.float32)), format="csr")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    sup = sp.coo_matrix((np.ones(row.size, dtype=np.float32), (row, col)), shape=(n + num_classes, n + num_classes)).tocsr()
    base = sp.block_diag((adj, sp.csr_matrix((num_classes, num_classes), dtype=np.float32)), format="csr")
    out = (base + sup).tocsr()
    out.data = np.ones_like(out.data, dtype=np.float32)
    out.eliminate_zeros()
    return out


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _ppr_single_numba(src, indptr, indices, deg, alpha, eps, max_itr):
        # Numba typed dictionaries/lists are used to mirror the push operation
        # from the original implementation.
        alpha_eps = alpha * eps
        fzero = np.float32(0.0)
        p = {src: fzero}
        r = {src: np.float32(alpha)}
        q = [src]
        it = 0
        while len(q) > 0 and it < max_itr:
            u = q.pop()
            res = r[u] if u in r else fzero
            if u in p:
                p[u] += res
            else:
                p[u] = res
            r[u] = fzero
            du = deg[u]
            if du > 0:
                val = np.float32((1.0 - alpha) * res / du)
                for ptr in range(indptr[u], indptr[u + 1]):
                    v = indices[ptr]
                    if v in r:
                        r[v] += val
                    else:
                        r[v] = val
                    rv = r[v] if v in r else fzero
                    if rv >= alpha_eps * deg[v]:
                        in_q = False
                        for old in q:
                            if old == v:
                                in_q = True
                                break
                        if not in_q:
                            q.append(v)
            it += 1
        keys = np.empty(len(p), dtype=np.int64)
        vals = np.empty(len(p), dtype=np.float32)
        ii = 0
        for k in p:
            keys[ii] = k
            vals[ii] = p[k]
            ii += 1
        return keys, vals
else:
    _ppr_single_numba = None


def _ppr_single_py(src: int, indptr: np.ndarray, indices: np.ndarray, deg: np.ndarray, alpha: float, eps: float, max_itr: int) -> Tuple[np.ndarray, np.ndarray]:
    alpha_eps = alpha * eps
    p: Dict[int, float] = {int(src): 0.0}
    r: Dict[int, float] = {int(src): float(alpha)}
    q: List[int] = [int(src)]
    it = 0
    while q and it < max_itr:
        u = q.pop()
        res = r.get(u, 0.0)
        p[u] = p.get(u, 0.0) + res
        r[u] = 0.0
        du = int(deg[u])
        if du > 0:
            val = (1.0 - alpha) * res / du
            for v in indices[indptr[u]:indptr[u+1]]:
                v = int(v)
                r[v] = r.get(v, 0.0) + val
                if r[v] >= alpha_eps * deg[v] and v not in q:
                    q.append(v)
        it += 1
    keys = np.fromiter(p.keys(), dtype=np.int64)
    vals = np.fromiter(p.values(), dtype=np.float32)
    return keys, vals


def ppr_topk_for_nodes(adj: sp.csr_matrix, output_nodes: np.ndarray, topk: int, alpha: float, eps: float, max_itr: int, rng: np.random.Generator, desc: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    adj = adj.tocsr()
    # PPR push only needs unweighted outgoing neighbor lists.
    indptr = adj.indptr.astype(np.int64, copy=False)
    indices = adj.indices.astype(np.int64, copy=False)
    deg = np.diff(indptr).astype(np.float32)
    n = adj.shape[0]
    top_indices: List[np.ndarray] = []
    top_weights: List[np.ndarray] = []
    use_numba = _HAS_NUMBA and _ppr_single_numba is not None

    for src in tqdm(output_nodes.astype(np.int64), desc=desc, total=len(output_nodes)):
        if use_numba:
            keys, vals = _ppr_single_numba(int(src), indptr, indices, deg, float(alpha), float(eps), int(max_itr))
        else:
            keys, vals = _ppr_single_py(int(src), indptr, indices, deg, float(alpha), float(eps), int(max_itr))
        if keys.size == 0:
            keys = np.asarray([src], dtype=np.int64)
            vals = np.asarray([1.0], dtype=np.float32)
        order = np.argsort(vals)[::-1]
        keys = keys[order]
        vals = vals[order]
        if topk > 0:
            keys = keys[:topk]
            vals = vals[:topk]
        # Original code pads with random nodes when PPR returns fewer entries.
        if keys.size < topk:
            pad = rng.integers(0, n, size=topk - keys.size, dtype=np.int64)
            keys = np.concatenate([keys, pad])
            vals = np.concatenate([vals, np.zeros(pad.size, dtype=np.float32)])
        top_indices.append(keys.astype(np.int64, copy=False))
        top_weights.append(vals.astype(np.float32, copy=False))
    return top_indices, top_weights


def build_or_load_tokens(
    *,
    dataset_name: str,
    adj: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TokenizerConfig,
    cache_dir: str | os.PathLike,
    force: bool = False,
) -> Tuple[Path, Tuple[int, int, int], Dict[str, object]]:
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint_graph(adj, features, labels, train_idx, val_idx, test_idx, cfg)
    token_path = cache_root / f"{dataset_name}_vcr_tokens_{fp}.npy"
    meta_path = cache_root / f"{dataset_name}_vcr_tokens_{fp}.json"
    num_nodes, feat_dim = features.shape
    token_len = 1 + cfg.hops + cfg.num_structure_tokens + cfg.num_content_tokens
    shape = (num_nodes, token_len, feat_dim + 1)

    if token_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        return token_path, tuple(meta["shape"]), meta

    rng = np.random.default_rng(cfg.token_seed)
    dtype = np.float32 if cfg.dtype == "float32" else np.float64
    print(f"[Tokenize] Building VCR token list: shape={shape}, dtype={np.dtype(dtype).name}")
    tokens = np.lib.format.open_memmap(token_path, mode="w+", dtype=dtype, shape=shape)
    tokens[:] = 0

    x0 = features.astype(dtype, copy=False)
    # Component 1: target node feature || 1.
    tokens[:, 0, :feat_dim] = x0
    tokens[:, 0, feat_dim] = 1.0

    # Component 2: local topology through L sym-normalized propagation steps.
    print("[Tokenize] Local diffusion tokens (symmetrically normalized A+I) ...")
    p_sym = _sym_normalize_with_self_loop(adj)
    x = x0
    denom = float(sum(range(1, cfg.hops + 1))) if cfg.hops > 0 else 1.0
    for l in range(1, cfg.hops + 1):
        x = p_sym.dot(x).astype(dtype, copy=False)
        slot = l
        tokens[:, slot, :feat_dim] = x
        tokens[:, slot, feat_dim] = float(cfg.hops - l + 1) / denom

    # Component 3: structure-aware virtual connections + approximate PPR.
    if cfg.num_structure_tokens > 0:
        print(f"[Tokenize] Structure-aware partition: method={cfg.partition_method}, supernodes={cfg.num_structure_supernodes}")
        clusters = partition_graph(adj, cfg.num_structure_supernodes, method=cfg.partition_method, seed=cfg.token_seed)
        adj_struct = add_structure_supernodes(adj, clusters)
        ppr_idx, ppr_w = ppr_topk_for_nodes(
            adj_struct, np.arange(num_nodes, dtype=np.int64), cfg.num_structure_tokens,
            cfg.ppr_alpha, cfg.structure_ppr_eps, cfg.structure_ppr_max_itr, rng,
            desc="[Tokenize] Structure PPR",
        )
        # Features of virtual supernodes are zero.
        padded_features = np.zeros((adj_struct.shape[0], feat_dim), dtype=dtype)
        padded_features[:num_nodes] = x0
        start = 1 + cfg.hops
        for i in tqdm(range(num_nodes), desc="[Tokenize] Structure token assembly"):
            ids = ppr_idx[i]
            ws = ppr_w[i]
            tokens[i, start:start + cfg.num_structure_tokens, :feat_dim] = padded_features[ids]
            tokens[i, start:start + cfg.num_structure_tokens, feat_dim] = ws[:cfg.num_structure_tokens]

    # Component 4: content-aware virtual connections + approximate PPR.
    if cfg.num_content_tokens > 0:
        num_classes = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        print(f"[Tokenize] Content-aware virtual connections: source={cfg.content_label_source}, classes={num_classes}")
        adj_content = add_content_supernodes(adj, labels, train_idx, val_idx, test_idx, num_classes, source=cfg.content_label_source)
        ppr_idx, ppr_w = ppr_topk_for_nodes(
            adj_content, np.arange(num_nodes, dtype=np.int64), cfg.num_content_tokens,
            cfg.ppr_alpha, cfg.content_ppr_eps, cfg.content_ppr_max_itr, rng,
            desc="[Tokenize] Content PPR",
        )
        padded_features = np.zeros((adj_content.shape[0], feat_dim), dtype=dtype)
        padded_features[:num_nodes] = x0
        start = 1 + cfg.hops + cfg.num_structure_tokens
        for i in tqdm(range(num_nodes), desc="[Tokenize] Content token assembly"):
            ids = ppr_idx[i]
            ws = ppr_w[i]
            tokens[i, start:start + cfg.num_content_tokens, :feat_dim] = padded_features[ids]
            tokens[i, start:start + cfg.num_content_tokens, feat_dim] = ws[:cfg.num_content_tokens]

    tokens.flush()
    meta: Dict[str, object] = {
        "dataset": dataset_name,
        "shape": list(shape),
        "dtype": np.dtype(dtype).name,
        "config": asdict(cfg),
        "fingerprint": fp,
        "token_path": str(token_path),
        "note": "Offline VCR tokenization. Training/evaluation time reported by vcr_bilgt_baseline.py excludes this one-time preprocessing unless --include_preprocess_in_time is used.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return token_path, shape, meta
