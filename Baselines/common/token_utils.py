"""Vectorised versions of the token-construction steps shared by GCFormer and SwapGT.

Both official repos (JHL-HUST/GCFormer, JHL-HUST/SwapGT; built on NAGphormer) build node tokens by
  1. normalize_adj(A + I)                       D^-1/2 (A+I) D^-1/2          (data.py)
  2. node_neighborhood_feature(adj, X, K, a)    X_{t+1} = (1-a) A X_t + a X_0   (utils.py)
  3. node_seq_feature                          s = x_b x^T, s[i, i] = -1000, topk(s, k)   (utils.py)
The official step 3 fills [N, k+1, F] tensors node by node in Python and step 2/3 run on CPU; the
functions below return exactly the same neighbours/features, computed in chunks (on the GPU when
one is available) so that graphs with 10^5-10^6 nodes finish.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def sym_norm_adj_with_self_loops(adj: sp.spmatrix) -> sp.csr_matrix:
    a = sp.csr_matrix(adj, dtype=np.float32) + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    rowsum = np.asarray(a.sum(1)).flatten()
    with np.errstate(divide="ignore"):
        r_inv = np.power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.0
    d = sp.diags(r_inv.astype(np.float32))
    return (d @ a @ d).tocsr().astype(np.float32)


def propagate(adj_norm: sp.csr_matrix, x: np.ndarray, steps: int, alpha: float) -> np.ndarray:
    """X_{t+1} = (1 - alpha) * A X_t + alpha * X_0, `steps` times (alpha = 0.1 in both repos)."""
    x0 = np.asarray(x, dtype=np.float32)
    out = x0
    for _ in range(steps):
        out = (1.0 - alpha) * (adj_norm @ out) + alpha * x0
    return np.ascontiguousarray(out, dtype=np.float32)


@torch.no_grad()
def topk_similar(x: np.ndarray, k: int, device: torch.device, l2_normalize: bool,
                 max_elems: int = 1 << 27) -> torch.Tensor:
    """Indices [N, k] of the k most similar nodes (self excluded by setting s[i, i] = -1000).

    l2_normalize=True  -> cosine similarity (GCFormer: torch.nn.functional.normalize before x x^T)
    l2_normalize=False -> raw dot product (SwapGT)
    Ties are broken by torch.topk, as in the official code.
    """
    n = x.shape[0]
    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
    if l2_normalize:
        xt = torch.nn.functional.normalize(xt, dim=1)
    rows = max(1, min(n, max_elems // max(n, 1)))
    out = torch.empty(n, k, dtype=torch.long)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        s = xt[start:stop] @ xt.t()
        r = torch.arange(stop - start, device=device)
        s[r, r + start] = -1000.0
        out[start:stop] = torch.topk(s, k, dim=1).indices.cpu()
    return out


def sample_excluding(n: int, exclude: torch.Tensor, k: int, rng: np.random.Generator,
                     replace: bool = False) -> torch.Tensor:
    """For every row, k nodes drawn uniformly from {0..n-1} minus that row's `exclude` indices.

    Official GCFormer: np.random.choice(list(set(all) - set(topk)), k, replace=False) for graphs with
    fewer than sample_size (=50000) nodes. Drawing uniform candidates and keeping the first k that
    are new and allowed is the same distribution (sequential sampling without replacement).
    """
    rows = exclude.shape[0]
    excl = exclude.numpy().astype(np.int64)
    if not replace and n - excl.shape[1] < k:
        replace = True
    out = np.empty((rows, k), dtype=np.int64)
    draw = 2 * (k + excl.shape[1]) + 8
    todo = np.arange(rows)
    while len(todo):
        still = []
        for start in range(0, len(todo), 100_000):
            idx = todo[start:start + 100_000]
            cand = rng.integers(0, n, size=(len(idx), draw))
            ok = ~(cand[:, :, None] == excl[idx][:, None, :]).any(axis=2)
            if not replace:
                # keep only the first occurrence of each value within a row
                order = np.argsort(cand, axis=1, kind="stable")
                srt = np.take_along_axis(cand, order, axis=1)
                dup_sorted = np.zeros_like(srt, dtype=bool)
                dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
                dup = np.empty_like(dup_sorted)
                np.put_along_axis(dup, order, dup_sorted, axis=1)
                ok &= ~dup
            rank = np.cumsum(ok, axis=1)
            full = rank[:, -1] >= k
            sel = ok & (rank <= k)
            good = np.flatnonzero(full)
            out[idx[good]] = cand[good][sel[good]].reshape(len(good), k)
            still.append(idx[~full])
        todo = np.concatenate(still) if still else np.empty(0, dtype=np.int64)
    return torch.from_numpy(out)
