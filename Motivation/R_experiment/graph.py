"""Graph loading and original-graph lookup tables (relation-evidence diagnostic, Fig. 1a).

load_graph    symmetrize, drop self loops, binarize; row-L1 or raw features; splits as indices or 1-D/2-D boolean masks
GraphTables   CSR adjacency + sorted edge keys (O(log E) edge lookup), degrees, row-normalized P = D^-1 A,
              degree bins and center distances inside h-hop balls
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

DEGREE_EDGES = (1, 2, 4, 8, 16, 32)          # degree bins: 0 | 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33+
N_DEGREE_BINS = len(DEGREE_EDGES) + 2


def load_graph(root, normalize="l1", split_index=0):
    root = Path(root)
    if (root / 'train_split0.npy').is_file():
        from SCOPE_GT.data import load_graph as load_scope_graph
        graph = load_scope_graph(root, f'_split{split_index}', normalize)
        return graph.adj, graph.features, graph.labels, graph.splits
    adj = sp.load_npz(root / "adj.npz").tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    feat = np.load(root / "feat.npy").astype(np.float32)
    if normalize not in ("l1", "none"):
        raise ValueError("feature normalization must be l1 or none")
    if normalize == "l1":
        feat /= np.maximum(np.abs(feat).sum(1, keepdims=True), 1e-12)
    label = np.load(root / "label.npy").astype(np.int64).reshape(-1)
    split = {}
    for name in ("train", "val", "test"):
        ids = np.load(root / f"{name}.npy")
        if ids.ndim == 2:
            if ids.dtype != np.bool_:
                raise ValueError("2-D splits must be boolean masks")
            if ids.shape[0] == len(label):
                if not 0 <= split_index < ids.shape[1]:
                    raise ValueError("split-index out of range")
                ids = ids[:, split_index]
            elif ids.shape[1] == len(label):
                if not 0 <= split_index < ids.shape[0]:
                    raise ValueError("split-index out of range")
                ids = ids[split_index]
            else:
                raise ValueError("2-D split mask has no node-sized axis")
        split[name] = np.flatnonzero(ids) if ids.dtype == np.bool_ else ids.astype(np.int64).reshape(-1)
    return adj, feat, label, split


class GraphTables:
    def __init__(self, adj):
        a = sp.csr_matrix(adj, dtype=np.float32)
        a.sort_indices()
        self.n = a.shape[0]
        self.indptr = a.indptr.astype(np.int64)
        self.indices = a.indices.astype(np.int64)
        self.degree = np.diff(self.indptr)
        self.inv_degree = np.where(self.degree > 0, 1.0 / np.maximum(self.degree, 1), 0.0)
        self.A = sp.csr_matrix((np.ones(len(self.indices), np.float32), self.indices, self.indptr),
                               shape=(self.n, self.n))
        # full-graph row-normalized P (isolated nodes have zero rows)
        self.P = sp.csr_matrix((np.repeat(self.inv_degree, self.degree), self.indices, self.indptr),
                               shape=(self.n, self.n))
        self.keys = np.repeat(np.arange(self.n, dtype=np.int64), self.degree) * self.n + self.indices
        # adjacency reads of the two-step row P[i,:]P: sum_{k in N(i)} deg(k)
        self.two_hop_cost = np.asarray(self.A @ self.degree.astype(np.float64)).ravel()
        self.mean_degree = float(self.degree.mean()) if self.n else 0.0
        self.ball_row = None
        self.ball = None

    def has_edge(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, np.int64), np.asarray(v, np.int64))
        if not len(self.keys):
            return np.zeros(u.shape, bool)
        query = np.maximum(u, 0) * self.n + np.maximum(v, 0)
        pos = np.minimum(np.searchsorted(self.keys, query), len(self.keys) - 1)
        return (self.keys[pos] == query) & (u >= 0) & (v >= 0)

    def degree_bin(self, nodes):
        return np.digitize(self.degree[nodes], DEGREE_EDGES, right=True) + (self.degree[nodes] > 0)

    def build_balls(self, nodes, hops, block=1024):
        """h-hop BFS from every node: stores ball members (excluding the node) and shortest-path distances 1..h; blocked to bound peak memory."""
        nodes = np.unique(np.asarray(nodes, np.int64))
        parts = []
        for start in range(0, len(nodes), block):
            rows = nodes[start:start + block]
            m = len(rows)
            seen = sp.csr_matrix((np.ones(m, np.float32), (np.arange(m), rows)), shape=(m, self.n))
            frontier, dist = seen, sp.csr_matrix((m, self.n), dtype=np.float32)
            for k in range(1, hops + 1):
                nxt = (frontier @ self.A).tocsr()
                nxt.data[:] = 1.0
                nxt = (nxt - nxt.multiply(seen)).tocsr()
                nxt.eliminate_zeros()
                if not nxt.nnz:
                    break
                dist = (dist + k * nxt).tocsr()
                seen = (seen + nxt).tocsr()
                seen.data[:] = 1.0
                frontier = nxt
            parts.append(dist)
        ball = sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, self.n), dtype=np.float32)
        ball.sort_indices()
        self.ball = sp.csr_matrix((ball.data.astype(np.int8), ball.indices, ball.indptr), shape=ball.shape)
        self.ball_row = np.full(self.n, -1, np.int64)
        self.ball_row[nodes] = np.arange(len(nodes))
        return self.ball
