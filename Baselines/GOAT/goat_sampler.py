"""GOAT local sampler (official/sampler.py: LocalSampler) without torch_sparse, vectorised over the mini-batch.

Official procedure for one prediction node c and sizes [s_1, ..., s_L] (LocalSampler.sample_one):
    n_id = [c]
    for s in sizes:  adj_t, n_id = adj_t.sample_adj(n_id, s, replace=False)
torch_sparse.sample_adj draws min(s, deg(v)) distinct neighbours, without replacement, for EVERY node v already
in n_id (the prediction node and all nodes of earlier hops, not only the last frontier) and appends the nodes
that are not in n_id yet. A node added in round k gets distance k (the prediction node 0), and every edge
(token -> c) carries [distance, number of tokens of c at that distance]. LocalSampler.sample concatenates the
token sets of the batch, relabels them so that the batch nodes come first, followed by the other sampled nodes
in increasing id order, and returns ([edge_index; edge_dist], node_idx, batch_size).

Here the rounds run for all prediction nodes of the batch at once: (owner, node) pairs keyed by owner * N + node,
distinct neighbours drawn with Floyd's algorithm (uniform over the s-subsets of the neighbour list, as sampling
without replacement), and duplicates removed with sorted keys. The order of the edges inside a batch differs
from the official per-node loop; the model is invariant to it (attention is a softmax over each node's tokens,
distances are given explicitly). test_goat.py compares this sampler with a per-node transcription of
sample_one: identical token sets, distances and counts whenever s >= degree (no randomness), and the
expected structure and subset frequencies otherwise.

Random numbers come from torch's global generator, which DataLoader seeds separately in every worker.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch


class GraphCSR:
    """Neighbour lists of a symmetric graph: neighbours of v = indices[indptr[v]:indptr[v + 1]]."""

    def __init__(self, adj: sp.csr_matrix):
        adj = adj.tocsr()
        self.num_nodes = int(adj.shape[0])
        self.indptr = adj.indptr.astype(np.int64)
        self.indices = adj.indices.astype(np.int64)
        self.deg = np.diff(self.indptr)


def _floyd_positions(deg: np.ndarray, s: int) -> np.ndarray:
    """[M, s] distinct positions in [0, deg) per row (deg > s), uniform over the s-subsets (Floyd)."""
    m = deg.shape[0]
    out = np.empty((m, s), dtype=np.int64)
    for j in range(s):
        upper = deg - s + j                                   # draw from [0, upper]
        u = torch.rand(m, dtype=torch.float64).numpy()
        t = np.minimum((u * (upper + 1)).astype(np.int64), upper)
        if j:
            taken = (out[:, :j] == t[:, None]).any(axis=1)
            t = np.where(taken, upper, t)
        out[:, j] = t
    return out


def _sample_round(g: GraphCSR, owner: np.ndarray, node: np.ndarray, s: int):
    """min(s, deg) distinct neighbours of every (owner, node) pair; returns the (owner, neighbour) pairs."""
    deg = g.deg[node]
    start = g.indptr[node]
    full = deg <= s
    # pairs whose whole neighbour list is taken
    f_deg = deg[full]
    f_tot = int(f_deg.sum())
    if f_tot:
        f_owner = np.repeat(owner[full], f_deg)
        offs = np.arange(f_tot, dtype=np.int64) - np.repeat(np.cumsum(f_deg) - f_deg, f_deg)
        f_nb = g.indices[np.repeat(start[full], f_deg) + offs]
    else:
        f_owner = f_nb = np.empty(0, dtype=np.int64)
    # pairs with more than s neighbours: s distinct positions
    part = ~full
    if part.any():
        pos = _floyd_positions(deg[part], s)
        p_nb = g.indices[start[part][:, None] + pos].reshape(-1)
        p_owner = np.repeat(owner[part], s)
    else:
        p_owner = p_nb = np.empty(0, dtype=np.int64)
    return np.concatenate([f_owner, p_owner]), np.concatenate([f_nb, p_nb])


def _sorted_unique(a: np.ndarray) -> np.ndarray:
    """np.unique by sorting (NumPy >= 2.3 switched np.unique to a hash table, several times slower here)."""
    if a.size == 0:
        return a
    s = np.sort(a)
    keep = np.empty(s.size, dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def _merge_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union of two sorted, disjoint arrays (timsort merges the two runs in linear time)."""
    return np.sort(np.concatenate([a, b]), kind="stable")


def sample_batch(g: GraphCSR, batch: np.ndarray, sizes: List[int]):
    """Tokens of every prediction node in `batch` -> (edge_pack [4, E], node_idx [n], batch_size)."""
    batch = np.asarray(batch, dtype=np.int64).reshape(-1)
    b = batch.shape[0]
    n = g.num_nodes
    L = len(sizes)
    owner = np.arange(b, dtype=np.int64)          # pairs (owner = position in batch, node)
    node = batch.copy()
    dist = np.zeros(b, dtype=np.int64)
    known = np.sort(owner * n + node)             # keys of all pairs so far
    counts = np.zeros((b, L + 1), dtype=np.int64)
    counts[:, 0] = 1
    for hop, s in enumerate(sizes, start=1):
        c_owner, c_node = _sample_round(g, owner, node, int(s))   # rounds sample from ALL nodes so far
        if c_owner.size:
            keys = _sorted_unique(c_owner * n + c_node)
            pos = np.searchsorted(known, keys)
            pos_c = np.minimum(pos, known.size - 1)
            new = keys[known[pos_c] != keys]
        else:
            new = np.empty(0, dtype=np.int64)
        if new.size:
            n_owner, n_node = new // n, new % n
            owner = np.concatenate([owner, n_owner])
            node = np.concatenate([node, n_node])
            dist = np.concatenate([dist, np.full(new.size, hop, dtype=np.int64)])
            counts[:, hop] = np.bincount(n_owner, minlength=b)
            known = _merge_sorted(known, new)     # both sorted and disjoint
    # relabel: batch nodes first (in batch order), then the other tokens in increasing id order
    uniq = _sorted_unique(node)
    order = np.argsort(batch, kind="stable")
    is_batch = np.zeros(uniq.size, dtype=bool)
    is_batch[np.searchsorted(uniq, batch[order])] = True
    others = uniq[~is_batch]
    local_of_uniq = np.empty(uniq.size, dtype=np.int64)
    local_of_uniq[~is_batch] = b + np.arange(others.size, dtype=np.int64)
    local_of_uniq[is_batch] = order
    src = local_of_uniq[np.searchsorted(uniq, node)]
    edge_pack = np.stack([src, owner, dist, counts[owner, dist]], axis=0)
    node_idx = np.concatenate([batch, others])
    return torch.from_numpy(edge_pack), torch.from_numpy(node_idx), b


class LocalSampler(torch.utils.data.DataLoader):
    """DataLoader over prediction nodes; same outputs as official/sampler.py LocalSampler."""

    def __init__(self, graph: GraphCSR, sizes: List[int], node_idx: Optional[torch.Tensor] = None, **kwargs):
        kwargs.pop("collate_fn", None)
        kwargs.pop("dataset", None)
        self.graph = graph
        self.sizes = [int(s) for s in sizes]
        if node_idx is None:
            node_idx = torch.arange(graph.num_nodes)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        super().__init__(node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        return sample_batch(self.graph, np.asarray(batch, dtype=np.int64), self.sizes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sizes={self.sizes})"
