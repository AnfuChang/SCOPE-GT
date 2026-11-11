"""Graph index, edge directions and center-keyed random streams.

GraphIndex      symmetric binary graph (the adj returned by load_graph): CSR adjacency + sorted edge keys u*N + v,
                O(log E) edge lookup and the position of an edge in the CSR arrays
DirectionIndex  directions of the raw adj.npz: directed-arc lookup. load_graph symmetrizes the graph, so directions
                are read again from the raw file; disabled automatically when the raw graph is symmetric or stores
                a single triangle (directions carry no information)
CenterStream    every uniform number depends only on (derived key, center id, column), not on batch composition
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)

DIRECTION_MODES = ("auto", "on", "off")


class CenterStream:
    """Center-keyed random stream: each uniform number depends only on (derived key, center id, column), not on batch composition."""

    def __init__(self, *parts):
        self.key = np.uint64(np.random.SeedSequence([int(p) for p in parts]).generate_state(1, np.uint64)[0])

    def random(self, centers, cols):
        """centers [B], cols: nonnegative integers broadcastable to [B, W] -> uniform numbers in (0, 1), shape [B, W]."""
        c = np.asarray(centers).astype(np.uint64)[:, None]
        z = self.key ^ (c * _M1) ^ (np.asarray(cols).astype(np.uint64) * _M2 + _M3)
        z ^= z >> np.uint64(30)
        z *= _M2
        z ^= z >> np.uint64(27)
        z *= _M3
        z ^= z >> np.uint64(31)
        return ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)

def binary_pattern(mat):
    """Any sparse matrix -> binary CSR without self loops or explicit zeros (sorted column indices)."""
    coo = sparse.coo_matrix(mat)
    keep = (coo.row != coo.col) & (coo.data != 0)
    out = sparse.csr_matrix((np.ones(int(keep.sum()), np.int8), (coo.row[keep], coo.col[keep])), shape=coo.shape)
    out.sum_duplicates()
    out.sort_indices()
    return out


class EdgeKeys:
    """CSR pattern + sorted edge keys u*N + v."""

    def __init__(self, csr):
        self.n = int(csr.shape[0])
        self.indptr = csr.indptr.astype(np.int64)
        self.indices = csr.indices.astype(np.int64)
        self.degree = np.diff(self.indptr)
        self.keys = np.repeat(np.arange(self.n, dtype=np.int64), self.degree) * self.n + self.indices

    @property
    def nnz(self):
        return len(self.indices)

    def edge_pos(self, u, v):
        """Global CSR position of edge (u, v); -1 if absent (or if u / v < 0)."""
        u, v = np.broadcast_arrays(np.asarray(u, np.int64), np.asarray(v, np.int64))
        if not len(self.keys):
            return np.full(u.shape, -1, np.int64)
        query = np.maximum(u, 0) * self.n + np.maximum(v, 0)
        pos = np.minimum(np.searchsorted(self.keys, query), len(self.keys) - 1)
        hit = (self.keys[pos] == query) & (u >= 0) & (v >= 0)
        return np.where(hit, pos, -1)

    def has_edge(self, u, v):
        return self.edge_pos(u, v) >= 0


class GraphIndex(EdgeKeys):
    """Symmetric binary graph used for candidate expansion and relation queries."""

    def __init__(self, adj):
        super().__init__(binary_pattern(adj))


def load_raw_adjacency(root):
    """Raw adj.npz (not symmetrized), used only to recover edge directions."""
    return binary_pattern(sparse.load_npz(Path(root) / "adj.npz"))


class DirectionIndex:
    """Directions of the raw arcs. When directed is False, the direction category of an adjacent pair
    degenerates to 'bidirectional / undirected'.

    auto  enabled when the raw graph is asymmetric; if it stores exactly one triangle (all arcs i<j or all i>j)
          with no reciprocal arcs, a directed graph cannot be told apart from half of an undirected one, so
          directions are disabled with a notice (use on to force them)
    on    enabled whenever the raw graph is asymmetric
    off   directions are not used
    """

    def __init__(self, raw, sym, mode="auto"):
        """raw: raw adjacency (may be None when mode = off); sym: GraphIndex."""
        if mode not in DIRECTION_MODES:
            raise ValueError(f"direction must be one of {DIRECTION_MODES}, got {mode!r}")
        self.arcs = int(sym.nnz)
        self.directed = False
        self.reciprocity = 1.0
        if mode != "off":
            raw = binary_pattern(raw)
            if raw.shape[0] != sym.n:
                raise ValueError("raw adjacency and symmetric graph have different node counts")
            self.arcs = int(raw.nnz)
        if mode == "off":
            self.note = "direction off: treated as undirected"
        elif self.arcs == sym.nnz:              # with nonnegative weights the raw pattern is contained in max(A, A^T); equal arc counts mean symmetric
            self.note = "raw adjacency is symmetric: directions carry no information, treated as undirected"
        else:
            arcs = EdgeKeys(raw)
            rows = np.repeat(np.arange(arcs.n, dtype=np.int64), arcs.degree)
            reverse = arcs.has_edge(arcs.indices, rows)
            self.reciprocity = float(reverse.mean()) if self.arcs else 1.0
            triangular = bool((rows < arcs.indices).all() or (rows > arcs.indices).all())
            if mode == "auto" and triangular and not reverse.any():
                self.note = ("raw adjacency stores a single triangle with no reciprocal arcs: possibly half of an undirected "
                             "graph; directions disabled (use --direction on for a genuinely directed graph)")
            else:
                self.directed = True
                self.fwd = arcs
                self.note = (f"directed: {self.arcs} arcs, reciprocity {self.reciprocity:.1%}; "
                             "edge directions enter the topology relation")

    @property
    def lookups_per_pair(self):
        """Number of directed lookups needed to classify an adjacent pair (non-adjacent pairs are not queried)."""
        return 2 if self.directed else 0

    def pair_category(self, u, v, adjacent):
        """0 = no edge, 1 = bidirectional (or undirected), 2 = only u->v, 3 = only v->u. adjacent is has_edge(u, v) on the symmetric graph."""
        u, v, adjacent = np.broadcast_arrays(np.asarray(u, np.int64), np.asarray(v, np.int64),
                                             np.asarray(adjacent, bool))
        if not self.directed:
            return adjacent.astype(np.int64)
        result = np.zeros(adjacent.shape, np.int64)
        if adjacent.any():
            a, b = u[adjacent], v[adjacent]
            forward, backward = self.fwd.has_edge(a, b), self.fwd.has_edge(b, a)
            result[adjacent] = np.where(forward & backward, 1, np.where(forward, 2, np.where(backward, 3, 0)))
        return result
