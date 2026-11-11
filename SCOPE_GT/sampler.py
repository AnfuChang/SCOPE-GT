"""Batched bounded walks, uniform distance strata, and exact pair-owned T/C/B.

H<=3. Discovery depth is not graph distance. C contains all original-graph
witnesses; three-hop DAG rows are shared by center and queried boundary vertex.
Background uses fixed proposals per unique context, filtered per center, no refill.
"""
from __future__ import annotations

import zlib
import numpy as np

N_EDGE, TOPOLOGY_DIM = 4, 3  # log degree c/u, Jaccard; hop and direction are categorical
QUERY_CHUNK = 262144
CTX_KEYS = ("ctx", "ctx_valid", "sample_hop", "hop", "edge", "topofeat", "connection_known",
            "moment_count", "moment_mass", "context_bg_count")
ACCESS_KEYS = ("candidate_reads", "distance_reads", "edge_queries", "witness_reads", "context_background_reads")
DIAGNOSTIC_KEYS = ("candidates", "candidates_before_cap", "candidate_truncated_centers", "witness_records",
                   "witness_external_records", "background_records", "background_proposal_records",
                   "relation_pairs", "three_hop_pairs")


def graph_fingerprint(graph):
    return f"{graph.n}:{graph.nnz}:{zlib.crc32(graph.indptr.tobytes()):08x}:{zlib.crc32(graph.indices.tobytes()):08x}"


def default_hop_shares(hops, near=.5):
    return np.ones(1) if hops == 1 else np.array([near] + [(1. - near) / (hops - 1)] * (hops - 1))


def largest_remainder(total, shares):
    raw = total * np.asarray(shares, np.float64)
    base = np.floor(raw).astype(np.int64)
    rest = total - int(base.sum())
    if rest:
        base[np.argsort(-(raw - base), kind='stable')[:rest]] += 1
    return base


def _find(keys, queries):
    if not len(keys):
        return np.zeros(np.shape(queries), np.int64), np.zeros(np.shape(queries), bool)
    at = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    return at, keys[at] == queries


def _coalesce(keys, weights=None):
    order = np.argsort(keys, kind='stable')
    values = keys[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1] if len(values) else np.empty(0, np.int64)
    if weights is not None:
        return values[starts], np.add.reduceat(weights[order], starts) if len(starts) else np.empty(0, np.float64)
    inverse = np.empty(len(keys), np.int64)
    if len(keys):
        first = np.r_[True, values[1:] != values[:-1]]
        inverse[order] = np.cumsum(first) - 1
    return values[starts], inverse


def _ragged_chunks(starts, counts):
    """Yield owner and source offsets, with bounded temporary arrays."""
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    for start in range(0, int(offsets[-1]), QUERY_CHUNK):
        position = np.arange(start, min(start + QUERY_CHUNK, int(offsets[-1])), dtype=np.int64)
        owner = np.searchsorted(offsets[1:], position, side='right')
        yield owner, starts[owner] + position - offsets[owner]


def _neighbors(graph, nodes, acc, purpose='witness_reads'):
    counts = graph.degree[nodes]
    acc[purpose] += int(counts.sum())
    owners, vertices = [], []
    for owner, at in _ragged_chunks(graph.indptr[nodes], counts):
        owners.append(owner)
        vertices.append(graph.indices[at])
    return (np.concatenate(owners), np.concatenate(vertices)) if owners else (
        np.empty(0, np.int64), np.empty(0, np.int64))


def balanced_offsets(degrees, draws):
    """Partial Fisher-Yates across rows; each cycle is without replacement."""
    degrees = np.asarray(degrees, np.int64)
    b, width = draws.shape
    offsets = np.full((b, width), -1, np.int64)
    prefix = np.broadcast_to(np.arange(width), (b, width)).copy()
    swapped_at = np.full((b, width), -1, np.int64)
    swapped_value = np.zeros((b, width), np.int64)
    rows = np.flatnonzero(degrees > 0)
    degree = degrees[rows]
    for column in range(width):
        position = column % degree
        prefix[rows[position == 0]] = np.arange(width)
        chosen = position + np.minimum((draws[rows, column] * (degree - position)).astype(np.int64),
                                       degree - position - 1)
        old = prefix[rows, position].copy()
        picked = chosen.copy()
        small = chosen < width
        picked[small] = prefix[rows[small], chosen[small]]
        far_rows = rows[~small]
        if column and len(far_rows):
            match = swapped_at[far_rows, :column] == chosen[~small, None]
            last = np.max(np.where(match, np.arange(column), -1), axis=1)
            seen = last >= 0
            far = picked[~small]
            far[seen] = swapped_value[far_rows[seen], last[seen]]
            picked[~small] = far
        offsets[rows, column] = picked
        prefix[rows[small], chosen[small]] = old[small]
        swapped_at[far_rows, column] = chosen[~small]
        swapped_value[far_rows, column] = old[~small]
    return offsets


def balanced_walks(graph, centers, streams, walks, hops, acc):
    """[B,W,H] non-backtracking paths, at most B*W*H actual moves, no retries."""
    b = len(centers)
    nodes = np.full((b, walks, hops), -1, np.int64)
    first = balanced_offsets(graph.degree[centers],
                             streams['walk_first'].random(centers, np.arange(walks)[None]))
    row, walk = np.repeat(np.arange(b), walks), np.tile(np.arange(walks), b)
    center = centers[row]
    current, previous = center.copy(), center.copy()
    alive = np.ones(len(row), bool)
    big = np.iinfo(np.int64).max // 4
    for depth in range(hops):
        index = np.flatnonzero(alive)
        if not len(index):
            break
        x, c = current[index], center[index]
        start, degree = graph.indptr[x], graph.degree[x]
        lo = hi = np.full(len(index), big, np.int64)
        if depth:
            at = graph.edge_pos(x, c)
            lo = np.where(at >= 0, at - start, big)
            acc['edge_queries'] += len(index)
            if depth > 1:
                at = graph.edge_pos(x, previous[index])
                hi = np.where((at >= 0) & (at - start != lo), at - start, big)
                acc['edge_queries'] += len(index)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        allowed = degree - (lo < big) - (hi < big)
        valid = allowed > 0
        if depth == 0:
            offset = first[row[index], walk[index]]
        else:
            uniform = streams['walk_step'].random(c, (walk[index] * hops + depth)[:, None])[:, 0]
            offset = np.minimum((uniform * np.maximum(allowed, 1)).astype(np.int64), np.maximum(allowed - 1, 0))
            offset += offset >= lo
            offset += offset >= hi
        active = index[valid]
        target = graph.indices[start[valid] + offset[valid]]
        nodes[row[active], walk[active], depth] = target
        previous[active], current[active] = current[active], target
        alive[index[~valid]] = False
        acc['candidate_reads'] += len(active)
    return nodes


def uniform_walks(graph, centers, streams, walks, hops, acc):
    """Ordinary unbiased walks: IID neighbor choices, including backtracking/revisits.

    Uses the same W*H move ceiling as the structured sampler, without refill.
    Center visits count as graph reads and are excluded only when forming tokens.
    """
    b = len(centers)
    nodes = np.full((b, walks, hops), -1, np.int64)
    row, walk = np.repeat(np.arange(b), walks), np.tile(np.arange(walks), b)
    center = centers[row]
    current = center.copy()
    alive = np.ones(len(row), bool)
    first = streams['walk_first'].random(centers, np.arange(walks)[None]).reshape(-1)
    for depth in range(hops):
        index = np.flatnonzero(alive)
        if not len(index):
            break
        degree = graph.degree[current[index]]
        valid = degree > 0
        active, degree = index[valid], degree[valid]
        if depth == 0:
            uniform = first[active]
        else:
            uniform = streams['walk_step'].random(center[active], (walk[active] * hops + depth)[:, None])[:, 0]
        offset = np.minimum((uniform * degree).astype(np.int64), degree - 1)
        target = graph.indices[graph.indptr[current[active]] + offset]
        nodes[row[active], walk[active], depth] = target
        current[active] = target
        alive[index[~valid]] = False
        acc['candidate_reads'] += len(active)
    return nodes


def common_pairs(graph, left, right, acc, purpose='witness_reads'):
    """Exact CN for all pairs, as flat (pair owner, vertex) arrays."""
    small = np.where(graph.degree[left] <= graph.degree[right], left, right)
    other = np.where(small == left, right, left)
    counts = graph.degree[small]
    acc[purpose] += int(counts.sum())
    owners, nodes = [], []
    for owner, at in _ragged_chunks(graph.indptr[small], counts):
        vertex = graph.indices[at]
        keep = graph.has_edge(other[owner], vertex)
        acc['edge_queries'] += len(vertex)
        owners.append(owner[keep])
        nodes.append(vertex[keep])
    return (np.concatenate(owners), np.concatenate(nodes)) if owners else (
        np.empty(0, np.int64), np.empty(0, np.int64))


def _take(rows, nodes, hops, quotas, centers, stream, radius):
    if not len(rows):
        return np.empty(0, np.int64)
    group = rows * radius + hops - 1
    random = stream.random(centers[rows], nodes[:, None])[:, 0]
    order = np.lexsort((nodes, random, group))
    ordered_group = group[order]
    starts = np.r_[0, np.flatnonzero(ordered_group[1:] != ordered_group[:-1]) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    return order[rank < quotas[rows[order], hops[order] - 1]]


def _quotas(available, total, shares):
    """Batched largest-remainder redistribution, preserving per-row tie order."""
    shares = np.asarray(shares, np.float64)
    take = np.minimum(largest_remainder(total, shares), available)
    while True:
        spare = (available > take) & (shares[None, :] > 0)
        rows = np.flatnonzero((take.sum(1) < total) & spare.any(1))
        if not len(rows):
            return take
        weights = np.where(spare[rows], shares[None, :], 0.)
        weights = weights / weights.sum(1, keepdims=True)
        need = total - take[rows].sum(1)
        raw = need[:, None] * weights
        add = np.floor(raw).astype(np.int64)
        rest = need - add.sum(1)
        order = np.argsort(-(raw - add), axis=1, kind='stable')
        add[np.arange(len(rows))[:, None], order] += np.arange(add.shape[1])[None, :] < rest[:, None]
        add = np.minimum(add, available[rows] - take[rows])
        empty = ~add.any(1)
        if empty.any():
            add[np.flatnonzero(empty), np.argmax(spare[rows[empty]], axis=1)] = 1
        take[rows] += add


class SequenceGraph:
    """Disjoint union of per-center induced graphs; vertex IDs map to original IDs."""
    def __init__(self, graph, centers, rows, contexts, acc):
        keys = np.unique(np.r_[np.arange(len(centers)) * graph.n + centers, rows * graph.n + contexts])
        self.node_ids = keys % graph.n
        owner = keys // graph.n
        sizes = np.bincount(owner, minlength=len(centers))
        offsets = np.r_[0, np.cumsum(sizes)]
        starts = offsets[:-1]
        edges_a, edges_b = [], []
        for sequence, at in _ragged_chunks(np.zeros(len(sizes), np.int64), sizes * sizes):
            a = starts[sequence] + at // sizes[sequence]
            b = starts[sequence] + at % sizes[sequence]
            keep = graph.has_edge(self.node_ids[a], self.node_ids[b])
            acc['edge_queries'] += len(a)
            edges_a.append(a[keep])
            edges_b.append(b[keep])
        a = np.concatenate(edges_a) if edges_a else np.empty(0, np.int64)
        self.indices = np.concatenate(edges_b) if edges_b else np.empty(0, np.int64)
        self.n = len(keys)
        self.degree = np.bincount(a, minlength=self.n)
        self.indptr = np.r_[0, np.cumsum(self.degree)]
        self.keys = a * self.n + self.indices
        self.center_pos = np.searchsorted(keys, np.arange(len(centers)) * graph.n + centers)
        self.context_pos = np.searchsorted(keys, rows * graph.n + contexts)

    def has_edge(self, a, b):
        return _find(self.keys, a * self.n + b)[1]

    def distances(self, radius, acc):
        distance = np.full(self.n, radius + 1, np.int64)
        frontier = self.center_pos
        distance[frontier] = 0
        for depth in range(1, radius + 1):
            _, nodes = _neighbors(self, frontier, acc, 'distance_reads')
            nodes = np.unique(nodes)
            frontier = nodes[distance[nodes] > depth]
            distance[frontier] = depth
        return distance[self.context_pos]


def three_hop_records(graph, centers, pair_rows, contexts, pair_ids, acc):
    """Exact, center-shared requested DAG; stream path contributions in bounded chunks.

    Only boundary vertices requested by the selected contexts are stored. For each
    center choose the cheaper of expanding N(c)'s rows or querying CN(c,b) on the
    requested b boundary. Both branches produce the same DAG edges a->b.
    """
    empty = np.empty(0, np.int64)
    if not len(pair_ids):
        return np.zeros(len(contexts)), np.zeros(len(contexts)), empty, empty, empty, np.empty(0)
    link_owner, boundary = _neighbors(graph, contexts[pair_ids], acc)
    link_pair = pair_ids[link_owner]
    boundary_keys, link_index = _coalesce(pair_rows[link_pair] * graph.n + boundary)
    boundary_row, boundary_node = boundary_keys // graph.n, boundary_keys % graph.n
    roots = np.unique(pair_rows[pair_ids])
    neighbor_owner, first_nodes = _neighbors(graph, centers[roots], acc)
    first_rows = roots[neighbor_owner]
    root_cost = np.bincount(first_rows, weights=graph.degree[first_nodes], minlength=len(centers))
    query_cost = np.bincount(boundary_row, weights=np.minimum(graph.degree[centers[boundary_row]],
                                                            graph.degree[boundary_node]), minlength=len(centers))
    expand_root = root_cost <= query_cost
    dag_boundary, dag_first = [], []
    use = np.flatnonzero(~expand_root[boundary_row])
    owner, vertex = common_pairs(graph, centers[boundary_row[use]], boundary_node[use], acc)
    dag_boundary.append(use[owner])
    dag_first.append(vertex)
    use = np.flatnonzero(expand_root[first_rows])
    counts = graph.degree[first_nodes[use]]
    acc['witness_reads'] += int(counts.sum())
    for owner, at in _ragged_chunks(graph.indptr[first_nodes[use]], counts):
        root = first_rows[use[owner]]
        target = graph.indices[at]
        found, keep = _find(boundary_keys, root * graph.n + target)
        dag_boundary.append(found[keep])
        dag_first.append(first_nodes[use[owner[keep]]])
    dag_b, dag_a = np.concatenate(dag_boundary), np.concatenate(dag_first)
    order = np.lexsort((dag_a, dag_b))
    dag_b, dag_a = dag_b[order], dag_a[order]
    counts = np.bincount(dag_b, minlength=len(boundary_keys))
    offsets = np.r_[0, np.cumsum(counts)]
    forward = 1. / (graph.degree[centers[boundary_row[dag_b]]] * graph.degree[dag_a])
    forward_mass = np.bincount(dag_b, weights=forward, minlength=len(boundary_keys))
    valid = counts[link_index] > 0
    links, bi = link_pair[valid], link_index[valid]
    count = np.bincount(links, weights=counts[bi], minlength=len(contexts))
    second_weight = forward_mass[bi] / graph.degree[boundary_node[bi]]
    mass = np.bincount(links, weights=second_weight, minlength=len(contexts))
    first_keys, first_weight = [], []
    for owner, at in _ragged_chunks(offsets[bi], counts[bi]):
        pair = links[owner]
        a = dag_a[at]
        weight = forward[at] / graph.degree[boundary_node[bi[owner]]]
        key, value = _coalesce(pair * graph.n + a, weight)
        first_keys.append(key)
        first_weight.append(value)
    if first_keys:
        key, value = _coalesce(np.concatenate(first_keys), np.concatenate(first_weight))
        first_pair, first_vertex = key // graph.n, key % graph.n
    else:
        first_pair, first_vertex, value = empty, empty, np.empty(0)
    return (count, mass, np.r_[first_pair, links], np.r_[first_vertex, boundary_node[bi]],
            np.r_[np.ones(len(first_pair), np.int64), np.full(len(links), 2, np.int64)],
            np.r_[value, second_weight])


def three_hop_records_independent(graph, centers, pair_rows, contexts, pair_ids, acc):
    """Same vectorized exact kernel, with a private root identity per requested pair.

    Retains chunking, cheaper-branch selection and within-pair coalescing. Only
    cross-pair DAG/query reuse is removed; no Python path enumeration is used.
    """
    ids = np.asarray(pair_ids, np.int64)
    count = np.zeros(len(contexts), np.float64)
    mass = np.zeros(len(contexts), np.float64)
    local = np.arange(len(ids), dtype=np.int64)
    n, m, pair, node, position, weight = three_hop_records(
        graph, centers[pair_rows[ids]], local, contexts[ids], local, acc)
    count[ids], mass[ids] = n, m
    return count, mass, ids[pair], node, position, weight


class ContextSampler:
    def __init__(self, graph, direction, budget=32, hops=2, hop_shares=None, background=4,
                 walks=None, candidate_multiplier=2, relation_source='graph', sampling='structured'):
        if budget < 1 or hops not in (1, 2, 3) or background < 0 or (walks is not None and walks < 1):
            raise ValueError('positive K/W, H in 1..3, nonnegative background proposals required')
        if not np.isfinite(candidate_multiplier) or candidate_multiplier < 1:
            raise ValueError('candidate multiplier must be finite and at least one')
        if relation_source not in ('graph', 'sequence', 'none'):
            raise ValueError('invalid relation source')
        if sampling not in ('structured', 'uniform_rw'):
            raise ValueError('invalid sampling mode')
        shares = default_hop_shares(hops) if hop_shares is None else np.asarray(hop_shares, np.float64)
        if shares.shape != (hops,) or not np.isfinite(shares).all() or (shares < 0).any() or shares.sum() <= 0:
            raise ValueError('one nonnegative quota per hop required')
        self.graph, self.direction = graph, direction
        self.k, self.hops, self.walks = int(budget), int(hops), int(walks or budget)
        self.background, self.relation_source = int(background), relation_source
        self.sampling = sampling
        self.shares = shares / shares.sum()
        self.base_quota = largest_remainder(self.k, self.shares)
        self.candidate_limit = int(np.ceil(candidate_multiplier * self.k))
        self.candidate_access_budget = self.walks * self.hops
        self.log_degree = np.log1p(graph.degree)
        self.fingerprint = graph_fingerprint(graph)

    @property
    def index_bytes(self):
        return int(self.log_degree.nbytes)

    @property
    def cache_signature(self):
        return (self.fingerprint, self.k, self.hops, self.walks, self.background,
                self.candidate_limit, self.shares.tobytes(), self.relation_source, self.direction.directed,
                self.sampling)

    def _select(self, rows, nodes, hops, total, centers, stream):
        if self.sampling == 'uniform_rw':
            # One pool per center: no true-distance quotas or preference for rare hops.
            groups = np.ones_like(hops)
            available = np.bincount(rows, minlength=len(centers))[:, None]
            return _take(rows, nodes, groups, np.minimum(available, total), centers, stream, 1)
        available = np.bincount(rows * self.hops + hops - 1,
                                minlength=len(centers) * self.hops).reshape(len(centers), self.hops)
        return _take(rows, nodes, hops, _quotas(available, total, self.shares), centers, stream, self.hops)

    def _sample(self, centers, streams, acc):
        walk = uniform_walks if self.sampling == 'uniform_rw' else balanced_walks
        paths = walk(self.graph, centers, streams, self.walks, self.hops, acc)
        valid = (paths >= 0) & (paths != centers[:, None, None])
        rr, _, tt = np.nonzero(valid)
        keys = rr * self.graph.n + paths[valid]
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1] if len(keys) else np.empty(0, np.int64)
        unique = sorted_keys[starts]
        rows, nodes = unique // self.graph.n, unique % self.graph.n
        upper = np.minimum.reduceat(tt[order] + 1, starts) if len(starts) else np.empty(0, np.int64)
        adjacent = upper == 1
        need = ~adjacent
        if need.any():
            adjacent[need] = self.graph.has_edge(centers[rows[need]], nodes[need])
            acc['edge_queries'] += int(need.sum())
        hops = np.where(adjacent, 1, 2)
        unresolved = np.flatnonzero((upper == 3) & ~adjacent)
        owner, witness = common_pairs(self.graph, centers[rows[unresolved]], nodes[unresolved], acc, 'distance_reads')
        cn_counts = np.bincount(owner, minlength=len(unresolved))
        hops[unresolved[cn_counts == 0]] = 3
        cache = (unique[unresolved], np.r_[0, np.cumsum(cn_counts)], witness)
        available = np.bincount(rows * self.hops + hops - 1, minlength=len(centers) * self.hops).reshape(len(centers), self.hops)
        acc['candidates_before_cap'] = len(nodes)
        acc['candidate_truncated_centers'] = int((available.sum(1) > self.candidate_limit).sum())
        take = self._select(rows, nodes, hops, self.candidate_limit, centers, streams['candidate'])
        rows, nodes, hops = rows[take], nodes[take], hops[take]
        acc['candidates'] = len(nodes)
        take = self._select(rows, nodes, hops, self.k, centers, streams['select'])
        rows, nodes, hops = rows[take], nodes[take], hops[take]
        counts = np.bincount(rows, minlength=len(centers))
        cols = np.arange(len(rows)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
        ctx = np.full((len(centers), self.k), -1, np.int64)
        sample_hop = np.zeros(ctx.shape, np.int64)
        ctx[rows, cols], sample_hop[rows, cols] = nodes, hops
        return dict(ctx=ctx, ctx_valid=ctx >= 0, sample_hop=sample_hop), cache

    def _common(self, graph, centers, rows, contexts, cache, acc):
        if cache is None:
            return common_pairs(graph, centers[rows], contexts, acc)
        keys, offsets, vertices = cache
        at, found = _find(keys, rows * graph.n + contexts)
        selected = np.flatnonzero(found)
        owners, nodes = [], []
        for owner, pos in _ragged_chunks(offsets[at[selected]], np.diff(offsets)[at[selected]]):
            owners.append(selected[owner])
            nodes.append(vertices[pos])
        missing = np.flatnonzero(~found)
        owner, vertex = common_pairs(graph, centers[rows[missing]], contexts[missing], acc)
        owners.append(missing[owner])
        nodes.append(vertex)
        return np.concatenate(owners), np.concatenate(nodes)

    def relations(self, centers, sample, streams, relation_source=None, common_cache=None, acc=None, three_hop_mode="shared"):
        """Build evidence for fixed tokens; useful for source controls and independent references."""
        if three_hop_mode not in ("shared", "independent"):
            raise ValueError("three_hop_mode must be shared or independent")
        centers = np.asarray(centers, np.int64)
        source_name = relation_source or self.relation_source
        if source_name not in ('graph', 'sequence', 'none'):
            raise ValueError('invalid relation source')
        acc = dict.fromkeys(ACCESS_KEYS + DIAGNOSTIC_KEYS, 0) if acc is None else acc
        out = {key: value.copy() for key, value in sample.items()}
        shape = out['ctx'].shape
        for name in ('hop', 'edge'):
            out[name] = np.zeros(shape, np.int64)
        out['topofeat'] = np.zeros(shape + (TOPOLOGY_DIM,), np.float32)
        out['connection_known'] = np.zeros(shape, bool)
        for name in ('moment_count', 'moment_mass', 'context_bg_count'):
            out[name] = np.zeros(shape, np.float64 if name != 'context_bg_count' else np.float32)
        rows, cols = np.nonzero(out['ctx_valid'])
        contexts = out['ctx'][rows, cols]
        pairs = len(rows)
        rec_pair, rec_node, rec_position, rec_weight = (np.empty(0, np.int64), np.empty(0, np.int64),
                                                       np.empty(0, np.int64), np.empty(0, np.float64))
        bg_pair, bg_node = np.empty(0, np.int64), np.empty(0, np.int64)
        if source_name != 'none' and pairs:
            graph = self.graph
            roots, targets = centers, contexts
            distance = out['sample_hop'][rows, cols]
            if source_name == 'sequence':
                graph = SequenceGraph(graph, centers, rows, contexts, acc)
                roots, targets = graph.center_pos, graph.context_pos
                distance = graph.distances(self.hops, acc)
                common_cache = None
            owner, vertex = self._common(graph, roots, rows, targets, common_cache, acc)
            count = np.bincount(owner, minlength=pairs)
            degree_c, degree_u = graph.degree[roots[rows]], graph.degree[targets]
            log_c = self.log_degree[roots[rows]] if source_name == 'graph' else np.log1p(degree_c)
            log_u = self.log_degree[targets] if source_name == 'graph' else np.log1p(degree_u)
            out['topofeat'][rows, cols] = np.column_stack([
                log_c, log_u, count / np.maximum(degree_c + degree_u - count, 1)])
            out['hop'][rows, cols] = distance
            out['edge'][rows, cols] = self.direction.pair_category(centers[rows], contexts,
                                                                  out['sample_hop'][rows, cols] == 1)
            acc['edge_queries'] += self.direction.lookups_per_pair * int((out['sample_hop'][rows, cols] == 1).sum())
            short = (distance <= 2) & (distance <= self.hops)
            keep = short[owner]
            rec_pair, rec_node = owner[keep], vertex[keep]
            rec_position = np.ones(len(rec_pair), np.int64)
            rec_weight = 1. / (degree_c[rec_pair] * graph.degree[rec_node])
            path_count = np.where(short, count, 0).astype(np.float64)
            path_mass = np.bincount(rec_pair, weights=rec_weight, minlength=pairs).astype(np.float64)
            triple = np.flatnonzero(distance == 3) if self.hops >= 3 else np.empty(0, np.int64)
            if len(triple):
                retrieve = three_hop_records if three_hop_mode == "shared" else three_hop_records_independent
                n3, m3, p3, v3, s3, w3 = retrieve(graph, roots, rows, targets, triple, acc)
                path_count += n3
                path_mass += m3
                rec_pair, rec_node = np.r_[rec_pair, p3], np.r_[rec_node, v3]
                rec_position, rec_weight = np.r_[rec_position, s3], np.r_[rec_weight, w3]
            order = np.lexsort((rec_node, rec_position, rec_pair))
            rec_pair, rec_node, rec_position, rec_weight = (x[order] for x in (rec_pair, rec_node, rec_position, rec_weight))
            if source_name == 'sequence':
                rec_node = graph.node_ids[rec_node]
            out['moment_count'][rows, cols], out['moment_mass'][rows, cols] = path_count, path_mass
            out['connection_known'][rows, cols] = distance <= self.hops
            acc['relation_pairs'], acc['three_hop_pairs'] = pairs, len(triple)
            if self.background and source_name == 'graph':
                unique, inverse = np.unique(contexts, return_inverse=True)
                degree = self.graph.degree[unique]
                width = min(self.background, int(degree.max()))
                random = streams['context_background'].random(unique, np.arange(width)[None])
                offsets = balanced_offsets(degree, random)
                valid = np.arange(width)[None] < degree[:, None]
                proposals = np.full(offsets.shape, -1, np.int64)
                a, b = np.nonzero(valid)
                proposals[a, b] = self.graph.indices[self.graph.indptr[unique[a]] + offsets[a, b]]
                acc['context_background_reads'] = len(a)
                p, slot = np.nonzero(proposals[inverse] >= 0)
                vertex = proposals[inverse[p], slot]
                sequence_keys = np.sort(np.r_[np.arange(len(centers)) * self.graph.n + centers,
                                               rows * self.graph.n + contexts])
                witness_keys = np.unique(rec_pair * self.graph.n + rec_node)
                allowed = ~_find(sequence_keys, rows[p] * self.graph.n + vertex)[1]
                allowed &= ~_find(witness_keys, p * self.graph.n + vertex)[1]
                bg_pair, bg_node = p[allowed], vertex[allowed]
                out['context_bg_count'][rows, cols] = np.bincount(bg_pair, minlength=pairs)
                acc['background_proposal_records'] = len(p)
        sequence_keys = np.sort(np.r_[np.arange(len(centers)) * self.graph.n + centers,
                                       rows * self.graph.n + contexts])
        external = ~_find(sequence_keys, rows[rec_pair] * self.graph.n + rec_node)[1]
        out.update(moment_rows=rows[rec_pair], moment_cols=cols[rec_pair], moment_nodes=rec_node,
                   moment_weight=rec_weight.astype(np.float32), moment_position=rec_position, moment_external=external,
                   context_bg_rows=rows[bg_pair], context_bg_cols=cols[bg_pair], context_bg_nodes=bg_node)
        acc['witness_records'], acc['witness_external_records'] = len(rec_node), int(external.sum())
        acc['background_records'] = len(bg_node)
        out.update(acc)
        return out

    def draw(self, centers, streams, relation_source=None):
        centers = np.asarray(centers, np.int64)
        if centers.ndim != 1 or not len(centers) or (centers < 0).any() or (centers >= self.graph.n).any():
            raise ValueError('centers must be a nonempty vector of valid node IDs')
        acc = dict.fromkeys(ACCESS_KEYS + DIAGNOSTIC_KEYS, 0)
        sample, cache = self._sample(centers, streams, acc)
        return self.relations(centers, sample, streams, relation_source, cache, acc)


def trim_padding(draw):
    columns = draw['ctx_valid'].any(0)
    width = int(np.flatnonzero(columns)[-1]) + 1 if columns.any() else 1
    if width == draw['ctx_valid'].shape[1]:
        return draw
    out = dict(draw)
    for key in CTX_KEYS:
        out[key] = out[key][:, :width]
    for key in ('connection_count_used', 'connection_mass_used', 'connection_positions_used'):
        if key in out:
            out[key] = out[key][:, :width]
    return out
