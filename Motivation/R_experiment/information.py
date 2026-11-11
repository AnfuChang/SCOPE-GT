"""Original-graph evidence channels for a common sampled Graph Transformer."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

CHANNELS = ('witness', 'strength', 'neighborhood')
MODES = ('plain', 'witness', 'strength', 'neighborhood', 'all')


def active_channels(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    return set(CHANNELS) if mode == 'all' else set() if mode == 'plain' else {mode}


class RelationOracle:
    """Blockwise full-neighbor feature sums; no dense N-by-N relation matrix."""
    def __init__(self, graph, features, neighborhood_block_size=512, need_neighborhood=True):
        if neighborhood_block_size < 1:
            raise ValueError('neighborhood_block_size must be positive')
        self.graph, self.features = graph, np.asarray(features, np.float32)
        if self.features.ndim != 2 or len(self.features) != graph.n:
            raise ValueError('Oracle feature/graph shape mismatch')
        self.neighbor_sums, self.preprocess_graph_reads = None, 0
        if need_neighborhood:
            self.neighbor_sums = np.empty_like(self.features)
            for start in range(0, graph.n, neighborhood_block_size):
                stop = min(start + neighborhood_block_size, graph.n)
                self.neighbor_sums[start:stop] = graph.A[start:stop] @ self.features
            self.preprocess_graph_reads = int(len(graph.indices))


class InformationTables:
    """Pair-owned full-graph evidence, including nodes in the sampled sequence.

    Witness is the uniform mean of distinct common-neighbor attributes.
    Strength is their count and sum 1/(degree(c)*degree(w)) = P²[c,u].
    Context background uses N(u) minus shared neighbors and both endpoints.
    Center self slots and padding have no relation evidence.
    """
    def __init__(self, oracle, centers, contexts, channels=None):
        channels = set(CHANNELS) if channels is None else set(channels)
        if not channels <= set(CHANNELS):
            raise ValueError(f'Unknown evidence channels: {channels}')
        graph, features = oracle.graph, oracle.features
        centers, contexts = np.asarray(centers, np.int64), np.asarray(contexts, np.int64)
        if centers.ndim != 1 or contexts.ndim != 2 or len(centers) != len(contexts):
            raise ValueError('centers/contexts must have shapes [B] and [B,K]')
        if ((centers < 0) | (centers >= graph.n)).any() or ((contexts < -1) | (contexts >= graph.n)).any():
            raise ValueError('Invalid node identifiers; padding must be -1')
        self.nodes = np.concatenate([centers[:, None], contexts], axis=1)
        self.valid = self.nodes >= 0
        shape, width = self.nodes.shape, features.shape[1]
        self.witness_mean = np.zeros(shape + (width,), np.float32)
        self.neighborhood_mean = np.zeros_like(self.witness_mean)
        self.strength_raw = np.zeros(shape + (2,), np.float32)
        self.present = np.zeros(shape + (3,), bool)
        self.witness_count = np.zeros(shape, np.int64)
        self.neighborhood_count = np.zeros(shape, np.int64)
        self.graph_reads = self.attribute_reads = 0
        if not channels:
            return
        if 'neighborhood' in channels and oracle.neighbor_sums is None:
            raise ValueError('Neighborhood sums were not prepared')
        for row, c in enumerate(centers):
            nc = graph.indices[graph.indptr[c]:graph.indptr[c + 1]]
            self.graph_reads += len(nc)
            for slot in np.flatnonzero(contexts[row] >= 0):
                col, u = slot + 1, contexts[row, slot]
                if u == c:
                    raise ValueError('A context cannot duplicate its center')
                nu = graph.indices[graph.indptr[u]:graph.indptr[u + 1]]
                self.graph_reads += len(nu)
                shared = np.intersect1d(nc, nu, assume_unique=True)
                count = len(shared)
                self.witness_count[row, col] = count
                if count:
                    self.present[row, col, 0] = 'witness' in channels
                    self.present[row, col, 1] = 'strength' in channels
                    if 'strength' in channels:
                        mass = np.sum(graph.inv_degree[c] * graph.inv_degree[shared])
                        self.strength_raw[row, col] = count, mass
                shared_sum = None
                if 'witness' in channels or 'neighborhood' in channels:
                    shared_sum = features[shared].sum(axis=0, dtype=np.float64)
                    self.attribute_reads += count
                    if count and 'witness' in channels:
                        self.witness_mean[row, col] = shared_sum / count
                if 'neighborhood' in channels:
                    adjacent = bool(graph.has_edge(c, u))
                    background_count = len(nu) - count - int(adjacent)
                    if background_count < 0:
                        raise AssertionError('Invalid nonshared neighborhood count')
                    self.neighborhood_count[row, col] = background_count
                    if background_count:
                        background_sum = oracle.neighbor_sums[u].astype(np.float64) - shared_sum
                        if adjacent:
                            background_sum -= features[c]
                            self.attribute_reads += 1
                        self.neighborhood_mean[row, col] = background_sum / background_count
                        self.present[row, col, 2] = True

    def inputs(self, mode, tau):
        enabled = np.array([name in active_channels(mode) for name in CHANNELS])
        present = self.present & enabled[None, None]
        tau = np.asarray(tau, np.float32)
        if tau.shape != (2,) or not np.isfinite(tau).all() or (tau <= 0).any():
            raise ValueError('Strength tau must contain two finite positive values')
        return (self.witness_mean * present[..., :1], np.log1p(self.strength_raw / tau) * present[..., 1:2],
                self.neighborhood_mean * present[..., 2:3], present)


class InformationBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout, channel_init):
        super().__init__()
        self.heads, self.dh, self.dim = heads, dim // heads, dim
        self.ln = nn.LayerNorm(dim)
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(dim, dim) for _ in range(3))
        self.rel_k = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in CHANNELS)
        self.rel_v = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in CHANNELS)
        self.lambda_k = nn.Parameter(torch.full((3,), float(channel_init)))
        self.lambda_v = nn.Parameter(torch.full((3,), float(channel_init)))
        self.out, self.drop = nn.Linear(dim, dim), nn.Dropout(dropout)
        self.ff_ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))

    def forward(self, h, valid, relation, last=False):
        b, t, _ = h.shape
        z = self.ln(h)
        source = h[:, :1] if last else h
        q = self.q_proj(z[:, :1] if last else z).view(b, -1, self.heads, self.dh)
        k, v = self.k_proj(z), self.v_proj(z)
        dk = sum(self.lambda_k[i] * self.rel_k[i](relation[..., i, :]) for i in range(3))
        dv = sum(self.lambda_v[i] * self.rel_v[i](relation[..., i, :]) for i in range(3))
        kh, vh = k.view(b, t, self.heads, self.dh), v.view(b, t, self.heads, self.dh)
        scores = torch.einsum('bqhd,bthd->bhqt', q, kh) / math.sqrt(self.dh)
        center_scores = torch.einsum('bhd,bthd->bht', q[:, 0], (k + dk).view(b, t, self.heads, self.dh))
        scores = torch.cat([center_scores[:, :, None] / math.sqrt(self.dh), scores[:, :, 1:]], dim=2)
        scores = scores.masked_fill(~valid[:, None, None], -float('inf'))
        weights = self.drop(torch.softmax(scores, -1))
        message = torch.einsum('bhqt,bthd->bqhd', weights, vh)
        center_message = torch.einsum('bht,bthd->bhd', weights[:, :, 0],
                                     (v + dv).view(b, t, self.heads, self.dh))
        message = torch.cat([center_message[:, None], message[:, 1:]], dim=1)
        updated = source + self.drop(self.out(message.reshape(b, -1, self.dim)))
        return updated + self.drop(self.ff(self.ff_ln(updated)))


class InformationReader(nn.Module):
    """Common two-layer full-context GT; relations modify center-row K/V only."""
    def __init__(self, in_dim, classes, dim=64, heads=4, ff_dim=128, dropout=.3, channel_init=.1):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError('hidden dim must be divisible by positive heads')
        self.proj = nn.Linear(in_dim, dim)
        self.witness = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU(), nn.LayerNorm(dim))
        self.strength = nn.Sequential(nn.Linear(2, dim, bias=False), nn.GELU(), nn.LayerNorm(dim))
        self.neighborhood = nn.Sequential(nn.Linear(2 * dim, dim, bias=False), nn.GELU(), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(InformationBlock(dim, heads, ff_dim, dropout, channel_init) for _ in range(2))
        self.final_ln, self.head = nn.LayerNorm(dim), nn.Linear(dim, classes)

    def forward(self, features, valid, witness, strength, neighborhood, present):
        h = self.proj(features)
        background = self.proj(neighborhood)
        relation = torch.stack([self.witness(self.proj(witness)), self.strength(strength),
                                self.neighborhood(torch.cat([background, h - background], -1))], -2)
        relation = relation * present[..., None].to(relation.dtype)
        for layer, block in enumerate(self.blocks):
            h = block(h, valid, relation, last=layer == 1)
        return self.head(self.final_ln(h[:, 0]))


class Coverage:
    def __init__(self):
        self.centers = self.contexts = self.witnesses = self.background_neighbors = 0
        self.positive = np.zeros(3, np.int64)

    def add(self, table):
        live = table.valid.copy()
        live[:, 0] = False
        self.centers += len(live)
        self.contexts += int(live.sum())
        self.positive += table.present[live].sum(0)
        self.witnesses += int(table.witness_count[live].sum())
        self.background_neighbors += int(table.neighborhood_count[live].sum())

    def summary(self):
        return {'actual_contexts': self.contexts / max(self.centers, 1),
                **{f'{name}_pair_share': float(self.positive[i] / max(self.contexts, 1))
                   for i, name in enumerate(CHANNELS)},
                'witnesses_per_pair': self.witnesses / max(self.contexts, 1),
                'background_neighbors_per_pair': self.background_neighbors / max(self.contexts, 1)}
