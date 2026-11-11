"""Minimal typed relation vectors: topology, connection and background.

Connections carry calibrated counts/mass and first-order means at each path position.
No second moments, external residuals, effective sample sizes, extra routes or gate descriptors.
"""
import numpy as np
import torch
from torch import nn

from .packed_ops import segment_sum
from .sampler import N_EDGE, TOPOLOGY_DIM

TYPES = ('topology', 'connection', 'background')
MIN_GROUP = 32


def bounded_rms(x):
    return x / x.square().mean(-1, keepdim=True).clamp_min(1.).sqrt()


def fit_statistic_scales(draws, hops):
    """Fit on fixed training centers and uniformly selected contexts, without labels."""
    hop = np.concatenate([d['hop'][d['ctx_valid']] for d in draws])
    count = np.concatenate([d['moment_count'][d['ctx_valid']] for d in draws]).astype(np.float64)
    mass = np.concatenate([d['moment_mass'][d['ctx_valid']] for d in draws]).astype(np.float64)
    known = np.concatenate([d['connection_known'][d['ctx_valid']] for d in draws])
    positive = known & (count > 0)
    log_count, log_mass = np.log1p(count), np.log(np.where(positive, mass, 1.))

    def moments(x):
        if not len(x):
            return 0., 1.
        std = float(x.std())
        return float(x.mean()), std if std > 1e-6 else 1.

    pooled = moments(log_count[known]), moments(log_mass[positive])
    path = np.zeros((hops + 2, 2, 2))
    for distance in range(hops + 2):
        sel = (hop == distance) & known
        path[distance, 0] = moments(log_count[sel]) if sel.sum() >= MIN_GROUP else pooled[0]
        sel &= positive
        path[distance, 1] = moments(log_mass[sel]) if sel.sum() >= MIN_GROUP else pooled[1]
    return {'path': path.tolist(), 'pairs': int(len(hop)), 'pairs_with_paths': int(positive.sum())}


class RelationEncoder(nn.Module):
    def __init__(self, dim, rel_dim, hops, topology_similarity=True, connection_attributes=True, connection_position=True,
                 external_attributes=True):
        super().__init__()
        self.hops, self.positions = hops, max(hops - 1, 1)
        self.topology_similarity = bool(topology_similarity)
        self.connection_attributes = bool(connection_attributes)
        self.connection_position = bool(connection_position)
        self.external_attributes = bool(external_attributes)
        self.hop = nn.Embedding(hops + 2, rel_dim)
        self.edge = nn.Embedding(N_EDGE, rel_dim)
        self.topology_input = nn.Linear(TOPOLOGY_DIM, rel_dim)
        self.topology_norm = nn.LayerNorm(rel_dim)
        width = self.positions * dim + self.positions + 4
        self.connection = nn.Sequential(nn.Linear(width, rel_dim), nn.GELU(),
                                        nn.Linear(rel_dim, rel_dim), nn.LayerNorm(rel_dim))
        self.background = nn.Sequential(nn.Linear(2 * dim, rel_dim, bias=False), nn.GELU(), nn.LayerNorm(rel_dim))
        self.register_buffer('path_scale', torch.tensor([[[0., 1.], [0., 1.]]] * (hops + 2)))

    def set_scales(self, scales):
        self.path_scale.copy_(torch.as_tensor(scales['path'], dtype=torch.float32))

    def _connection(self, pool, batch, pairs):
        positions = self.positions
        segment = batch.moment_owner * positions
        if self.connection_position:
            segment = segment + batch.moment_position - 1
        weight = batch.moment_weight.float()
        total = segment_sum(weight, segment, pairs * positions)
        alpha = weight / total[segment].clamp_min(torch.finfo(torch.float32).tiny)
        attributes = pool[batch.moment_node_pos].float()
        if not self.external_attributes:
            # A per-record mask, not a batch-wide node mask. The denominator
            # above still includes the original weights of external records.
            attributes = attributes * (~batch.moment_external)[:, None]
        mean = segment_sum(attributes * alpha[:, None], segment, pairs * positions)
        mean = mean.reshape(pairs, positions, pool.shape[-1])
        if not self.connection_attributes:
            mean = torch.zeros_like(mean)
        position_mask = (total > 0).reshape(pairs, positions).to(pool.dtype)
        if hasattr(batch, 'connection_positions_used'):
            position_mask = batch.connection_positions_used.to(pool.dtype)
            if not self.connection_position:
                present = position_mask.any(-1)
                position_mask = torch.zeros_like(position_mask)
                position_mask[:, 0] = present.to(pool.dtype)
        scale = self.path_scale[batch.hop]
        known = batch.connection_known
        path_count = getattr(batch, 'connection_count_used', batch.moment_count)
        path_mass = getattr(batch, 'connection_mass_used', batch.moment_mass)
        positive = known & (path_count > 0)
        count = (torch.log1p(path_count) - scale[:, 0, 0]) / scale[:, 0, 1]
        count = torch.where(known, count, torch.zeros_like(count))
        mass = torch.where(positive, (torch.log(path_mass.clamp_min(torch.finfo(torch.float32).tiny))
                                      - scale[:, 1, 0]) / scale[:, 1, 1], torch.zeros_like(count))
        statistics = torch.stack([count, mass, positive.to(pool.dtype), known.to(pool.dtype)], -1)
        inputs = torch.cat([mean.to(pool.dtype).flatten(1), position_mask, statistics], -1)
        return self.connection(inputs)

    def forward(self, pool, batch):
        pairs = len(batch.pair_ctx_pos)
        topology_inputs = batch.topofeat
        if not self.topology_similarity:
            topology_inputs = topology_inputs.clone()
            topology_inputs[:, 2] = 0.  # Jaccard; no degree-role/profile feature exists
        topology = self.topology_norm(self.hop(batch.hop) + self.edge(batch.edge) + self.topology_input(topology_inputs))
        connection = self._connection(pool, batch, pairs)
        # Background records are outside S_c by construction. Retain the context
        # encoding, original count, and availability when their attributes are masked.
        bg_sum = (segment_sum(pool[batch.context_bg_node_pos], batch.context_bg_owner, pairs)
                  if self.external_attributes else pool.new_zeros((pairs, pool.shape[-1])))
        bg_mean = bg_sum / batch.context_bg_count[:, None].clamp_min(1)
        background = self.background(torch.cat([bg_mean, pool[batch.pair_ctx_pos] - bg_mean], -1))
        present = torch.stack([torch.ones(pairs, dtype=torch.bool, device=pool.device),
                               batch.connection_known, batch.context_bg_count > 0], -1)
        vectors = bounded_rms(torch.stack([topology, connection, background], 1)) * present[..., None]
        return vectors, present
