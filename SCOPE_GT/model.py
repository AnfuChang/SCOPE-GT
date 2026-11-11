"""SCOPE-GT center-query reader with typed, dataset-learned K/V relation residuals.

Only center states evolve in the main model. Base node K/V rows are shared; relation
residuals remain specific to (center, context). Every layer/type has independent learned
Key/Value scales; there is no auxiliary pair gate. Context-update is a dense ablation.
"""
import math

import torch
from torch import nn

from .packed_ops import segment_sum, segment_softmax
from .relation import TYPES, RelationEncoder

ISOLATED_INIT = {'relation': 1009}   # offsets of the isolated init streams for added modules


class TokenEncoder(nn.Module):
    """Per-node encoder; nodes do not interact here."""

    def __init__(self, in_dim, dim, layers, ff_mult, dropout, input_dropout=0.0):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout)
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * ff_mult), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim))
            for _ in range(layers))

    def forward(self, x):
        h = self.proj(self.input_drop(x))
        for block in self.blocks:
            h = h + block(h)
        return h


class CenterReadBlock(nn.Module):
    """One center-query attention layer with typed relation residuals on K and V."""

    def __init__(self, dim, heads, ff_dim, dropout, rel_dim, scale_init):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.q_ln, self.q_proj = nn.LayerNorm(dim), nn.Linear(dim, dim)
        self.ctx_ln = nn.LayerNorm(dim)
        self.k_proj, self.v_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ff_ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.rel_k = nn.ModuleList(nn.Linear(rel_dim, dim, bias=False) for _ in TYPES)
        self.rel_v = nn.ModuleList(nn.Linear(rel_dim, dim, bias=False) for _ in TYPES)
        self.lam_k = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        self.lam_v = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def table(self, rows):
        """LN(H) W_K and LN(H) W_V for node rows (independent of the center)."""
        z = self.ctx_ln(rows)
        return self.k_proj(z), self.v_proj(z)

    def deltas(self, rel):
        """rel = (vectors [E, 3, R], coef_k [E, 3], coef_v [E, 3]) with coef = type presence."""
        vectors, coef_k, coef_v = rel
        dk = sum(self.rel_k[t](vectors[:, t]) * (self.lam_k[t] * coef_k[:, t])[:, None] for t in range(len(TYPES)))
        dv = sum(self.rel_v[t](vectors[:, t]) * (self.lam_v[t] * coef_v[:, t])[:, None] for t in range(len(TYPES)))
        return dk, dv

    def finish(self, g, o, has):
        g = g + has[:, None].to(g.dtype) * self.drop(self.out(o))
        return g + self.drop(self.ff(self.ff_ln(g)))

    def read_packed(self, g, keys, values, owner, has=None):
        """g [B, D]; keys/values [E, D] (relations already added); owner [E] center of each row."""
        b, e = len(g), len(owner)
        q = self.q_proj(self.q_ln(g)).view(b, self.heads, self.dh)
        scores = (q[owner] * keys.reshape(e, self.heads, self.dh)).sum(-1) / math.sqrt(self.dh)
        probs = segment_softmax(scores, owner, b)
        o = segment_sum(self.drop(probs)[..., None] * values.reshape(e, self.heads, self.dh), owner, b).reshape(b, self.dim)
        # SCOPEGT supplies a host-known mask: every center has a self-read.
        # Keep the fallback for generic callers that may have empty segments.
        if has is None:
            has = torch.bincount(owner, minlength=b) > 0
        return self.finish(g, o, has)

    def attend_dense(self, x, keys, values, mask):
        """x [B, Q, D] queries; keys/values [B, T, D]; mask [B, T] valid keys. Returns [B, Q, D]."""
        b, nq, t = x.shape[0], x.shape[1], keys.shape[1]
        q = self.q_proj(self.q_ln(x)).view(b, nq, self.heads, self.dh).transpose(1, 2)
        k = keys.view(b, t, self.heads, self.dh).transpose(1, 2)
        v = values.view(b, t, self.heads, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        probs = torch.softmax(scores.masked_fill(~mask[:, None, None, :], -float('inf')), -1)
        return (self.drop(probs) @ v).transpose(1, 2).reshape(b, nq, self.dim)


class SCOPEGT(nn.Module):
    def __init__(self, in_dim, classes, dim=128, heads=4, layers=2, ff_dim=256, dropout=.3, input_dropout=0.,
                 enc_layers=2, enc_ff_mult=2, rel_dim=64, hops=2, relation_scale_init=.05, background_scale_init=.05,
                 topology_similarity=True, connection_attributes=True, connection_position=True,
                 learn_scales=True, relations=True, relation_drop=(), query='center', external_attributes=True):
        super().__init__()
        if min(dim, heads, rel_dim, ff_dim, layers, enc_ff_mult) < 1 or dim % heads or enc_layers < 0:
            raise ValueError('positive widths and layers are required; hidden dim must be divisible by heads')
        if query not in ('center', 'context'):
            raise ValueError("query must be center/context")
        if set(relation_drop) - set(TYPES):
            raise ValueError(f'relation_drop must be a subset of {TYPES}')
        self.relations, self.query = bool(relations), query
        self.drop_types = tuple(TYPES.index(t) for t in relation_drop)
        scale_init = [relation_scale_init, relation_scale_init, background_scale_init]
        self.encoder = TokenEncoder(in_dim, dim, enc_layers, enc_ff_mult, dropout, input_dropout)
        self.blocks = nn.ModuleList(CenterReadBlock(dim, heads, ff_dim, dropout, rel_dim, scale_init)
                                    for _ in range(layers))
        self.self_rel = nn.Parameter(torch.zeros(rel_dim))
        self.final_ln = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, classes)
        base = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):           # added modules: isolated, variant-independent streams
            torch.manual_seed(base + ISOLATED_INIT['relation'])
            self.relation = RelationEncoder(dim, rel_dim, hops, topology_similarity, connection_attributes, connection_position,
                                            external_attributes=external_attributes)
        if not learn_scales:
            for block in self.blocks:
                block.lam_k.requires_grad_(False)
                block.lam_v.requires_grad_(False)

        if not self.relations:
            # Construct once to preserve identical initialization of the shared backbone, then
            # remove every unused relation parameter from the strict no-relation model.
            del self.relation
            del self.self_rel
            for block in self.blocks:
                del block.rel_k
                del block.rel_v
                del block.lam_k
                del block.lam_v

    # ---------------------------------------------------------------- relation inputs
    def _relation_rows(self, pool, batch):
        """Per-pair (vectors, coef_k, coef_v) and the center self rows, or None without relations."""
        if not self.relations:
            return None, None
        vectors, present = self.relation(pool, batch)
        present = present.clone()
        for t in self.drop_types:
            present[:, t] = False
        presentf = present.to(vectors.dtype)
        pairs = (vectors, presentf, presentf)
        self_vectors = vectors.new_zeros((batch.size,) + vectors.shape[1:])
        self_vectors[:, 0] = self.self_rel
        self_coef = vectors.new_zeros((batch.size, len(TYPES)))
        self_coef[:, 0] = 0. if TYPES.index('topology') in self.drop_types else 1.
        return pairs, (self_vectors, self_coef, self_coef)

    @torch.no_grad()
    def relation_diagnostics(self, batch):
        """Context-pair residual/base Frobenius-norm ratios, evaluated without dropout."""
        if not self.relations:
            return {}
        training = self.training
        self.eval()
        try:
            pool = self.encoder(batch.features)
            vectors, present = self.relation(pool, batch)
            for t in self.drop_types:
                present[:, t] = False
            rows = pool[batch.pair_ctx_pos]
            result = {}
            for layer, block in enumerate(self.blocks, 1):
                base_k, base_v = block.table(rows)
                for t, name in enumerate(TYPES):
                    for side, base, projection, scale in (('k', base_k, block.rel_k[t], block.lam_k[t]),
                                                          ('v', base_v, block.rel_v[t], block.lam_v[t])):
                        delta = projection(vectors[:, t]) * scale * present[:, t, None]
                        ratio = (delta.float().square().sum() / base.float().square().sum().clamp_min(1e-30)).sqrt()
                        result[f'relation_{side}_ratio_layer{layer}_{name}'] = float(ratio)
                        result[f'relation_{side}_scale_layer{layer}_{name}'] = float(scale)
            return result
        finally:
            self.train(training)

    def forward(self, batch):
        pool = self.encoder(batch.features)
        pairs, selves = self._relation_rows(pool, batch)
        g = pool[batch.center_pos]
        if batch.packed:
            if self.query != 'center':
                raise ValueError('context updates need a dense batch')
            return self._packed(pool, g, batch, pairs, selves)
        return self._dense(pool, g, batch, pairs, selves)

    def _packed(self, pool, g, batch, pairs, selves):
        rows = pool[batch.kv_rows]
        owner, order, index = batch.read_owner, batch.read_order, batch.read_kv
        rel = None if pairs is None else tuple(torch.cat([a, b])[order] for a, b in zip(pairs, selves))
        for block in self.blocks:
            if getattr(batch, 'kv_per_read', False):
                keys, values = block.table(rows[index])
            else:
                keys, values = block.table(rows)
                keys, values = keys[index], values[index]
            if rel is not None:
                dk, dv = block.deltas(rel)
                keys, values = keys + dk, values + dv
            g = block.read_packed(g, keys, values, owner, has=batch.read_has)
        return self.classifier(self.final_ln(g))

    def _dense(self, pool, g, batch, pairs, selves):
        b, t = batch.ctx_valid.shape
        mask = torch.cat([batch.ctx_valid.new_ones((b, 1)), batch.ctx_valid], 1)
        context_update = self.query == 'context'
        states = pool[batch.ctx_pos] if context_update else None       # per-sequence private context states
        rows = pool[batch.kv_rows]
        for layer, block in enumerate(self.blocks):
            base_k, base_v = block.table(rows)
            self_k, self_v = base_k[batch.center_kv], base_v[batch.center_kv]      # center self K/V: original h_c
            if context_update:
                ctx_k, ctx_v = block.table(states)
            else:
                ctx_k, ctx_v = base_k[batch.ctx_kv], base_v[batch.ctx_kv]
            keys, values = torch.cat([self_k[:, None], ctx_k], 1), torch.cat([self_v[:, None], ctx_v], 1)
            center_k, center_v = keys, values
            if pairs is not None:
                dk, dv = block.deltas(pairs)
                sk, sv = block.deltas(selves)
                delta_k, delta_v = torch.zeros_like(keys), torch.zeros_like(values)
                delta_k[batch.pair_rows, batch.pair_cols + 1] = dk
                delta_v[batch.pair_rows, batch.pair_cols + 1] = dv
                delta_k[:, 0], delta_v[:, 0] = sk, sv
                center_k, center_v = keys + delta_k, values + delta_v
            o = block.attend_dense(g[:, None], center_k, center_v, mask)[:, 0]
            new_g = block.finish(g, o, torch.ones(b, dtype=torch.bool, device=g.device))
            if context_update and layer + 1 < len(self.blocks):         # the last layer's context output is unused
                # Context queries use base K/V; center-specific relation deltas
                # apply only to the center read above, not context-to-context reads.
                o_ctx = block.attend_dense(states, keys, values, mask)
                states = states + block.drop(block.out(o_ctx))
                states = states + block.drop(block.ff(block.ff_ln(states)))
            g = new_g
        return self.classifier(self.final_ln(g))
