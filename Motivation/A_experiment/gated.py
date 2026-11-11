"""Matched Query-set interventions for the A Motivation experiment.

Only context Query membership changes. All tokens execute the same token-wise
FFN, and the center always belongs to the K/V set. The final layer evaluates
only the center Query in every mode: final context outputs have no consumer.

Dropout masks are drawn in the same conceptual full-sequence layout and then
selected by Query index. This prioritizes paired comparisons over speed; this
model does not establish sparse-attention acceleration.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class GatedBlock(nn.Module):
    """Pre-LN attention on selected rows, followed by identical token-wise FFN."""

    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("hidden dimension must be positive and divisible by heads")
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.ln = nn.LayerNorm(dim)
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ff_ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))

    def forward(self, h, valid, gate=None, center_only=False):
        """h [B,T,D], valid [B,T], gate [B,T-1] context Query membership."""
        b, t, _ = h.shape
        if valid.shape != (b, t) or not bool(valid[:, 0].all()):
            raise ValueError("valid must include a valid center at slot zero")
        query_mask = torch.zeros_like(valid)
        query_mask[:, 0] = True
        if not center_only:
            if gate is None or gate.shape != (b, t - 1):
                raise ValueError("context Query membership must have shape [B,T-1]")
            query_mask[:, 1:] = gate.bool() & valid[:, 1:]
        rows, positions = torch.where(query_mask)
        ranks = query_mask.long().cumsum(1)[rows, positions] - 1
        query_ids = torch.zeros((b, int(query_mask.sum(1).max())), device=h.device, dtype=torch.long)
        query_ids[rows, ranks] = positions
        z = self.ln(h)
        query_z = z.gather(1, query_ids[..., None].expand(-1, -1, self.dim))
        q = self.q_proj(query_z).view(b, -1, self.heads, self.dh)
        k = self.k_proj(z).view(b, t, self.heads, self.dh)
        v = self.v_proj(z).view(b, t, self.heads, self.dh)
        scores = torch.einsum("bqhd,bthd->bqht", q, k) / math.sqrt(self.dh)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # Identical RNG consumption across modes, including attention dropout.
        weights = probs
        if self.training and self.drop.p:
            mask = self.drop(torch.ones((b, t, self.heads, t), device=h.device, dtype=h.dtype))
            weights = probs * mask.gather(1, query_ids[..., None, None].expand(-1, -1, self.heads, t))
        attended = torch.einsum("bqht,bthd->bqhd", weights, v).reshape(b, -1, self.dim)
        residual = torch.zeros_like(h)
        residual[rows, positions] = self.out(attended)[rows, ranks]
        h = h + self.drop(residual)
        h = h + self.drop(self.ff(self.ff_ln(h)))
        h = h.masked_fill(~valid[..., None], 0.0)
        center_attention = probs[:, 0].mean(1)
        return h, center_attention


class GatedReader(nn.Module):
    def __init__(self, in_dim, classes, dim=64, heads=4, layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        if layers != 2:
            raise ValueError("A Motivation fixes the depth at two layers")
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(GatedBlock(dim, heads, ff_dim, dropout) for _ in range(layers))
        self.final_ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, center_x, ctx_x, valid, gate, return_attention=False):
        """center_x [b, F], ctx_x [b, K, F], valid [b, 1+K], gate [b, K]."""
        h = self.proj(torch.cat([center_x[:, None], ctx_x], 1))
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            h, attn = block(h, valid, gate, center_only=(i == last))
        logits = self.head(self.final_ln(h[:, 0]))
        return (logits, attn) if return_attention else logits
