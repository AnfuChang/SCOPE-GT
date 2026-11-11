from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyAttn(nn.Module):
    """Official PolyAttn block adapted from the released PolyFormer code."""

    def __init__(self, hidden: int, K: int, n_head: int, q: float, multi: float, dprate: float):
        super().__init__()
        if hidden % n_head != 0:
            raise ValueError(f"hidden={hidden} must be divisible by n_head={n_head}")
        self.K = K + 1
        self.hidden = hidden
        self.n_head = n_head
        self.d_head = hidden // n_head
        self.multi = multi
        inner = max(1, int(hidden * multi))

        self.norm = nn.LayerNorm(hidden)
        self.token_wise_network = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, inner), nn.ReLU(), nn.Linear(inner, hidden))
            for _ in range(self.K)
        ])
        self.W_Q = nn.Linear(hidden, n_head * self.d_head, bias=False)
        self.W_K = nn.Linear(hidden, n_head * self.d_head, bias=False)
        self.bias_scale = nn.Parameter(torch.ones(n_head, self.K))
        self.register_buffer("bias", torch.tensor([((j + 1) ** q) ** (-1) for j in range(self.K)], dtype=torch.float32))
        self.dprate = float(dprate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.token_wise_network:
            layer[0].reset_parameters()
            layer[2].reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, K+1, hidden]
        batch_size = src.shape[0]
        origin_src = src
        src = self.norm(src)
        token = torch.stack(
            [layer(src[:, idx, :]) for idx, layer in enumerate(self.token_wise_network)],
            dim=1,
        )
        value = src
        query = self.W_Q(token)
        key = self.W_K(token)
        q_heads = query.view(batch_size, self.K, self.n_head, self.d_head).transpose(1, 2)
        k_heads = key.view(batch_size, self.K, self.n_head, self.d_head).transpose(1, 2)
        v_heads = value.view(batch_size, self.K, self.n_head, self.d_head).transpose(1, 2)
        attention_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_scores = torch.tanh(attention_scores)
        attn_mask = torch.einsum("hk,k->hk", self.bias_scale, self.bias)
        attention_scores = torch.einsum("bhij,hj->bhij", attention_scores, attn_mask)
        attention_scores = F.dropout(attention_scores, p=self.dprate, training=self.training)
        context_heads = torch.matmul(attention_scores, v_heads)
        context_sequence = context_heads.transpose(1, 2).contiguous().view(batch_size, self.K, self.hidden)
        out = F.dropout(context_sequence, p=self.dprate, training=self.training)
        return out + origin_src


class FFNNetwork(nn.Module):
    def __init__(self, hidden: int, d_ffn: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden, d_ffn)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(d_ffn, hidden)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.gelu(self.lin1(x)))


class FFN(nn.Module):
    def __init__(self, hidden: int, d_ffn: int, dprate: float):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn_net = FFNNetwork(hidden, d_ffn)
        self.dropout = float(dprate)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        origin_src = src
        src = self.ffn_norm(src)
        src = self.ffn_net(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        return src + origin_src


class PolyFormerBlock(nn.Module):
    def __init__(self, hidden: int, K: int, n_head: int, q: float, multi: float, d_ffn: int, dprate: float):
        super().__init__()
        self.attnmodule = PolyAttn(hidden, K, n_head, q, multi, dprate)
        self.ffnmodule = FFN(hidden, d_ffn, dprate)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.attnmodule(src)
        src = self.ffnmodule(src)
        return src


class PolyFormer(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden: int,
        out_dim: int,
        *,
        K: int,
        nlayer: int,
        n_head: int,
        d_ffn: int,
        q: float,
        multi: float,
        dropout: float,
        dprate: float,
    ):
        super().__init__()
        self.K = K + 1
        self.dropout = float(dropout)
        self.lin1 = nn.Linear(num_features, hidden)
        self.attn = nn.ModuleList([
            PolyFormerBlock(hidden, K, n_head, q, multi, d_ffn, dprate)
            for _ in range(nlayer)
        ])
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, poly_tokens: torch.Tensor) -> torch.Tensor:
        # poly_tokens: [B, K+1, F]
        x = self.lin1(poly_tokens)
        for block in self.attn:
            x = block(x)
        x = torch.sum(x, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin3(x)
