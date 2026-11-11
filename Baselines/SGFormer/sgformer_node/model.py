from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.norm(x)


class SimpleGlobalAttention(nn.Module):
    """SGFormer linear all-pair attention.

    The denominator follows the v5 paper form after rewriting
      (V + Q(K^T V)/N) / (1 + Q(K^T 1)/N)
    as
      (N V + Q(K^T V)) / (N + Q(K^T 1)).
    This keeps the intended normalizer explicit and avoids accidentally using
    earlier denominator variants. Q and K use Frobenius normalization.
    """

    def __init__(self, hidden_dim: int, beta: float = 0.5, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.beta = beta
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        n = z0.size(0)
        q = self.q(z0)
        k = self.k(z0)
        v = self.v(z0)

        q = q / (torch.norm(q, p="fro") + self.eps)
        k = k / (torch.norm(k, p="fro") + self.eps)

        k_sum = k.sum(dim=0)  # [d] = K^T 1
        denom = float(max(n, 1)) + (q * k_sum).sum(dim=-1, keepdim=True)
        denom = denom.clamp_min(self.eps)

        kv = torch.matmul(k.transpose(0, 1), v)  # [d, d] = K^T V
        z_attn = (float(max(n, 1)) * v + torch.matmul(q, kv)) / denom
        z_attn = self.dropout(z_attn)
        out = self.beta * z_attn + (1.0 - self.beta) * z0
        return self.out_norm(out)


class GCNBranch(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.lins = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, z0: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = z0
        for i, lin in enumerate(self.lins):
            h = torch.sparse.mm(adj_norm, h)
            h = lin(h)
            if i != len(self.lins) - 1:
                h = F.relu(h)
                h = self.dropout(h)
            h = self.norms[i](h)
        return h


class SGFormerNode(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gcn_layers: int = 2,
        trans_layers: int = 2,
        dropout: float = 0.3,
        gnn_dropout: float | None = None,
    ):
        """dropout: Transformer side (input MLP, attention, fused output), the official
        ours_dropout / trans_dropout; gnn_dropout: GCN branch (official dropout / gnn_dropout),
        defaults to dropout."""
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        if trans_layers < 1:
            raise ValueError("trans_layers must be >= 1")
        self.input = InputMLP(in_dim, hidden_dim, dropout=dropout)
        self.attn_layers = nn.ModuleList(
            [SimpleGlobalAttention(hidden_dim, beta=beta, dropout=dropout) for _ in range(trans_layers)]
        )
        self.gcn = GCNBranch(hidden_dim, num_layers=gcn_layers, dropout=dropout if gnn_dropout is None else gnn_dropout)
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        z0 = self.input(x)
        z_attn = z0
        for attn in self.attn_layers:
            z_attn = attn(z_attn)
        z_gnn = self.gcn(z0, adj_norm)
        z = (1.0 - self.alpha) * z_attn + self.alpha * z_gnn
        z = self.fuse_norm(z)
        z = self.dropout(z)
        return self.out(z)


# ---------------------------------------------------------------------------------------------
# Model of the official large-graph code (github.com/qitianwu/SGFormer, large/ours.py), used by
# large/run.sh for ogbn-arxiv, amazon2m (ogbn-products graph) and pokec. It differs from
# SGFormerNode above: the GCN branch has its own input layer, ReLU after every layer and an
# initial residual (every layer adds the input-layer output), the two branches are mixed without
# an extra LayerNorm/dropout, and the attention residual is a plain average.
# adj_norm is D^-1/2 (A + I) D^-1/2, the normalisation GraphConvLayer computes from the
# self-looped edge_index of main.py / main-batch.py.
# ---------------------------------------------------------------------------------------------


class OfficialGraphConv(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, *, use_bn: bool = True,
                 use_residual: bool = True, use_weight: bool = True, use_init: bool = False, use_act: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers + 1)])
        # GraphConvLayer.W: always created; input [A x, x0] when use_init
        self.lins = nn.ModuleList([nn.Linear(2 * hidden_dim if use_init else hidden_dim, hidden_dim)
                                   for _ in range(num_layers)])
        self.dropout = dropout
        self.use_bn, self.use_residual, self.use_weight = use_bn, use_residual, use_weight
        self.use_init, self.use_act = use_init, use_act

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x0 = x  # layer_[0]: the official list is appended once, so the residual always adds x0
        for i, lin in enumerate(self.lins):
            h = torch.sparse.mm(adj_norm, x)
            if self.use_init:
                h = lin(torch.cat([h, x0], dim=1))
            elif self.use_weight:
                h = lin(h)
            if self.use_bn:
                h = self.bns[i + 1](h)
            if self.use_act:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_residual:
                h = h + x0
            x = h
        return x


class OfficialAttention(nn.Module):
    """TransConvLayer: Q and K divided by their norm over all nodes and heads, (N V + Q(K^T V)) / (N + Q K^T 1),
    mean over heads."""

    def __init__(self, hidden_dim: int, num_heads: int = 1, use_weight: bool = True):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads) if use_weight else None
        self.hidden_dim, self.num_heads = hidden_dim, num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, h, d = x.size(0), self.num_heads, self.hidden_dim
        q = self.q(x).reshape(n, h, d)
        k = self.k(x).reshape(n, h, d)
        v = self.v(x).reshape(n, h, d) if self.v is not None else x.reshape(n, 1, d)
        q = q / torch.norm(q, p=2)
        k = k / torch.norm(k, p=2)
        num = torch.einsum("nhm,hmd->nhd", q, torch.einsum("lhm,lhd->hmd", k, v)) + n * v
        den = torch.einsum("nhm,hm->nh", q, k.sum(dim=0)).unsqueeze(-1) + n
        return (num / den).mean(dim=1)


class OfficialTransConv(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, *,
                 use_bn: bool = True, use_residual: bool = True, use_weight: bool = True, use_act: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers + 1)])
        self.layers = nn.ModuleList([OfficialAttention(hidden_dim, num_heads, use_weight) for _ in range(num_layers)])
        self.dropout = dropout
        self.use_bn, self.use_residual, self.use_act = use_bn, use_residual, use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.use_bn:
            x = self.norms[0](x)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        prev = [x]
        for i, attn in enumerate(self.layers):
            x = attn(x)
            if self.use_residual:
                x = (x + prev[i]) / 2.0
            if self.use_bn:
                x = self.norms[i + 1](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            prev.append(x)
        return x


class SGFormerOfficialLarge(nn.Module):
    """large/ours.py SGFormer with aggregate='add': fc(graph_weight * GCN(x) + (1 - graph_weight) * Trans(x))."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, graph_weight: float = 0.5,
                 trans_layers: int = 1, trans_heads: int = 1, trans_dropout: float = 0.5, trans_use_bn: bool = True,
                 trans_use_residual: bool = True, trans_use_weight: bool = True, trans_use_act: bool = False,
                 gcn_layers: int = 3, gnn_dropout: float = 0.5, gnn_use_bn: bool = True, gnn_use_residual: bool = True,
                 gnn_use_weight: bool = True, gnn_use_init: bool = False, gnn_use_act: bool = True):
        super().__init__()
        if not (0.0 <= graph_weight <= 1.0):
            raise ValueError("graph_weight must be in [0, 1]")
        self.graph_weight = graph_weight
        self.trans_conv = OfficialTransConv(in_dim, hidden_dim, trans_layers, trans_heads, trans_dropout,
                                            use_bn=trans_use_bn, use_residual=trans_use_residual,
                                            use_weight=trans_use_weight, use_act=trans_use_act)
        self.graph_conv = OfficialGraphConv(in_dim, hidden_dim, gcn_layers, gnn_dropout, use_bn=gnn_use_bn,
                                            use_residual=gnn_use_residual, use_weight=gnn_use_weight,
                                            use_init=gnn_use_init, use_act=gnn_use_act)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def param_groups(self):
        """Official optimiser groups: params1 = Transformer (trans_weight_decay), params2 = GCN + fc (gnn_weight_decay)."""
        return list(self.trans_conv.parameters()), list(self.graph_conv.parameters()) + list(self.fc.parameters())

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        z = self.graph_weight * self.graph_conv(x, adj_norm) + (1.0 - self.graph_weight) * self.trans_conv(x)
        return self.fc(z)
