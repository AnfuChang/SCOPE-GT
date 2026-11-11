"""GOAT model (official/large_model.py) without torch_sparse, PyG MessagePassing and einops.

Same modules, parameter shapes, initialisation distributions and forward computation as official/large_model.py
(upstream commit in UPSTREAM.txt); test_goat.py checks that both give identical outputs and gradients for the
same weights. Differences are implementation only:
  local attention   PyG propagate + torch_geometric.utils.softmax -> gather / grouped softmax / index_add_
  rearrange         einops -> view + permute
  empty centroids   log(0) = -inf -> -1e9 (both give attention weight exactly 0)
  c_idx             int16 -> int64 buffer (same values; num_centroids <= 4096)
  lin_skip          official calls lin_skip(x[1]) (only reached with --skip 1, which no official command uses);
                    here lin_skip(x)
The codebook (official/vq.py) is imported unchanged.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _load_official_vq():
    path = Path(__file__).resolve().parent / "official" / "vq.py"
    spec = importlib.util.spec_from_file_location("goat_official_vq", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VectorQuantizerEMA = _load_official_vq().VectorQuantizerEMA


def _split_heads(t: Tensor, heads: int, head_dim: int) -> Tensor:
    """Native replacement for einops: [N, H*D] -> [H, N, D]."""
    return t.view(t.size(0), heads, head_dim).permute(1, 0, 2).contiguous()


def _merge_heads(t: Tensor) -> Tensor:
    """Native replacement for einops: [H, N, D] -> [N, H*D]."""
    return t.permute(1, 0, 2).contiguous().view(t.size(1), t.size(0) * t.size(2))


class TransformerConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int,
        num_nodes: int,
        spatial_size: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        skip: bool = True,
        dist_count_norm: bool = True,
        conv_type: str = "local",
        num_centroids: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and skip
        self.skip = skip
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.spatial_size = spatial_size
        self.dist_count_norm = dist_count_norm
        self.conv_type = conv_type
        self.num_centroids = int(num_centroids) if num_centroids is not None else None

        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)

        if concat:
            self.lin_skip = nn.Linear(in_channels, heads * out_channels, bias=bias)
            self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False) if self.beta else None
        else:
            self.lin_skip = nn.Linear(in_channels, out_channels, bias=bias)
            self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False) if self.beta else None

        # distance ids are 0..L. spatial_size is L, so allocate L+1 embeddings.
        self.spatial_encoder = nn.Embedding(spatial_size + 1, heads)

        if self.conv_type != "local":
            if self.num_centroids is None or self.num_centroids <= 0:
                raise ValueError("num_centroids must be positive for GOAT global/full attention.")
            self.vq = VectorQuantizerEMA(self.num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, self.num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax
            self.lin_proj_g = nn.Linear(in_channels, global_dim, bias=bias)
            self.lin_key_g = nn.Linear(global_dim * 2, heads * out_channels, bias=bias)
            self.lin_query_g = nn.Linear(global_dim * 2, heads * out_channels, bias=bias)
            self.lin_value_g = nn.Linear(global_dim, heads * out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for mod in [self.lin_key, self.lin_query, self.lin_value, self.lin_skip]:
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()
        torch.nn.init.zeros_(self.spatial_encoder.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, pos_enc=None, batch_idx=None):
        batch_size = len(batch_idx) if batch_idx is not None else x.size(0)
        if self.conv_type == "local":
            out = self.local_forward(x, edge_index, edge_attr)[:batch_size]
        elif self.conv_type == "global":
            out = self.global_forward(x[:batch_size], pos_enc, batch_idx)
        elif self.conv_type == "full":
            out_local = self.local_forward(x, edge_index, edge_attr)[:batch_size]
            out_global = self.global_forward(x[:batch_size], pos_enc, batch_idx)
            out = torch.cat([out_local, out_global], dim=1)
        else:
            raise NotImplementedError(f"Unknown conv_type={self.conv_type}")
        return out

    def global_forward(self, x: Tensor, pos_enc: Tensor, batch_idx):
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = torch.cat([self.lin_proj_g(x), pos_enc], dim=1)
        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q = _split_heads(q, h, d)
        k = _split_heads(k, h, d)
        v = _split_heads(v, h, d)
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        # Corollary 3.4 in GOAT adds log cluster sizes. Empty centroids must not
        # receive mass, so they get a large negative bias instead of log(0).
        c, c_count = self.c_idx.unique(return_counts=True)
        centroid_count = torch.zeros(self.num_centroids, dtype=torch.float32, device=x.device)
        centroid_count[c.to(device=x.device, dtype=torch.long)] = c_count.to(device=x.device, dtype=torch.float32)
        log_count = torch.full_like(centroid_count, -1e9)
        non_empty = centroid_count > 0
        log_count[non_empty] = torch.log(centroid_count[non_empty])
        dots = dots + log_count.view(1, 1, -1)

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = _merge_heads(out)

        if self.training:
            x_idx = self.vq.update(q_x)
            if not torch.is_tensor(batch_idx):
                batch_idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.c_idx.device)
            else:
                batch_idx = batch_idx.to(device=self.c_idx.device, dtype=torch.long)
            self.c_idx[batch_idx] = x_idx.squeeze().to(device=self.c_idx.device, dtype=torch.long)
        return out

    @staticmethod
    def _grouped_softmax(src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Softmax over edges sharing the same target index, pure PyTorch.

        src: [E, H], index: [E]. Returns [E, H].
        """
        if src.numel() == 0:
            return src
        E, H = src.shape
        expanded_index = index.view(-1, 1).expand(-1, H)
        if hasattr(torch.Tensor, "scatter_reduce_"):
            max_per = torch.full((num_nodes, H), -torch.inf, device=src.device, dtype=src.dtype)
            max_per.scatter_reduce_(0, expanded_index, src, reduce="amax", include_self=True)
            out = torch.exp(src - max_per[index])
            denom = torch.zeros((num_nodes, H), device=src.device, dtype=src.dtype)
            denom.scatter_add_(0, expanded_index, out)
            return out / (denom[index] + 1e-16)

        # Compatibility fallback for older PyTorch: loop only over batch targets.
        out = torch.empty_like(src)
        for t in torch.unique(index):
            mask = index == t
            out[mask] = torch.softmax(src[mask], dim=0)
        return out

    def local_forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        H, C = self.heads, self.out_channels
        if edge_index.numel() == 0:
            out = torch.zeros(x.size(0), H, C, device=x.device, dtype=x.dtype)
        else:
            edge_index = edge_index.to(device=x.device, dtype=torch.long)
            source = edge_index[0]
            target = edge_index[1]

            query = self.lin_query(x).view(-1, H, C)
            key = self.lin_key(x).view(-1, H, C)
            value = self.lin_value(x).view(-1, H, C)

            alpha = (query[target] * key[source]).sum(dim=-1) / math.sqrt(C)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device=x.device, dtype=torch.long)
                edge_dist = edge_attr[0].clamp(min=0, max=self.spatial_size)
                edge_dist_count = edge_attr[1].clamp_min(1).to(dtype=x.dtype)
                alpha = alpha + self.spatial_encoder(edge_dist)
                if self.dist_count_norm:
                    alpha = alpha - torch.log(edge_dist_count).unsqueeze(1)

            alpha = self._grouped_softmax(alpha, target, x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            msg = value[source] * alpha.view(-1, H, 1)
            out = torch.zeros(x.size(0), H, C, device=x.device, dtype=x.dtype)
            out.index_add_(0, target, msg)

        if self.concat:
            out = out.reshape(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.skip:
            x_r = self.lin_skip(x)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1)).sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


class Transformer(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, global_dim, num_layers, heads, ff_dropout, attn_dropout, spatial_size, skip, dist_count_norm, conv_type, num_centroids, no_bn, norm_type):
        super().__init__()
        if norm_type == "batch_norm":
            norm_func = nn.BatchNorm1d
        elif norm_type == "layer_norm":
            norm_func = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_type={norm_type}")

        if no_bn:
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
        else:
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.convs = nn.ModuleList()
        self.ffs = nn.ModuleList()
        if num_layers != 1:
            raise ValueError("Official GOAT uses one attention layer; keep num_layers=1.")

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    spatial_size=spatial_size,
                    heads=heads,
                    dropout=attn_dropout,
                    skip=skip,
                    dist_count_norm=dist_count_norm,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                )
            )
            h_times = 2 if conv_type == "full" else 1
            if no_bn:
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times * hidden_channels * heads, hidden_channels * heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels * heads, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )
            else:
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times * hidden_channels * heads, hidden_channels * heads),
                        norm_func(hidden_channels * heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels * heads, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )

        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, pos_enc, batch_idx):
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, pos_enc, batch_idx)
            x = self.ffs[i](x)
        return self.fc_out(x)

    def global_forward(self, x, pos_enc, batch_idx):
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv.global_forward(x, pos_enc, batch_idx)
            x = self.ffs[i](x)
        return self.fc_out(x)
