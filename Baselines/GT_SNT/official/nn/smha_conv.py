import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import scatter


class SMultiHeadAttention(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hid_channels: int, 
        cb_channels: int, 
        num_heads: int, 
        dropout: float=0.5, 
        attn_dropout: float=0.5
    ) -> None:
        super().__init__()
        self.hid_channels = hid_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.lin_cb = nn.Linear(cb_channels, hid_channels)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
    
    # VQ attention block
    def vq_attn_block(self, query: Tensor, value: Tensor, codebook: Tensor, coding_indices: Tensor, mask: Tensor=None):
        num_nodes, head_dim = query.size(0), self.hid_channels // self.num_heads
        
        q, v, cb = query, value, self.lin_cb(codebook)
        q = q.view(-1, self.num_heads, head_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, head_dim).transpose(0, 1)
        cb = cb.view(-1, self.num_heads, head_dim).transpose(0, 1)
        q, v, cb = F.normalize(q, dim=-1), F.normalize(v, dim=-1), F.normalize(cb, dim=-1)
        
        v = scatter(v, coding_indices, dim=1, reduce="mean") # H×N×d -> H×C×d
        if mask is not None:
            cb, v = cb[:, mask], v[:, mask]
        
        q_scaled = q / math.sqrt(q.size(2))
        attn_output_weights = torch.bmm(q_scaled, cb.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.attn_dropout, training=self.training)
        
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_nodes, self.hid_channels)
        attn_output = attn_output + value
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        return attn_output
    
    # VQ attention block
    def vq_attn_block_subgraph(
        self, query: Tensor, value: Tensor, codebook: Tensor, 
        coding_indices: Tensor, node_indices: Tensor, mask: Tensor=None):
        num_nodes, head_dim = len(node_indices), self.hid_channels // self.num_heads
        q = query[node_indices]
        
        v = value
        v_res = v[node_indices]
        
        cb = self.lin_cb(codebook)
        v = scatter(v, coding_indices, dim=0, reduce="mean") # H×N×d -> H×C×d
        if mask is not None:
            cb, v = cb[mask], v[mask]
        
        q = q.view(-1, self.num_heads, head_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, head_dim).transpose(0, 1)
        cb = cb.view(-1, self.num_heads, head_dim).transpose(0, 1)
        q, v, cb = F.normalize(q, dim=-1), F.normalize(v, dim=-1), F.normalize(cb, dim=-1)
        
        q_scaled = q / math.sqrt(q.size(2))
        attn_output_weights = torch.bmm(q_scaled, cb.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.attn_dropout, training=self.training)
        
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_nodes, self.hid_channels)
        attn_output = attn_output + v_res
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        return attn_output
    
    def forward(self):
        pass