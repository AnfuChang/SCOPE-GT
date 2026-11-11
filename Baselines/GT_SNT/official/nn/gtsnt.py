from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.mpnn_conv import GCNConv
from nn.smha_conv import SMultiHeadAttention
from nn.snt_conv import SNT


class GTSNT(nn.Module):
    _cached_cb = []
    _cached_code_indices = []
    
    def __init__(
        self, num_nodes: int, in_channels: int, out_channels: int, 
        hid_channels: int, cb_channels: int, num_layers: int, num_heads: int, 
        T: int, neuron: str, v_threshold: float=1.0, 
        dropout: float=0.5, attn_dropout: float=0.5, 
        maximum_codes_num: int=None, normalize: bool=False, 
        enable_residual: bool=True, enable_norm: bool=False
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        self.maximum_codes_num = maximum_codes_num
        self.enable_norm = enable_norm
        self.enable_residual = enable_residual
        
        channels = [in_channels] + [hid_channels] * num_layers
        self.trans_convs, self.cb_convs = nn.ModuleList(), nn.ModuleList()
        self.mpnn_convs, self.norm_convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(channels) - 1):    
            # GT with Learnable Codebook Blocks
            self.cb_convs.append(SNT(num_nodes, cb_channels, T, neuron, v_threshold))
            self.trans_convs.append(SMultiHeadAttention(channels[i], channels[i + 1], 
                cb_channels, num_heads, dropout, attn_dropout))

            self.mpnn_convs.append(GCNConv(channels[i], channels[i + 1], normalize=normalize))
            self.norm_convs.append(nn.BatchNorm1d(channels[i + 1]))
        
        self.lin_in = nn.Linear(in_channels, hid_channels)
        self.lin_out = nn.Linear(hid_channels, out_channels)
        self.reset_parameters()
    
    # Reset the membrane potential of spiking neurons for each step
    def reset_membrane_potential(self):
        for conv in self.cb_convs:
            conv.neuron_reset()
    
    def reset_parameters(self):
        for i in range(len(self.trans_convs)):
            self.trans_convs[i].reset_parameters()
            self.mpnn_convs[i].reset_parameters()
            self.cb_convs[i].reset_parameters()
            self.norm_convs[i].reset_parameters()
        
        self.lin_out.reset_parameters()
        self.lin_in.reset_parameters()
        self._cached_cb, self._cached_code_indices = [], []
        self.reset_membrane_potential()
    
    def get_cb_size(self) -> List:
        return [indices.max().item() + 1 for (indices, _) in self._cached_code_indices]
    
    def forward(self, data):
        embd, edge_weight = data.x, data.edge_weight
        adj = data.adj_t if hasattr(data, "adj_t") else data.edge_index
        
        self._cached_cb, self._cached_code_indices = [], []
        for i in range(self.num_layers):
            embd_res = self.lin_in(embd) if i == 0 else embd

            # MPNN Conv
            embd_mpnn = self.mpnn_convs[i](embd, adj, edge_weight).relu()
            embd_mpnn = F.dropout(embd_mpnn, self.dropout, training=self.training)

            # SVQ Conv
            codebook, code_indices, counts = self.cb_convs[i](adj, edge_weight)

            # Transformer Conv
            if self.maximum_codes_num is not None and codebook.size(0) > self.maximum_codes_num:
                sorted_counts_indices = torch.argsort(counts, descending=True)
                mask = sorted_counts_indices[:self.maximum_codes_num]
            else:
                mask = None
            embd_trans = self.trans_convs[i].vq_attn_block(embd_mpnn, embd_mpnn, codebook, code_indices, mask)
            embd = embd_trans + embd_mpnn

            if self.enable_residual:
                embd = embd + embd_res
                
            if self.enable_norm:
                embd = self.norm_convs[i](embd)

            self._cached_cb.append(codebook)
            self._cached_code_indices.append((code_indices, counts))
        
        out = self.lin_out(embd)
        self.reset_membrane_potential()
        return out
    
    def forward_subgraph(self, x, edge_index, edge_weight=None, node_indices=None, adj_t=None):
        embd = x
        
        self._cached_cb, self._cached_code_indices = [], []
        for i in range(self.num_layers):
            embd_res = self.lin_in(embd) if i == 0 else embd
            
            # MPNN Conv
            embd_mpnn = self.mpnn_convs[i](embd, edge_index, edge_weight).relu()
            embd_mpnn = F.dropout(embd_mpnn, self.dropout, training=self.training)

            # SVQ Conv
            codebook, code_indices, counts = self.cb_convs[i](adj_t)
            
            # Transformer Conv
            if self.maximum_codes_num is not None and codebook.size(0) > self.maximum_codes_num:
                sorted_counts_indices = torch.argsort(counts, descending=True)
                mask = sorted_counts_indices[:self.maximum_codes_num]
            else:
                mask = None
            embd_trans = self.trans_convs[i].vq_attn_block_subgraph(embd_mpnn, embd_mpnn, codebook, code_indices, node_indices, mask)
            embd = embd_mpnn.index_add_(0, node_indices, embd_trans)
            
            if self.enable_residual:
                embd = embd + embd_res
                
            if self.enable_norm:
                embd = self.norm_convs[i](embd)
            
            self._cached_cb.append(codebook)
            self._cached_code_indices.append((code_indices, counts))
        
        out = self.lin_out(embd)[node_indices]
        self.reset_membrane_potential()
        return out