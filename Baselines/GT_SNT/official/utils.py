import os
import yaml
import random
import numpy as np
from pathlib import Path

import torch
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import (
    scatter,
    to_edge_index,
    add_remaining_self_loops,
    is_sparse,
    is_torch_sparse_tensor
)
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(args):
    if not hasattr(args, "config_dir") or not os.path.exists(args.config_dir):
        return args

    path = Path(args.config_dir)
    config = yaml.safe_load(open(path, "r"))
    for key, val in config.items():
        if hasattr(config[key], "keys"):
            for key, val in config[key].items():
                args.__dict__[key] = val
    return args


def count_model_params(model):
    """
    Return the size of model in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def precalculate_edge_attr(
    edge_index, 
    num_nodes: int,
    coding_type: str=None,
    add_self_loops: bool=True,
    return_sparse: bool=False,
) -> SparseTensor:
    """
    precalculate edges' attributes
    """
    if is_sparse(edge_index):
        edge_index, _ = to_edge_index(edge_index)
    
    mask = edge_index[0] != edge_index[1]
    if add_self_loops:
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index[:, mask], loop_index], dim=-1)
    else:
        edge_index = edge_index[:, mask]

    row, col = edge_index
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    deg = scatter(edge_weight, col, 0, dim_size=num_nodes, reduce="sum")
    
    if coding_type == "lap":
        edge_weight = edge_weight[:mask.sum().item()]
        edge_weight = torch.cat([-edge_weight, deg - 1 if add_self_loops else deg], dim=0) # L = D - A
    elif coding_type == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] # L = D^(-1/2)AD^(-1/2)
    elif coding_type == "degree":
        edge_weight = deg.pow_(-1.0)[row]
    
    if return_sparse:
        size = (num_nodes, num_nodes)
        adj_t = SparseTensor(
            row=edge_index[1], 
            col=edge_index[0],
            value=edge_weight, 
            sparse_sizes=size, 
            is_sorted=True, 
            trust_data=True
        )
        return adj_t
    else:
        return edge_index, edge_weight