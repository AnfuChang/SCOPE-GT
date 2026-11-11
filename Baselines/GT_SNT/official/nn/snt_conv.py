import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import (
    spmm, 
    scatter,
)

from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
    MultiStepIFNode,
)


class SNT(MessagePassing):
    def __init__(
        self, num_nodes: int, cb_channels: int, T: int, 
        neuron: str="PLIF", v_threshold: float=1., init_beta: float=0.2, 
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.T = T
        self.init_beta = init_beta
        self.beta = nn.Parameter(torch.tensor([self.init_beta], dtype=torch.float))
        self.rand_feat = nn.Parameter(torch.rand((num_nodes, cb_channels), dtype=torch.float))
        
        if neuron == "IF":
            self.neuron = MultiStepIFNode(v_threshold=v_threshold)
        elif neuron == "LIF":
            self.neuron = MultiStepLIFNode(v_threshold=v_threshold)
        elif neuron == "PLIF":
            self.neuron = MultiStepParametricLIFNode(v_threshold=v_threshold)
        else:
            raise NotImplementedError

        self.reset_parameters()

    def neuron_reset(self):
        self.neuron.reset()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.uniform_(self.rand_feat)
        nn.init.constant_(self.beta, self.init_beta)
        self.neuron_reset() # reset membrane potential

    def forward(self, edge_index: Adj, edge_weight: Tensor=None):
        embds = [self.rand_feat]
        for _ in range(len(embds), self.T):
            embd = embds[-1]
            embd = self.propagate(edge_index, x=embd, edge_weight=edge_weight)
            embds.append(embd)
        
        # Utilize spiking neurons sample the structural/positional patterns from the graph
        embds = torch.stack(embds, dim=0)
        spikes_out = self.neuron(embds).sum(0) # T×N×D -> N×D
        
        _, codes_indices, counts = torch.unique(spikes_out.detach(), return_inverse=True, return_counts=True, dim=0)
        codebook = scatter(spikes_out, codes_indices, dim=0, reduce="mean")
        return codebook, codes_indices, counts

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)