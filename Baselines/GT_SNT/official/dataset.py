import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import(
    Planetoid,
    Amazon,
    Coauthor,
    Actor,
    DeezerEurope,
    WebKB
)
from ogb.nodeproppred import PygNodePropPredDataset

from utils import precalculate_edge_attr


def ratio_rand_splits_(
    data: Data, 
    num_val: float=0.1,
    num_test: float=0.8,
):
    num_nodes, device = data.num_nodes, data.x.device
    idx_shuffled = torch.randperm(num_nodes, device=device)
    
    val_split = int(num_nodes * num_val)
    test_split = int(num_nodes * num_test)
    
    data.test_indices = idx_shuffled[:test_split]
    data.val_indices = idx_shuffled[test_split:test_split+val_split]
    data.train_indices = idx_shuffled[test_split+val_split:]
    

def class_rand_splits_(
    data: Data, 
    label_num_per_class: int=20, 
    valid_num: int=500, 
    test_num: int=1000
):
    label, device = data.y, data.x.device

    idx = torch.arange(label.size(0), device=device)
    class_list = label.unique()
    
    train_idx, non_train_idx = [], []
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]

    data.train_indices = train_idx
    data.val_indices = non_train_idx[:valid_num]
    data.test_indices = non_train_idx[valid_num:valid_num+test_num]


def get_split_indices_(data: Data, dataset_name: str):
    if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        class_rand_splits_(data)
    elif dataset_name in ["CS", "Physics"]:
        ratio_rand_splits_(data)
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        ratio_rand_splits_(data, 0.2, 0.2)
    elif dataset_name == "Deezer":
        ratio_rand_splits_(data, 0.25, 0.25)
    elif dataset_name == "Actor":
        ratio_rand_splits_(data, 0.2, 0.2)


def load_node_cls_dataset(root: str, dataset_name: str, normalization: str=None):   
    if dataset_name.startswith("ogbn"):
        dataset = PygNodePropPredDataset(dataset_name, root)
        data = dataset[0]
        
        if dataset_name == "ogbn-mag":    
            # For ogbn-mag, we are only interested in paper <-> paper relations.
            data = Data(
                x=data.x_dict["paper"],
                edge_index=data.edge_index_dict[("paper", "cites", "paper")],
                y=data.y_dict["paper"],
            )
            # data = T.ToSparseTensor()(data)
            # data.adj_t = data.adj_t.to_symmetric()
            data = T.ToUndirected()(data)
            data.y = data.y.squeeze()
            
            split_idx = dataset.get_idx_split()
            data.train_indices = split_idx["train"]["paper"]
            data.val_indices = split_idx["valid"]["paper"]
            data.test_indices = split_idx["test"]["paper"]
        elif dataset_name in ["ogbn-arxiv", "ogbn-products"]:
            if dataset_name == "ogbn-arxiv":
                if hasattr(data, "adj_t"):
                    data.adj_t = data.adj_t.to_symmetric()
                else:
                    data = T.ToUndirected()(data)
            data.y = data.y.squeeze()
            
            split_idx = dataset.get_idx_split()
            data.train_indices = split_idx["train"]
            data.val_indices = split_idx["valid"]
            data.test_indices = split_idx["test"]
        row, col = data.edge_index
        data.full_graph = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes))
    else:
        if dataset_name in ["Cora", "Pubmed", "Citeseer"]:
            dataset = Planetoid(root, dataset_name)
        elif dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root, dataset_name, transform=T.NormalizeFeatures())
        elif dataset_name in ["CS", "Physics"]:
            dataset = Coauthor(root, dataset_name, transform=T.NormalizeFeatures())
        elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
            dataset = WebKB(osp.join(root, "WebKB"), dataset_name.lower())
        elif dataset_name == "Actor":
            dataset = Actor(osp.join(root, "Actor"))
        elif dataset_name == "Deezer":
            dataset = DeezerEurope(osp.join(root, "Deezer"))
        data = dataset[0]
        
    # Pre-calculate the graph laplacian
    edge_index, edge_weight = precalculate_edge_attr(data.edge_index, data.num_nodes, normalization)
    data.edge_index, data.edge_weight = edge_index, edge_weight

    # generate default splits
    get_split_indices_(data, dataset_name) 

    data.num_classes = dataset.num_classes
    return data