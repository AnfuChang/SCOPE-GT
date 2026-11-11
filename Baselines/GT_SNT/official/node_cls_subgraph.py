import argparse
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor

from nn import GTSNT
from functional import compute_loss, compute_metric
from dataset import load_node_cls_dataset
from utils import (
    load_config, 
    count_model_params,
    seed_everything,
)


def train(data, loader, model, optimizer, loss_func):
    model.train()
    total_loss = 0
    for batch_indices in tqdm(loader):
        _, edge_index, _, _ = k_hop_subgraph(batch_indices, args.num_layers, data.edge_index, num_nodes=data.num_nodes, directed=True)
        sub_graph = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        
        pred = model.forward_subgraph(data.x, sub_graph.t(), node_indices=batch_indices, adj_t=data.full_graph.t())
        loss = compute_loss(pred, data.y[batch_indices], loss_func)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(data, loader, model, metric_type):
    model.eval()
    preds, targets = [], []
    for batch_indices in loader:
        _, edge_index, _, _ = k_hop_subgraph(batch_indices, args.num_layers, data.edge_index, num_nodes=data.num_nodes, directed=True)
        sub_graph = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))

        preds.append(model.forward_subgraph(data.x, sub_graph.t(), node_indices=batch_indices, adj_t=data.full_graph.t()).cpu())
        targets.append(data.y[batch_indices].cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = compute_metric(preds, targets, metric_type)
    return acc

def main():
    # Load dataset
    data = load_node_cls_dataset(args.data_dir, args.name, args.coding_type)

    # Initialize the model, optimizer and criteria
    model = GTSNT(
        data.num_nodes, data.num_features, data.num_classes,
        args.hid_channels, args.cb_channels, args.num_layers, args.num_heads,
        args.T, args.neuron, args.v_threshold,
        args.dropout, args.attn_dropout, maximum_codes_num=args.maximum_codes_num,
        normalize=args.normalize, enable_residual=args.enable_residual, enable_norm=args.enable_norm
    )
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    model, data = model.to(device), data.to(device)

    print(f"Number of Nodes: {data.num_nodes}, Number of Edges: {data.num_edges}, "
          f"Number of Classes: {data.num_classes}, Feature Dimension: {data.num_features} \n"
          f"The number of training, validation and test samples: {data.train_indices.size(0)}, {data.val_indices.size(0)}, {data.test_indices.size(0)}")
    print("Model size (MB): {:.4f}".format(count_model_params(model)))
        
    train_loader = DataLoader(data.train_indices, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data.val_indices, batch_size=args.batch_size)
    test_loader = DataLoader(data.test_indices, batch_size=args.batch_size)
    print("Create train, val and test dataloader...Done!")

    model.reset_parameters()
    best_val_acc, best_test_acc, PATIENCE = 0, 0, 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train(data, train_loader, model, optimizer, args.loss_func)
        
        val_acc = evaluate(data, val_loader, model, args.metric_type)
        test_acc = evaluate(data, test_loader, model, args.metric_type)
        if epoch == 1 or (epoch % args.eval_step) == 0:
            print(f"Epoch: {epoch}/{args.epochs} Loss: {loss:.4f}, "
                f"Val ACC: {val_acc:.6f}, Test ACC: {test_acc:.6f}, ")
        
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                PATIENCE = 0
            else:
                PATIENCE += 1
                if PATIENCE > args.patience:
                    break
        
    print(f"Best Val ACC: {best_val_acc:.6f}, Test ACC: {best_test_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str)
    args =  parser.parse_args()
    args = load_config(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_everything(args.seed)
    
    main()
