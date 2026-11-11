import argparse

import torch
from torch.optim import Adam

from nn import GTSNT
from functional import compute_loss, compute_metric
from dataset import load_node_cls_dataset
from utils import (
    load_config, 
    count_model_params,
    seed_everything,
)


def train(data, model, optimizer, loss_func):
    model.train()
    pred = model(data).cpu()
    target = data.y.cpu()
    train_indices = data.train_indices.cpu()
    loss = compute_loss(pred[train_indices], target[train_indices], loss_func)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(data, model, metric_type):
    model.eval()
    pred = model(data).cpu()
    target = data.y.cpu()
    
    accs = []
    for indices in [data.val_indices, data.test_indices]:
        indices = indices.cpu()
        acc = compute_metric(pred[indices], target[indices], metric_type)
        accs.append(acc)
    return accs


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
    data, model = data.to(device), model.to(device)
    
    print(f"Number of Nodes: {data.num_nodes}, Number of Classes: {data.num_classes}, Feature Dimension: {data.num_features} \n"
          f"The number of training, validation and test samples: {data.train_indices.size(0)}, {data.val_indices.size(0)}, {data.test_indices.size(0)}")
    print("Model size (MB): {:.4f}".format(count_model_params(model)))

    best_val_acc, best_test_acc, PATIENCE = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        loss = train(data, model, optimizer, args.loss_func)
        
        (val_acc, test_acc) = evaluate(data, model, args.metric_type)
        if epoch == 1 or (epoch % args.eval_step) == 0:
            print(f"Epoch: {epoch}/{args.epochs} Loss: {loss:.4f}, "
                f"Val ACC: {val_acc:.6f}, Test ACC: {test_acc:.6f}")
        
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
    parser.add_argument("--config_dir", type=str, default="./config/cora.yaml")
    args =  parser.parse_args()
    args = load_config(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)
    main()
