import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

import torch


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


def compute_metric(pred, target, metric_type="accuracy"):
    if isinstance(pred, list):
        pred = torch.cat(pred)
    
    if isinstance(target, list):
        target = torch.cat(target)
    
    if metric_type == "accuracy":
        pred = pred.log_softmax(dim=-1).argmax(dim=-1)
        return (pred == target).sum().item() / len(pred)
    elif metric_type == "accuracy-SBM":
        pred = pred.log_softmax(dim=-1).argmax(dim=-1)
        return accuracy_SBM(target, pred)
    elif metric_type == "mae":
        return (pred.squeeze() - target).abs().mean()
    else:
        raise NotImplementedError
        
