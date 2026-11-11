import torch
import torch.nn.functional as F


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        return loss


def cross_entropy(pred, true):
    # pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    # true = true.squeeze(-1) if true.ndim > 1 else true
    
    # if pred.ndim > 1 and true.ndim == 1:
    pred = F.log_softmax(pred, dim=1)
    return F.nll_loss(pred, true)
    # else:
    #     true = true.float()
    #     return F.binary_cross_entropy_with_logits(pred, true)
    

def compute_loss(pred, target, loss_func=None):
    if loss_func == "cross_entropy":
        return cross_entropy(pred, target)
    elif loss_func == "weighted_cross_entropy":
        return weighted_cross_entropy(pred, target)
    elif loss_func == "mae":
        return (pred.squeeze() - target).abs().mean()
    else:
        raise NotImplementedError