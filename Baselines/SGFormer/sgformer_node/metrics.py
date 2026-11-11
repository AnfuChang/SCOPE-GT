from __future__ import annotations

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn.functional as F


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def rocauc_from_logits(logits: torch.Tensor, labels: torch.Tensor, problem_type: str = "multiclass") -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ROC-AUC requires scikit-learn. Please install scikit-learn.") from exc

    y_true = labels.detach().cpu().numpy()
    if problem_type == "multilabel":
        y_score = torch.sigmoid(logits).detach().cpu().numpy()
        valid_cols = []
        for c in range(y_true.shape[1]):
            if len(np.unique(y_true[:, c])) == 2:
                valid_cols.append(c)
        if not valid_cols:
            return float("nan")
        return float(roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average="macro"))

    if logits.shape[-1] == 2:
        y_score = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        return float(roc_auc_score(y_true, y_score))
    y_score = F.softmax(logits, dim=-1).detach().cpu().numpy()
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def compute_metric(logits: torch.Tensor, labels: torch.Tensor, metric: str, problem_type: str) -> float:
    metric = metric.lower()
    if metric in {"acc", "accuracy"}:
        if problem_type == "multilabel":
            pred = (torch.sigmoid(logits) > 0.5).to(labels.dtype)
            return float((pred == labels).float().mean().item())
        return accuracy_from_logits(logits, labels)
    if metric in {"auc", "rocauc", "roc_auc"}:
        return rocauc_from_logits(logits, labels, problem_type=problem_type)
    raise ValueError(f"Unsupported metric: {metric}")
