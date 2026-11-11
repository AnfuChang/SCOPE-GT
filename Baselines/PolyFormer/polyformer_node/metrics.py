from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_metric(logits: torch.Tensor, labels: torch.Tensor, metric: str, problem_type: str) -> float:
    metric = metric.lower()
    if metric in {"accuracy", "acc"}:
        if problem_type == "binary" and logits.shape[-1] == 1:
            pred = (logits.squeeze(-1) > 0).long()
            y = labels.long()
        elif problem_type == "multilabel":
            pred = (logits > 0).long()
            y = labels.long()
            return float((pred == y).float().mean().item())
        else:
            pred = logits.argmax(dim=-1)
            y = labels.long()
        return float((pred == y).float().mean().item())

    if metric in {"rocauc", "auc", "roc_auc"}:
        y_np = labels.detach().cpu().numpy()
        log_np = logits.detach().cpu().numpy()
        try:
            if problem_type == "binary":
                score = roc_auc_score(y_np.reshape(-1), log_np.reshape(-1))
            elif problem_type == "multiclass":
                # Multiclass ROC-AUC is not standard for most citation datasets;
                # use ovo macro only if explicitly requested.
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                score = roc_auc_score(y_np.reshape(-1), prob, multi_class="ovo", average="macro")
            else:
                score = roc_auc_score(y_np, log_np, average="macro")
        except ValueError:
            # Happens when a split contains a single class. Keep training but mark score invalid.
            score = float("nan")
        return float(score)

    raise ValueError(f"Unknown metric: {metric}")
