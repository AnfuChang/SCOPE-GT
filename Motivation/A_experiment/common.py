"""Shared training loop: AdamW + cosine learning rate + validation-accuracy early stopping; restores the best validation parameters."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fit(model, train_batches, val_accuracy, epochs, patience, lr=1e-3, weight_decay=1e-4, log=None):
    """train_batches(epoch) yields (model input tuple, labels); val_accuracy() returns the validation accuracy."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, bad = {"acc": -1.0, "epoch": 0, "state": None}, 0
    for epoch in range(epochs):
        model.train()
        for inputs, y in train_batches(epoch):
            loss = F.cross_entropy(model(*inputs), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        acc = val_accuracy()
        if acc > best["acc"] + 1e-6:
            best = {"acc": acc, "epoch": epoch + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if log:
            print(f"    epoch {epoch + 1:3d}  val_acc={acc:.4f}", flush=True)
        if patience and bad >= patience:
            break
    best["trained_epochs"] = epoch + 1
    model.load_state_dict(best["state"])
    model.eval()
    return best
