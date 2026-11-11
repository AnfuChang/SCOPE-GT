"""Verbatim copies of utils/collator.py (Batch, collate_fetch) and utils/lr.py (PolynomialDecayLR).

The official utils/__init__.py imports torchmetrics, pandas, torch_geometric.profile and the Unix-only
`resource` module; these three objects need none of them. They live in their own module so that the
DataLoader worker processes (spawn) can unpickle the collate function.
"""
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

INF8 = 255


class Batch(object):
    def __init__(self, attn_bias, x, y, ids=None):
        super(Batch, self).__init__()
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.ids = ids

    def to(self, device):
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def collate_fetch(idx, c_handler, graph, s_total, std=0.0):
    idx = torch.stack(idx, dim=0).flatten() # avoid slice and copy for ids
    batch_size = idx.size(0)

    ids, attn_bias = c_handler.fetch_parallel(idx.numpy().astype(np.int32), s_total)
    ids = torch.tensor(ids, dtype=int).view(batch_size, s_total)
    attn_bias = torch.tensor(attn_bias, dtype=int).view(batch_size, s_total, s_total, 1)

    y = graph.y[ids[:, 0].view(-1)].view(-1)
    x = graph.x[ids.view(-1)].view(batch_size, s_total, -1)
    if std > 0:
        norm = torch.norm(x)
        x += torch.normal(0, std * norm, x.shape, device=x.device, dtype=x.dtype)

    return Batch(attn_bias, x, y, ids)


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup, tot, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup = warmup
        self.tot = tot
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer)


    def get_lr(self):
        if self._step_count <= self.warmup:
            self.warmup_factor = self._step_count / float(self.warmup)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot:
            lr = self.end_lr
        else:
            warmup = self.warmup
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


class GraphXY(object):
    """Picklable stand-in for the torch_geometric Data(x, y) passed to collate_fetch."""

    def __init__(self, x, y):
        self.x, self.y = x, y
