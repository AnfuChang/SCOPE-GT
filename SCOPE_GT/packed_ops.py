"""Padding-free reductions. No edges connect different center segments.

This is a portable PyTorch implementation, not a custom fused CUDA kernel.
CUDA index_add/scatter reductions may be nondeterministic at roundoff level.
"""
import torch


def segment_sum(values, owner, size):
    out = values.new_zeros((size,) + values.shape[1:])
    return out.index_add(0, owner, values)


def segment_softmax(scores, owner, size):
    """[edges, heads] -> probabilities; empty segments create no entries."""
    if scores.shape[0] == 0:
        return scores
    # Accumulate exponentials in FP32 under mixed precision.
    work = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    index = owner[:, None].expand_as(work)
    peak = work.new_full((size, work.shape[1]), -float('inf'))
    peak.scatter_reduce_(0, index, work.detach(), reduce='amax', include_self=True)
    exp = (work - peak[owner]).exp()
    denom = segment_sum(exp, owner, size)
    return (exp / denom[owner]).to(scores.dtype)
