#!/usr/bin/env python3
"""SwapGT (Chen et al., NeurIPS 2025, "Rethinking Tokenized Graph Transformers for Node Classification")
on the project's NPZ datasets.

Official code: https://github.com/JHL-HUST/SwapGT (commit in UPSTREAM.txt), copied into this folder.

Kept from the official code
  model.SwapGT (only the .squeeze() that breaks one-node batches is patched), lr.PolynomialDecayLR
  (warm-up 400 / total 1000 updates, end lr 1e-4, stepped per batch), AdamW, loss = CE + lambda1 * token
  consistency, token construction of train_test_sh.py (attribute view: top-k nodes by feature dot
  product; topology view: top-k by dot product of propagated features; `num_aug` extra sequences per
  view produced by token swapping), and the per-dataset hyperparameters of cmd.sh (dense split).
  The swapping itself is utils.mix_token, copied verbatim below and run on node ids.
Replaced by the shared protocol (../common/protocol.py)
  data files and splits, seeds/runs, epochs/patience/evaluation frequency, model selection on the
  validation metric (official: val accuracy and loss, patience 50), ROC-AUC on the binary datasets,
  timing and reporting. The class_bank update of train_test_sh.py is dropped: the bank is passed to
  the model but never read, so it does not affect training.
Implementation changes that do not change the inputs
  top-k search in chunks (../common/token_utils.py) instead of per-node Python loops; tokens stored as
  node ids and gathered per batch instead of the official [N, 2(k+1)(num_aug+1), F] tensor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common.protocol import (  # noqa: E402
    CostMeter, ProtocolData, RunResult, add_protocol_args, compute_metric, count_parameters,
    default_out_dir, finalize_args, fit, metric_name, report, resolve_device, set_seed,
)
from common.token_utils import propagate, sym_norm_adj_with_self_loops, topk_similar  # noqa: E402
from lr import PolynomialDecayLR  # noqa: E402  (official)
from model import SwapGT  # noqa: E402  (official)


def mix_token(token_sequence, token_idx, num_aug, depth, features):
    """Verbatim copy of utils.mix_token (official utils.py imports dgl/numba/networkx at module level).

    Note that new_idx aliases token_idx (and map_idx), so every augmentation round keeps swapping the
    result of the previous one; this is the official behaviour and is preserved here.
    """
    #token_sequence, size:[n, k+1, d]
    split_list = [1, token_sequence.shape[1]-1]
    split_tensor = torch.split(token_sequence, split_list, dim=1)

    # node itself
    node_self = split_tensor[0] #[n, 1, d]
    # nodes' tokens
    sampling_tokens = split_tensor[1] #[n, k, d]

    # print("token_idx:", token_idx)
    # output token
    final_aug_token = token_sequence
    map_idx = token_idx

    for i in range(num_aug):
        new_idx = token_idx
        for j in range(depth):
            # random_token_idx (n, k)
            random_indices = torch.randint(0, token_sequence.shape[1]-1, (token_sequence.shape[0],token_sequence.shape[1]-1))

            # mix_token index (n,k)
            # update swap strategy
            new_indices = map_idx[new_idx.to(torch.long), random_indices.to(torch.long)]

            k = new_idx.shape[1]
            kk = int(1*k/2)
            selected_cols = torch.randperm(k)[:kk]
            for col in selected_cols:
                new_idx[:,col] = new_indices[:,col]

        # print("new_idx:",new_idx)

        aug_token = features[new_idx.to(torch.long)]

        temp_token = torch.cat((node_self, aug_token), dim=1)

        final_aug_token = torch.cat((final_aug_token, temp_token), dim=1)

    return final_aug_token # [n, (k+1)*(num_aug+1), d]


def build_parser():
    p = argparse.ArgumentParser(description="SwapGT on NPZ datasets (shared comparison protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("SwapGT (names follow train_test_sh.py)")
    g.add_argument("--hidden", type=int, default=512)
    g.add_argument("--ffn-dim", type=int, default=64)
    g.add_argument("--n-layers", type=int, default=1)
    g.add_argument("--n-heads", type=int, default=8)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--attention-dropout", type=float, default=0.1)
    g.add_argument("--alpha", type=float, default=0.1, help="restart weight of the feature propagation")
    g.add_argument("--pp-k", type=int, default=3)
    g.add_argument("--num-aug", type=int, default=2)
    g.add_argument("--depth", type=int, default=1)
    g.add_argument("--sample-num", type=int, default=4)
    g.add_argument("--lambda1", type=float, default=0.5)
    g.add_argument("--lambda2", type=float, default=0.5)
    g.add_argument("--batch-size", type=int, default=2048)
    g.add_argument("--warmup-updates", type=int, default=400)
    g.add_argument("--tot-updates", type=int, default=1000)
    g.add_argument("--peak-lr", type=float, default=0.005)
    g.add_argument("--end-lr", type=float, default=0.0001)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    return p


class Tokens:
    """Node ids of the attribute view and the topology view, [N, (k+1)(num_aug+1)] each.

    Attribute view: every position reads the raw features. Topology view (as in train_test_sh.py,
    where mix_token receives the propagated sequence but gathers the swapped tokens from the raw
    features): the first sequence and the leading self token of every swapped sequence read the
    propagated features, the swapped tokens read the raw features.
    """

    def __init__(self, raw, prop, attr_ids, topo_ids, k):
        self.raw, self.prop, self.attr_ids, self.topo_ids = raw, prop, attr_ids, topo_ids
        pos = torch.arange(topo_ids.shape[1])
        self.topo_from_prop = (pos < k + 1) | (pos % (k + 1) == 0)

    def gather(self, nodes: torch.Tensor, device: torch.device) -> torch.Tensor:
        attr = self.raw[self.attr_ids[nodes].long()]
        ids = self.topo_ids[nodes].long()
        topo = self.raw[ids]
        topo[:, self.topo_from_prop] = self.prop[ids[:, self.topo_from_prop]]
        return torch.cat([attr, topo], dim=1).to(device, non_blocking=True)


def build_shared(data: ProtocolData, args, device: torch.device) -> dict:
    """Deterministic part: propagated features and top-k lists of both views."""
    adj_norm = sym_norm_adj_with_self_loops(data.adj)
    prop = propagate(adj_norm, data.features, args.pp_k, alpha=args.alpha)
    return {
        "prop": prop,
        "attr_topk": topk_similar(data.features, args.sample_num, device, l2_normalize=False),
        "topo_topk": topk_similar(prop, args.sample_num, device, l2_normalize=False),
    }


def build_tokens(data: ProtocolData, shared: dict, args) -> Tokens:
    """Token swapping of this run (uses the torch RNG seeded with the run seed, as the official code)."""
    n = data.num_nodes
    id_col = torch.arange(n, dtype=torch.float64).unsqueeze(1)          # "features" = node ids
    views = []
    for key in ("attr_topk", "topo_topk"):
        topk = shared[key]
        seq = torch.cat([id_col, topk.to(torch.float64)], dim=1).unsqueeze(-1)  # [N, k+1, 1]
        token_idx = topk.to(torch.float32).clone()                       # official: float tensor of ids
        ids = mix_token(seq, token_idx, args.num_aug, args.depth, id_col).squeeze(-1)
        views.append(ids.to(torch.int64 if n >= 2 ** 31 else torch.int32))
    return Tokens(torch.from_numpy(data.features), torch.from_numpy(shared["prop"]), views[0], views[1],
                  args.sample_num)


def main():
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "SwapGT", args.dataset)
    device = resolve_device(args.device)
    data = ProtocolData(args)
    print("==== SwapGT (official model, shared protocol) ====")
    print(f"data_root={args.data_root} {data.summary()} metric={metric_name(args.metric)} device={device}")
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    with CostMeter(device) as sm:
        shared = build_shared(data, args, device)
    print(f"shared token preprocessing: {sm.seconds:.2f}s (propagation + top-{args.sample_num} of both views)")

    labels = torch.from_numpy(data.labels)
    results = []
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)
        with CostMeter(device) as pm:
            tokens = build_tokens(data, shared, args)

        model_args = SimpleNamespace(sample_num=args.sample_num, num_aug=args.num_aug,
                                     lambda1=args.lambda1, lambda2=args.lambda2)
        model = SwapGT(n_layers=args.n_layers, input_dim=data.num_features, hidden_dim=args.hidden,
                       n_class=data.num_classes, num_heads=args.n_heads, ffn_dim=args.ffn_dim,
                       dropout_rate=args.dropout, attention_dropout_rate=args.attention_dropout,
                       args=model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
        scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.warmup_updates, tot_updates=args.tot_updates,
                                      lr=args.peak_lr, end_lr=args.end_lr, power=1.0)
        train_idx = torch.from_numpy(splits["train"])
        gen = torch.Generator().manual_seed(seed)

        def train_epoch(epoch):
            model.train()
            perm = train_idx[torch.randperm(len(train_idx), generator=gen)]
            total, count = 0.0, 0
            for start in range(0, len(perm), args.batch_size):
                nodes = perm[start:start + args.batch_size]
                x = tokens.gather(nodes, device)
                y = labels[nodes].to(device)
                optimizer.zero_grad()
                _, loss, _, _ = model(x, y, True, None)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += float(loss.item()) * len(nodes)
                count += len(nodes)
            return total / max(count, 1)

        @torch.no_grad()
        def evaluate(split):
            model.eval()
            idx = torch.from_numpy(splits[split])
            outs = []
            for start in range(0, len(idx), args.batch_size):
                nodes = idx[start:start + args.batch_size]
                out, _, _, _ = model(tokens.gather(nodes, device), labels[nodes].to(device), False, None)
                outs.append(out.float().cpu())
            return compute_metric(args.metric, torch.cat(outs), labels[idx], kind="probs")

        tag = f"SwapGT seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix, pre_time_s=sm.seconds + pm.seconds,
                                 pre_peak_mem_mb=max(sm.peak_mb, pm.peak_mb), params=count_parameters(model), **res))
        del model, optimizer, tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("SwapGT", results, args)


if __name__ == "__main__":
    main()
