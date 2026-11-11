#!/usr/bin/env python3
"""GCFormer (Chen et al., NeurIPS 2024) on the project's NPZ datasets.

Official code: https://github.com/JHL-HUST/GCFormer (commit in UPSTREAM.txt), copied into this folder.

Kept from the official code
  model_mix.SGT (the model; only the device placement inside celoss is patched), lr.PolynomialDecayLR
  (warm-up 400 / total 1000 updates, end lr 1e-4, stepped per batch), AdamW, loss = CE + beta * contrastive,
  token construction of train_test.py (attribute tokens + propagated "hop" tokens: top-p most similar
  nodes as positive tokens, p/n random dissimilar nodes as negative tokens) and the per-dataset
  hyperparameters of Solo.sh.
Replaced by the shared protocol (../common/protocol.py)
  data files and splits, seeds/runs, epochs/patience/evaluation frequency, model selection on the
  validation metric (the official stopper watches val accuracy and loss, patience 50 epochs), ROC-AUC on
  the binary datasets, timing and reporting.
Implementation changes that do not change the inputs
  utils.node_seq_feature loops over nodes in Python and draws negatives through set(range(N)) - topk
  for every node; ../common/token_utils.py computes the same top-k and the same negative distribution
  in chunks. Tokens are kept as node indices and gathered per batch instead of materialising the
  official [N, 2(p+n+2), F] tensor.
"""
from __future__ import annotations

import argparse
import sys
import time
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
from common.token_utils import propagate, sample_excluding, sym_norm_adj_with_self_loops, topk_similar  # noqa: E402
from lr import PolynomialDecayLR  # noqa: E402  (official)
from model_mix import SGT  # noqa: E402  (official)


def build_parser():
    p = argparse.ArgumentParser(description="GCFormer on NPZ datasets (shared comparison protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("GCFormer (names follow train_test.py)")
    g.add_argument("--sample-num-p", type=int, default=7)
    g.add_argument("--sample-num-n", type=int, default=7)
    g.add_argument("--hidden-dim", type=int, default=64)
    g.add_argument("--ffn-dim", type=int, default=64)
    g.add_argument("--n-layers", type=int, default=1)
    g.add_argument("--n-heads", type=int, default=1)
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--attention-dropout", type=float, default=0.5)
    g.add_argument("--alpha", type=float, default=0.1, help="aggregation weight of attribute vs hop tokens")
    g.add_argument("--beta", type=float, default=0.9, help="contrastive loss weight")
    g.add_argument("--pp-k", type=int, default=3, help="propagation steps of the hop tokens")
    g.add_argument("--temp", type=float, default=10.0)
    g.add_argument("--batch-size", type=int, default=2048)
    g.add_argument("--warmup-updates", type=int, default=400)
    g.add_argument("--tot-updates", type=int, default=1000)
    g.add_argument("--peak-lr", type=float, default=0.005)
    g.add_argument("--end-lr", type=float, default=0.0001)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    return p


class Tokens:
    """Per-node token indices: [self, p positives, self, n negatives] for the attribute and hop views."""

    def __init__(self, feat: torch.Tensor, hop: torch.Tensor, node_idx: torch.Tensor, hop_idx: torch.Tensor):
        self.feat, self.hop, self.node_idx, self.hop_idx = feat, hop, node_idx, hop_idx

    def gather(self, nodes: torch.Tensor, device: torch.device) -> torch.Tensor:
        # same layout as train_test.py: cat(node_p, node_n, hop_p, hop_n) along the token axis
        x = torch.cat([self.feat[self.node_idx[nodes]], self.hop[self.hop_idx[nodes]]], dim=1)
        return x.to(device, non_blocking=True)


def build_shared(data: ProtocolData, args, device: torch.device) -> dict:
    """Deterministic part of the tokens (propagation and top-p similar nodes), shared by all runs."""
    if args.pp_k <= 0:
        raise ValueError("pp_k must be > 0: SGT.forward always splits the input into four token groups")
    adj_norm = sym_norm_adj_with_self_loops(data.adj)
    hop = propagate(adj_norm, data.features, args.pp_k, alpha=0.1)  # 0.9 * A X + 0.1 * X0
    return {
        "hop": hop,
        "pos": topk_similar(data.features, args.sample_num_p, device, l2_normalize=True),
        "hop_pos": topk_similar(hop, args.sample_num_p, device, l2_normalize=True),
    }


def build_tokens(data: ProtocolData, shared: dict, args, rng: np.random.Generator) -> Tokens:
    """Adds the random negatives of this run (official: redrawn with the run's numpy seed)."""
    n = data.num_nodes
    self_col = torch.arange(n).unsqueeze(1)
    node_parts = [self_col, shared["pos"], self_col]
    hop_parts = [self_col, shared["hop_pos"], self_col]
    if args.sample_num_n > 0:
        node_parts.append(sample_excluding(n, shared["pos"], args.sample_num_n, rng))
        hop_parts.append(sample_excluding(n, shared["hop_pos"], args.sample_num_n, rng))
    return Tokens(torch.from_numpy(data.features), torch.from_numpy(shared["hop"]),
                  torch.cat(node_parts, dim=1), torch.cat(hop_parts, dim=1))


def main():
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "GCFormer", args.dataset)
    device = resolve_device(args.device)
    data = ProtocolData(args)
    print("==== GCFormer (official model, shared protocol) ====")
    print(f"data_root={args.data_root} {data.summary()} metric={metric_name(args.metric)} device={device}")
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    with CostMeter(device) as sm:
        shared = build_shared(data, args, device)
    print(f"shared token preprocessing: {sm.seconds:.2f}s (propagation + top-{args.sample_num_p} similarity)")

    results = []
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)
        rng = np.random.default_rng(seed)
        with CostMeter(device) as pm:
            tokens = build_tokens(data, shared, args, rng)
        labels = torch.from_numpy(data.labels)

        model_args = SimpleNamespace(sample_num_p=args.sample_num_p, sample_num_n=args.sample_num_n,
                                     alpha=args.alpha, temp=args.temp, device=device)
        model = SGT(n_layers=args.n_layers, input_dim=data.num_features, hidden_dim=args.hidden_dim,
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
                _, ce_loss, con_loss = model(x, y)
                loss = ce_loss + args.beta * con_loss
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
                out, _, _ = model(tokens.gather(nodes, device), labels[nodes].to(device))
                outs.append(out.float().cpu())
            return compute_metric(args.metric, torch.cat(outs), labels[idx], kind="log_probs")

        tag = f"GCFormer seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix, pre_time_s=sm.seconds + pm.seconds,
                                 pre_peak_mem_mb=max(sm.peak_mb, pm.peak_mb), params=count_parameters(model), **res))
        del model, optimizer, tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("GCFormer", results, args)


if __name__ == "__main__":
    main()
