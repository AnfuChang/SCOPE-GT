#!/usr/bin/env python3
"""GT-SNT (Zhang et al., AAAI 2026, "GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node
Tokenization"; arXiv:2504.11840, v1 named GT-SVQ) on the project's NPZ datasets.

Official code: https://github.com/Zhhuizhe/GT-SNT (commit in UPSTREAM.txt), every file copied unchanged into
official/; the model (official/nn: GTSNT = GCN branch + spiking node tokenizer SNT + codebook-guided attention),
the edge pre-processing (utils.precalculate_edge_attr) and the loss are imported from there.

Pipelines -- the run script chooses the one the official repository uses for the dataset
  full      node_cls.py: whole graph, one Adam step per epoch (Planetoid, Coauthor, ogbn-arxiv configs)
  subgraph  node_cls_subgraph.py (ogbn-products config): batches of --batch-size target nodes, each with its
            k-hop in-subgraph (k = layers) for the GCN branch, while the tokenizer runs on the full graph
Kept from the official code
  model and its arguments, edge attributes ("default" coding: self loops, unit weights), feature preprocessing of
  dataset.py (PyG NormalizeFeatures for Amazon / Coauthor, raw features otherwise), Adam, cross-entropy.
  In the subgraph pipeline the GCN branch gets the k-hop subgraph built from the self-looped edges and the
  tokenizer the raw full graph, as node_cls_subgraph.py does (data.full_graph is built before the self loops).
Replaced by the shared protocol (../common/protocol.py)
  data files and splits, seeds/runs (official configs: 1 run, seed 123; the paper: 5 seeds), epochs (upper bound
  max(1000, official epochs)), early stopping and validation frequency (official: every epoch / every 20 epochs,
  patience 200; the test set was also evaluated at every check), model selection on the validation metric with
  one test evaluation, ROC-AUC on the binary imbalanced datasets, timing and reporting.
Implementation notes (same computation as the official scripts)
  edges are passed as edge-index tensors (GCNConv and SNT compute the same normalisation and sums as on the official
  SparseTensors), except for the GCN branch of the subgraph pipeline: there an [edges x hid] message tensor does not
  fit (pokec: ~30M subgraph edges per batch, 3.5 GiB per tensor), so it gets the k-hop subgraph as a torch.sparse
  CSR adj_t and aggregates with spmm. That matrix already holds one unit self loop per node (the subgraph's own loops
  dropped, one per node inserted: what fill_diag / add_remaining_self_loops give) and the convs' add_self_loops is
  off, so gcn_norm does not re-sort it in every layer. SNT stays on the edge index: its messages are only
  cb_channels wide, while on CUDA the backward of a CSR spmm converts the transposed matrix back to CSR, i.e. sorts
  all edges of the full graph, at each of its (T - 1) x layers propagations per batch. torch_sparse is therefore
  optional: when it is not installed a placeholder module lets official/utils.py import (its SparseTensor branches
  are never reached).
  The loss is computed on the training device (node_cls.py moves the logits to the CPU first).
  Params counts every trainable weight, including the per-node random features of SNT (layers x N x cb_channels).
Speed (same values; test_gtsnt.py compares each against the official computation)
  SNT's torch.unique(dim=0) over the N spike rows (a comparison sort that reads the rows column by column) is
  replaced by a 1-D unique of the rows read as base-(T+1) numbers: same order, inverse indices and counts. The
  official file is unchanged; SNT.forward is re-bound below with only that line different.
  Subgraph pipeline: the k-hop edges come from a growing node mask instead of k_hop_subgraph, which also runs a
  unique over the collected hop lists (~30M entries per pokec batch) whose result is not used; at evaluation the
  tokenizer runs once per evaluation instead of once per batch (its output depends on its parameters and the full
  graph, not on the batch).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # ../common
sys.path.insert(0, str(HERE / "official"))     # the official top-level packages: nn, functional, utils

try:
    import torch_sparse  # noqa: F401
except ImportError:  # [ACE-baseline] only the edge-index (Tensor) code paths of official/ are used here
    from torch_geometric.typing import SparseTensor as _SparseTensorPlaceholder

    def _needs_torch_sparse(*_args, **_kwargs):
        raise ImportError("torch_sparse is required for SparseTensor inputs (not used by gtsnt_npz.py)")

    _shim = types.ModuleType("torch_sparse")
    _shim.SparseTensor = _SparseTensorPlaceholder
    for _name in ("fill_diag", "sum", "mul"):
        setattr(_shim, _name, _needs_torch_sparse)
    sys.modules["torch_sparse"] = _shim

from common.protocol import (  # noqa: E402
    ProtocolData, RunResult, add_protocol_args, compute_metric, count_parameters, default_out_dir, finalize_args,
    fit, report, resolve_device, set_seed,
)
from torch_geometric.utils import scatter  # noqa: E402
from nn import GTSNT  # noqa: E402  (official)
from nn.snt_conv import SNT  # noqa: E402  (official)
from functional import compute_loss  # noqa: E402  (official)
from utils import precalculate_edge_attr  # noqa: E402  (official)


def unique_spike_codes(spikes: torch.Tensor, T: int):
    """inverse indices and counts of torch.unique(spikes, return_inverse=True, return_counts=True, dim=0) for SNT's
    spike counts (0..T per channel): a row read as a base-(T+1) number, first channel most significant, sorts like
    the row, so a 1-D unique (radix sort) gives the same codes in the same order. Other input: the original call."""
    base, channels = T + 1, spikes.size(1)
    digits = spikes.to(torch.int64)
    if (channels * math.log2(base) < 62 and torch.equal(digits.to(spikes.dtype), spikes)
            and int(digits.min()) >= 0 and int(digits.max()) < base):
        weights = base ** torch.arange(channels - 1, -1, -1, device=spikes.device)
        _, inverse, counts = torch.unique((digits * weights).sum(1), return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = torch.unique(spikes, return_inverse=True, return_counts=True, dim=0)
    return inverse, counts


_official_snt_forward = SNT.forward


def _snt_forward(self, edge_index, edge_weight=None):
    """official SNT.forward (official/nn/snt_conv.py) with torch.unique(dim=0) replaced by unique_spike_codes"""
    embds = [self.rand_feat]
    for _ in range(len(embds), self.T):
        embds.append(self.propagate(edge_index, x=embds[-1], edge_weight=edge_weight))
    spikes_out = self.neuron(torch.stack(embds, dim=0)).sum(0)
    codes_indices, counts = unique_spike_codes(spikes_out.detach(), self.T)
    codebook = scatter(spikes_out, codes_indices, dim=0, reduce="mean")
    return codebook, codes_indices, counts


SNT.forward = _snt_forward


@contextlib.contextmanager
def tokenizer_once(model, adj_t):
    """evaluation: every layer's tokenizer runs once and each batch reuses its output (the codebook depends on the
    tokenizer's parameters and the full graph, not on the batch); membrane potentials reset as after a forward"""
    outs = [conv(adj_t) for conv in model.cb_convs]
    model.reset_membrane_potential()
    try:
        for conv, out in zip(model.cb_convs, outs):
            conv.forward = lambda *args, _out=out, **kwargs: _out
        yield
    finally:
        for conv in model.cb_convs:
            conv.__dict__.pop("forward", None)


def build_parser():
    p = argparse.ArgumentParser(description="GT-SNT (AAAI'26) on NPZ datasets (shared comparison protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("GT-SNT (names follow the keys of official/config/*.yaml)")
    g.add_argument("--pipeline", choices=["full", "subgraph"], default="full")
    g.add_argument("--hid-channels", type=int, default=512)
    g.add_argument("--cb-channels", type=int, default=8, help="dimension of the random features of the tokenizer")
    g.add_argument("--num-layers", type=int, default=2)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--T", type=int, default=3, help="propagation steps of the spiking node tokenizer")
    g.add_argument("--neuron", choices=["IF", "LIF", "PLIF"], default="PLIF")
    g.add_argument("--v-threshold", type=float, default=1.0)
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--attn-dropout", type=float, default=0.5)
    g.add_argument("--coding-type", default="default", help="edge attributes: default (unit) | sym | lap | degree")
    g.add_argument("--maximum-codes-num", type=int, default=0, help="keep the most frequent codes only; 0 = all")
    g.add_argument("--normalize", type=int, default=0, help="L2-normalise the GCN output (1/0)")
    g.add_argument("--enable-residual", type=int, default=0)
    g.add_argument("--enable-norm", type=int, default=0, help="BatchNorm after every layer")
    g = p.add_argument_group("training")
    g.add_argument("--lr", type=float, default=0.001)
    g.add_argument("--wd", type=float, default=0.001, help="weight decay")
    g.add_argument("--loss-func", default="cross_entropy")
    g.add_argument("--batch-size", type=int, default=512, help="subgraph: target nodes per batch")
    g.add_argument("--eval-batch-size", type=int, default=0, help="subgraph: default = --batch-size (official)")
    g.add_argument("--feature-norm", choices=["none", "pyg"], default="none",
                   help="pyg: torch_geometric NormalizeFeatures (dataset.py: Amazon / Coauthor)")
    g.add_argument("--official-epochs", type=int, default=None, help="epochs of the official config (recorded)")
    return p


def pyg_normalize_features(x: np.ndarray) -> np.ndarray:
    """torch_geometric.transforms.NormalizeFeatures: x - x.min(), then rows divided by max(row sum, 1)."""
    x = x - x.min()
    return (x / np.clip(x.sum(axis=1, keepdims=True), 1.0, None)).astype(np.float32)


def target_major(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Edges sorted by (target, source). Boolean-mask subsets keep this order, so the edges k_hop_subgraph returns
    go to csr_adj_t without another sort."""
    return edge_index[:, torch.argsort(edge_index[1] * num_nodes + edge_index[0])]


def khop_in_edges(nodes: torch.Tensor, num_hops: int, edge_index: torch.Tensor, num_nodes: int,
                  self_loops: bool = True) -> torch.Tensor:
    """k_hop_subgraph(nodes, num_hops, edge_index, directed=True)[1] for edges with a self loop at every node
    (precalculate_edge_attr): the edges whose target lies within num_hops - 1 hops of the batch, in edge order.
    A node mask grows hop by hop instead of collecting the hop lists and running unique over them.
    self_loops=False leaves the self loops out of the result."""
    src, dst = edge_index[0], edge_index[1]
    ball = torch.zeros(num_nodes, dtype=torch.bool, device=src.device)
    ball[nodes] = True
    for _ in range(num_hops - 1):
        ball[src[ball[dst]]] = True
    keep = ball[dst]
    if not self_loops:
        keep &= src != dst
    return edge_index[:, keep]


def csr_adj_t(edge_index: torch.Tensor, num_nodes: int, self_loops: bool = False) -> torch.Tensor:
    """adj_t (row = target, col = source, unit values) of target-major edges as a torch.sparse CSR matrix, i.e.
    SparseTensor.from_edge_index(edge_index).t() of node_cls_subgraph.py. self_loops: the edges have none and one is
    inserted per node, in sorted place without a sort (row v: sources < v, then v, then sources > v)."""
    src, dst = edge_index[0], edge_index[1]
    key = dst * num_nodes + src
    if key.numel() > 1 and bool((key[1:] <= key[:-1]).any()):
        raise ValueError("csr_adj_t needs target-major edges without duplicates (see target_major)")
    del key
    counts = torch.bincount(dst, minlength=num_nodes)
    crow = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    if not self_loops:
        col = src.clone()
    else:
        if bool((src == dst).any()):
            raise ValueError("csr_adj_t(self_loops=True) needs edges without self loops")
        nodes = torch.arange(num_nodes, device=dst.device)
        col = src.new_empty(src.numel() + num_nodes)
        pos = torch.arange(src.numel(), device=src.device)
        pos += dst              # the loops of nodes 0..v-1 come before row v
        pos += src > dst        # and the loop of v before its larger sources
        col[pos] = src
        del pos
        col[crow[:-1] + nodes + torch.bincount(dst[src < dst], minlength=num_nodes)] = nodes
        crow = crow + torch.arange(num_nodes + 1, device=dst.device)
    return torch.sparse_csr_tensor(crow, col, torch.ones(col.numel(), device=col.device),
                                   size=(num_nodes, num_nodes))


def build_model(args, num_nodes, num_features, num_classes):
    model = GTSNT(num_nodes, num_features, num_classes, args.hid_channels, args.cb_channels, args.num_layers,
                  args.num_heads, args.T, args.neuron, args.v_threshold, args.dropout, args.attn_dropout,
                  maximum_codes_num=args.maximum_codes_num or None, normalize=bool(args.normalize),
                  enable_residual=bool(args.enable_residual), enable_norm=bool(args.enable_norm))
    model.reset_parameters()
    return model


def snt_weights(model) -> int:
    return int(sum(conv.rand_feat.numel() for conv in model.cb_convs))


def main():
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "GT-SNT", args.dataset)
    device = resolve_device(args.device)
    torch.backends.cudnn.deterministic = True     # seed_everything of the official scripts

    data = ProtocolData(args)
    n, num_classes = data.num_nodes, data.num_classes
    x_np = pyg_normalize_features(data.features) if args.feature_norm == "pyg" else data.features
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).to(device)
    y = torch.from_numpy(data.labels.astype(np.int64)).to(device)
    raw_ei = torch.from_numpy(data.edge_index())   # symmetric, no self loops (common/data.py)
    # dataset.py: data.edge_index, data.edge_weight = precalculate_edge_attr(edge_index, N, coding_type)
    edge_index, edge_weight = precalculate_edge_attr(raw_ei, n, args.coding_type)
    if args.pipeline == "full":
        graph = types.SimpleNamespace(x=x, edge_index=edge_index.to(device), edge_weight=edge_weight.to(device))
    else:
        # node_cls_subgraph.py: k-hop subgraphs of the self-looped edges for the GCN branch (target-major here, so
        # each one converts to CSR directly) and data.full_graph (raw edges, no self loops) for SNT
        edge_index = target_major(edge_index.to(device), n)
        full_graph = raw_ei.to(device)
    eval_bs = args.eval_batch_size or args.batch_size

    print("==== GT-SNT (official model, shared protocol) ====")
    print(f"data_root={args.data_root} {data.summary()} metric={args.metric} device={device} "
          f"pipeline={args.pipeline} feature_norm={args.feature_norm} official_epochs={args.official_epochs}"
          + (f" batch={args.batch_size}" if args.pipeline == "subgraph" else ""))
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    results = []
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)
        idx = {k: torch.from_numpy(v.astype(np.int64)).to(device) for k, v in splits.items()}
        model = build_model(args, n, data.num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        if args.pipeline == "full":
            def train_epoch(epoch):           # node_cls.py: train()
                model.train()
                pred = model(graph)
                loss = compute_loss(pred[idx["train"]], y[idx["train"]], args.loss_func)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                return float(loss.item())

            @torch.no_grad()
            def predict(nodes):
                model.eval()
                return model(graph)[nodes]
        else:
            for conv in model.mpnn_convs:
                conv.add_self_loops = False        # sub_adj_t already holds one unit self loop per node

            def subgraph_logits(nodes):
                # node_cls_subgraph.py: k-hop in-subgraph of the batch (on the self-looped edges) for the GCN branch,
                # here a CSR adj_t whose own self loops are replaced by one unit loop per node (fill_diag in the
                # official gcn_norm), and the raw full graph for the tokenizer
                sub_ei = khop_in_edges(nodes, args.num_layers, edge_index, n, self_loops=False)
                sub_adj_t = csr_adj_t(sub_ei, n, self_loops=True)
                del sub_ei
                return model.forward_subgraph(x, sub_adj_t, node_indices=nodes, adj_t=full_graph)

            def train_epoch(epoch):           # node_cls_subgraph.py: train()
                model.train()
                train_nodes = idx["train"][torch.randperm(len(idx["train"]), device=device)]
                total, steps = 0.0, 0
                for start in range(0, len(train_nodes), args.batch_size):
                    nodes = train_nodes[start:start + args.batch_size]
                    pred = subgraph_logits(nodes)
                    loss = compute_loss(pred, y[nodes], args.loss_func)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item())
                    steps += 1
                return total / max(steps, 1)

            @torch.no_grad()
            def predict(nodes):
                model.eval()
                with tokenizer_once(model, full_graph):
                    return torch.cat([subgraph_logits(nodes[s:s + eval_bs]) for s in range(0, len(nodes), eval_bs)])

        def evaluate(split):
            nodes = idx[split]
            return compute_metric(args.metric, predict(nodes), y[nodes])

        tag = f"GT-SNT seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix, params=count_parameters(model),
                                 extra={"snt_node_feature_weights": snt_weights(model)}, **res))
        print(f"[{tag}] params={results[-1].params:,} of which SNT per-node random features "
              f"{snt_weights(model):,} ({args.num_layers} layers x {n} nodes x {args.cb_channels})", flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("GT-SNT", results, args,
           extra_config={"pipeline": args.pipeline, "feature_norm": args.feature_norm,
                         "official_epochs": args.official_epochs,
                         "params_include": "per-node random features of SNT (layers x N x cb_channels)"})


if __name__ == "__main__":
    main()
