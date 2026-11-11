#!/usr/bin/env python3
"""GOAT (Kong et al., ICML 2023, "GOAT: A Global Transformer on Large-scale Graphs") on the project's NPZ
datasets under the shared comparison protocol (../common/protocol.py).

Official code: https://github.com/devnkong/GOAT (commit in UPSTREAM.txt; unmodified copies in official/).
Kept from the official code (arxiv_ERM_ns.py, large_model.py, sampler.py, vq.py, pos_enc/)
  model       one GOAT layer, conv_type full: local attention over the sampled [20, 10, 5] neighbourhood
              with hop-distance bias and distance-count normalisation, concatenated with the global
              attention over the EMA k-means codebook (num_centroids, log cluster-size bias, Corollary 3.4);
              fc_in / FFN / fc_out with BatchNorm, ff_dropout 0.5, attn_dropout 0, skip 0
              (goat_model.py = official large_model.py without torch_sparse / PyG / einops, test_goat.py)
  sampler     LocalSampler.sample_one: rounds of sample_adj over all nodes collected so far
              (goat_sampler.py, vectorised; official/sampler.py needs torch_sparse)
  inputs      raw node features; node2vec positional encodings of width global_dim, trained once per graph
              with the settings of pos_enc/*_node2vec.py (node2vec_pe.py)
  training    Adam (lr, no weight decay), mini-batches of prediction nodes, cross-entropy; single-logit BCE
              on binary datasets (train_bi / test_bi: pokec, genius, twitch-gamers, i.e. every binary dataset
              of the official script)
Replaced by the shared protocol
  data files and splits (symmetric graph without self loops, also for arxiv_year, which arxiv_ERM_ns.py
  keeps directed), seeds/runs, epoch budget (official: 500 epochs, best validation epoch), early stopping,
  validation every --eval-every epochs (official: --test_freq), checkpoint selection and one test evaluation
  with the best checkpoint, ROC-AUC on the binary imbalanced datasets, timing and reporting.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.protocol import (  # noqa: E402
    ProtocolData, RunResult, add_protocol_args, count_parameters, compute_metric, default_out_dir,
    finalize_args, fit, report, resolve_device, set_seed,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


goat_model = _load("goat_model", HERE / "goat_model.py")
goat_sampler = _load("goat_sampler", HERE / "goat_sampler.py")
node2vec_pe = _load("goat_node2vec_pe", HERE / "node2vec_pe.py")


def build_parser():
    p = argparse.ArgumentParser(description="GOAT (ICML'23) on NPZ datasets (shared comparison protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("GOAT (names and defaults of arxiv_ERM_ns.py)")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--test-batch-size", type=int, default=1024)
    g.add_argument("--sizes", type=int, nargs="+", default=[20, 10, 5])
    g.add_argument("--test-sizes", type=int, nargs="+", default=[20, 10, 5])
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--conv-type", choices=["local", "global", "full"], default="full")
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--global-dim", type=int, default=64)
    g.add_argument("--num-layers", type=int, default=1)
    g.add_argument("--num-heads", type=int, default=1)
    g.add_argument("--attn-dropout", type=float, default=0.0)
    g.add_argument("--ff-dropout", type=float, default=0.5)
    g.add_argument("--skip", type=int, default=0)
    g.add_argument("--dist-count-norm", type=int, default=1)
    g.add_argument("--num-centroids", type=int, default=4096)
    g.add_argument("--no-bn", action="store_true")
    g.add_argument("--norm-type", choices=["batch_norm", "layer_norm"], default="batch_norm")
    g.add_argument("--loss", choices=["auto", "ce", "bce"], default="auto",
                   help="auto: single-logit BCE for binary datasets (train_bi), cross-entropy otherwise")
    g = p.add_argument_group("positional encodings (pos_enc/*_node2vec.py)")
    g.add_argument("--pe-setting", choices=sorted(node2vec_pe.PE_SETTINGS), default="arxiv")
    g.add_argument("--pe-cache", default=str(HERE / "cache"))
    g.add_argument("--pe-seed", type=int, default=None, help="default: --base-seed (one encoding for all runs)")
    return p


def main():
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "GOAT", args.dataset)
    if args.num_layers != 1:
        raise SystemExit("GOAT uses one attention layer (assert num_layers == 1 in large_model.py)")
    if len(args.sizes) != len(args.test_sizes):
        raise SystemExit("--sizes and --test-sizes need the same length (distance embedding)")
    device = resolve_device(args.device)

    data = ProtocolData(args)
    n, num_classes = data.num_nodes, data.num_classes
    loss_kind = args.loss if args.loss != "auto" else ("bce" if num_classes == 2 else "ce")
    out_dim = 1 if loss_kind == "bce" else num_classes
    x_cpu = torch.from_numpy(np.ascontiguousarray(data.features, dtype=np.float32))
    y_cpu = torch.from_numpy(data.labels.astype(np.int64))
    graph = goat_sampler.GraphCSR(data.adj)

    pe_info = {"seconds": 0.0, "peak_mem_mb": 0.0, "cached": False}
    if args.conv_type == "local":
        pos_enc = x_cpu                                        # arxiv_ERM_ns.py: pos_enc = x (unused)
    else:
        pe_seed = args.base_seed if args.pe_seed is None else args.pe_seed
        pos_enc, pe_info = node2vec_pe.load_or_compute(data.adj, args.dataset_key, args.pe_setting,
                                                       args.global_dim, Path(args.pe_cache), device, pe_seed)
        pos_enc = pos_enc.float().contiguous()
        if pos_enc.shape != (n, args.global_dim):
            raise SystemExit(f"positional encoding shape {tuple(pos_enc.shape)} != ({n}, {args.global_dim})")

    print("==== GOAT (official model and sampler, shared protocol) ====")
    print(f"data_root={args.data_root} {data.summary()} metric={args.metric} device={device} loss={loss_kind} "
          f"conv={args.conv_type} sizes={args.sizes} batch={args.batch_size}/{args.test_batch_size} "
          f"pe={args.pe_setting} ({'cached' if pe_info.get('cached') else 'computed'}, "
          f"{pe_info['seconds']:.1f}s)")
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    loader_kw = dict(num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True

    def make_loader(ids, sizes, batch_size, shuffle):
        ids_t = torch.from_numpy(ids.astype(np.int64))
        # A last training batch with a single prediction node would stop BatchNorm1d in the FFN (official code
        # has no guard); only in that case the one leftover node of each shuffled epoch is dropped.
        drop_last = shuffle and len(ids) > batch_size and len(ids) % batch_size == 1
        if args.conv_type == "global":
            return torch.utils.data.DataLoader(ids_t, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                               **loader_kw)
        return goat_sampler.LocalSampler(graph, sizes, node_idx=ids_t, batch_size=batch_size, shuffle=shuffle,
                                         drop_last=drop_last, **loader_kw)

    def forward(model, batch):
        if args.conv_type == "global":
            node_idx = batch
            return model.global_forward(x_cpu[node_idx].to(device), pos_enc[node_idx].to(device), node_idx), \
                node_idx
        edge_pack, node_idx, bs = batch
        targets = node_idx[:bs]
        out = model(x_cpu[node_idx].to(device), edge_pack[:2].to(device), edge_pack[2:].to(device),
                    pos_enc[targets].to(device), targets)
        return out, targets

    results = []
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)
        loaders = {"train": make_loader(splits["train"], args.sizes, args.batch_size, True),
                   "val": make_loader(splits["val"], args.test_sizes, args.test_batch_size, False),
                   "test": make_loader(splits["test"], args.test_sizes, args.test_batch_size, False)}
        model = goat_model.Transformer(
            num_nodes=n, in_channels=data.num_features, hidden_channels=args.hidden_dim, out_channels=out_dim,
            global_dim=args.global_dim, num_layers=args.num_layers, heads=args.num_heads,
            ff_dropout=args.ff_dropout, attn_dropout=args.attn_dropout, spatial_size=len(args.sizes),
            skip=args.skip > 0, dist_count_norm=args.dist_count_norm > 0, conv_type=args.conv_type,
            num_centroids=args.num_centroids, no_bn=args.no_bn, norm_type=args.norm_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        def loss_fn(out, y):
            if loss_kind == "bce":
                return F.binary_cross_entropy_with_logits(out.view(-1), y.float())
            return F.cross_entropy(out, y)

        def train_epoch(epoch):
            model.train()
            total, count = 0.0, 0
            for batch in loaders["train"]:
                optimizer.zero_grad()
                out, targets = forward(model, batch)
                loss = loss_fn(out, y_cpu[targets].to(device))
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * int(targets.numel())
                count += int(targets.numel())
            return total / max(count, 1)

        @torch.no_grad()
        def evaluate(split):
            model.eval()
            outs, ids = [], []
            for batch in loaders[split]:
                out, targets = forward(model, batch)
                outs.append(out.float().cpu())
                # clone: tensors received from DataLoader workers live in shared memory, each holding a file
                # descriptor; keeping ~1.6k of them (pokec validation) exceeds `ulimit -n`
                ids.append(targets.clone())
            out = torch.cat(outs, dim=0)
            ids = torch.cat(ids, dim=0)
            return compute_metric(args.metric, out, y_cpu[ids])

        tag = f"GOAT seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix, params=count_parameters(model),
                                 pre_time_s=float(pe_info["seconds"]), pre_peak_mem_mb=float(pe_info["peak_mem_mb"]),
                                 extra={"pe_cached": float(bool(pe_info.get("cached")))}, **res))
        del model, optimizer, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("GOAT", results, args,
           extra_config={"loss": loss_kind, "out_dim": out_dim, "pe_setting": args.pe_setting,
                         "pe": {k: v for k, v in pe_info.items() if isinstance(v, (int, float, str, bool))}})


if __name__ == "__main__":
    main()
