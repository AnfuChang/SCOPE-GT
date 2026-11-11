#!/usr/bin/env python3
"""HubGT (Liao et al., NeurIPS 2025, "HubGT: Fast Graph Transformer with Decoupled Hierarchy Labeling")
on the project's NPZ datasets.

Official code: https://github.com/gdmnl/HubGT (commit in UPSTREAM.txt), copied into this folder. The C++
hub-labeling index (Precompute/, compiled with `python setup.py build_ext --inplace`) is unchanged.

Kept from the official code
  preprocess.process_data: degree-ordered PyPLL index over the undirected graph, N_BPROOT = 128 global
  roots appended to x/y, subgraphs of ss nodes fetched per target by collate_fetch (verbatim copy in
  hubgt_collate.py), the DataLoader settings (workers only for graphs with >= 2e4 nodes);
  main.py: model.GT, AdamW, PolynomialDecayLR(warmup = max(E // 10, 50), total E, end lr 1e-9) stepped
  once per epoch with E = 300 as in scripts/run.sh, BCE with one logit on binary datasets and CE
  otherwise, the evaluation read-out (sigmoid / log-softmax of the target token);
  load_data.py: PyG NormalizeFeatures, except on the datasets where the official loader drops it.
Replaced by the shared protocol (../common/protocol.py)
  data files and splits (official: 60/20/20 random), seeds/runs, epochs/patience/evaluation frequency
  (official: every epoch, patience 50), model selection on the validation metric, ROC-AUC on the binary
  imbalanced datasets, timing and reporting.
Deliberate differences
  load_data.py applies T.RemoveIsolatedNodes. It is not applied here, so that HubGT is evaluated on the
  same nodes as every other method; isolated nodes get a subgraph made of themselves (and roots).
  NormalizeFeatures is skipped (raw features) when the features take negative values (dense fastText
  embeddings: roman_empire, amazon_ratings, questions). It subtracts the global minimum before dividing
  by the row sum, so every entry becomes ~1/F plus a tiny node-specific part (roman_empire: per-feature
  std 1.5e-4 around 3.3e-3); the model then sees almost identical nodes and predicts the majority class
  (roman_empire: 13.85 ± 0.16 test accuracy, the largest class holds ~14% of the nodes). The transform
  is meant for non-negative bag-of-words features.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
# the extension is built as top-level module `kpll_cppext`; keep its folder importable for pickling
sys.path.insert(0, str(HERE / "Precompute"))

from common.protocol import (  # noqa: E402
    CostMeter, ProtocolData, RunResult, add_protocol_args, compute_metric, count_parameters,
    default_out_dir, finalize_args, fit, metric_name, report, resolve_device, set_seed,
)
from hubgt_collate import GraphXY, PolynomialDecayLR, collate_fetch  # noqa: E402  (official, verbatim)
from model import GT  # noqa: E402  (official)

N_BPROOT = 128  # preprocess.N_BPROOT
INDEX_VERSION = "v3"  # v2 folders on the server were built by a stale (unpatched) .so
# load_data.py removes T.NormalizeFeatures for these datasets (ogbn-arxiv, pokec, genius, physics, cs,
# flickr, reddit, actor) and keeps it for all others
NO_FEATURE_NORM = {"ogbn_arxiv", "arxiv", "pokec", "genius", "physics", "coauthor_physics", "cs",
                   "coauthor_cs", "flickr", "reddit", "actor"}


def build_parser():
    p = argparse.ArgumentParser(description="HubGT on NPZ datasets (shared comparison protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("HubGT (names and defaults follow utils/config.py)")
    g.add_argument("--n-layers", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=8)
    g.add_argument("--hidden-dim", type=int, default=128)
    g.add_argument("--ffn-dim", type=int, default=128)
    g.add_argument("--dp-input", type=float, default=0.1)
    g.add_argument("--dp-bias", type=float, default=0.1)
    g.add_argument("--dp-ffn", type=float, default=0.5)
    g.add_argument("--dp-attn", type=float, default=0.5)
    g.add_argument("--aggr-output", type=int, default=0)
    g.add_argument("--var-vfeat", type=int, default=1)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--peak-lr", type=float, default=2e-4)
    g.add_argument("--end-lr", type=float, default=1e-9)
    g.add_argument("--sched-epochs", type=int, default=300,
                   help="horizon of PolynomialDecayLR (official --epoch in scripts/run.sh)")
    g.add_argument("--batch", type=int, default=1024)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--perturb-std", type=float, default=0.0)
    g.add_argument("--kbias", type=int, default=1)
    g.add_argument("-ss", "--ss", type=int, default=48, help="total nodes per subgraph")
    g.add_argument("-s0", "--s0", type=int, default=24, help="max label nodes")
    g.add_argument("-s0g", "--s0g", type=int, default=8, help="max global nodes")
    g.add_argument("-s1", "--s1", type=int, default=12, help="max inverse label nodes")
    g.add_argument("--feature-norm", choices=["auto", "pyg", "none"], default="auto",
                   help="auto: the official per-dataset choice of load_data.py; none for signed features")
    g.add_argument("--cache-dir", default=None, help="where the hub-labeling index is stored")
    g.add_argument("--rebuild-index", action="store_true", help="rebuild even if a cached index exists")
    g.add_argument("--verbose-index", action="store_true")
    return p


def pyg_normalize_features(x: np.ndarray) -> np.ndarray:
    """torch_geometric.transforms.NormalizeFeatures: x - x.min(), then rows divided by max(row sum, 1)."""
    x = x - x.min()
    return (x / np.clip(x.sum(axis=1, keepdims=True), 1.0, None)).astype(np.float32)


def resolve_feature_norm(args, x: np.ndarray) -> str:
    """auto: the official choice of load_data.py, except for signed features (see the module docstring)."""
    if args.feature_norm != "auto":
        return args.feature_norm
    if args.dataset_key in NO_FEATURE_NORM or x.min() < 0:
        return "none"
    return "pyg"


REQUIRED_FIX_VERSION = 2  # PPLL_ACE_FIX_VERSION in Precompute/ppll.h


def check_extension():
    """The unpatched kpll_cppext segfaults in fetch_parallel on sparse / long-diameter graphs (roman_empire,
    minesweeper, amazon_ratings). Both module names are checked: the main process imports
    Precompute.kpll_cppext, DataLoader workers unpickle PyPLL from the top-level kpll_cppext."""
    import importlib
    rebuild = ("cd Precompute && rm -rf build kpll_cppext.cpp kpll_cppext*.so && "
               "python setup.py build_ext --inplace --force   (run_hubgt.sh does this automatically)")
    for name in ("Precompute.kpll_cppext", "kpll_cppext"):
        module = importlib.import_module(name)
        version = getattr(module, "ace_fix_version", lambda: 0)()
        if version < REQUIRED_FIX_VERSION:
            raise RuntimeError(f"{name} loaded from {module.__file__} is a stale build (ACE fix version {version}, "
                               f"need {REQUIRED_FIX_VERSION}); it segfaults while fetching subgraphs. Rebuild: {rebuild}")
        print(f"[HubGT] {name}: {module.__file__} (ACE fix version {version})", flush=True)


def _worker_init(_worker_id):
    faulthandler.enable()  # a native crash in a DataLoader worker prints its stack to the log


def main():
    faulthandler.enable()
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "HubGT", args.dataset)
    device = resolve_device(args.device)
    try:
        mp.set_start_method("spawn")  # as preprocess.py
    except RuntimeError:
        pass
    from Precompute import PyPLL  # compiled official extension
    check_extension()

    data = ProtocolData(args)
    n = data.num_nodes
    norm = resolve_feature_norm(args, data.features)
    x = pyg_normalize_features(data.features) if norm == "pyg" else data.features
    binary = data.num_classes == 2  # load_data.py: two classes -> one output with BCE
    out_dim = 1 if binary else data.num_classes
    graph = GraphXY(torch.cat([torch.from_numpy(x), torch.zeros(N_BPROOT, x.shape[1])], dim=0).contiguous(),
                    torch.cat([torch.from_numpy(data.labels), -torch.ones(N_BPROOT, dtype=torch.long)]).contiguous())

    edge_index = data.edge_index().astype(np.uint32)       # undirected graph, both directions, no self loops
    deg = np.bincount(edge_index[0], minlength=n)
    id_map = np.argsort(deg, kind="stable").astype(np.uint32)
    # INDEX_VERSION: v3 = index built by the fixed ppll.h (checked via ace_fix_version) (ConstructTwoHop no longer reads tmp_tri[-1]);
    # indices built by the unpatched code live in the old, unversioned folders and are not reused
    cache_dir = Path(args.cache_dir) if args.cache_dir else HERE / "cache" / (
        f"{args.dataset}_{data.fingerprint['adj.npz'][:12]}_{INDEX_VERSION}")
    if "/log" in cache_dir.as_posix():
        raise ValueError("PyPLL.get_index rewrites '/log' to '/cache' in the path; choose another --cache-dir")
    print("==== HubGT (official model and index, shared protocol) ====")
    print(f"data_root={args.data_root} {data.summary()} metric={metric_name(args.metric)} device={device} "
          f"feature_norm={norm} binary={binary} index_cache={cache_dir}")
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    workers = 0 if n < 2e4 else args.num_workers
    results, build_s, build_peak = [], None, 0.0
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)

        py_pll = PyPLL()
        py_pll.set_args(not args.verbose_index, seed, args.ss, args.s0g, args.s0, args.s1)
        rebuild = build_s is None and (args.rebuild_index or not (cache_dir / "index.bin").exists())
        with CostMeter(device) as im:
            # official call; the last argument (quiet=False) makes it store the index, which the
            # DataLoader workers reload when they unpickle PyPLL
            index_s = py_pll.get_index(edge_index, np.flip(id_map), str(cache_dir), rebuild, False)
        timing_file = cache_dir / "build_time.json"
        if rebuild:
            build_s, build_peak = im.seconds, im.peak_mb
            timing_file.write_text(json.dumps({"build_s": build_s, "cpp_reported_s": float(index_s)}))
            print(f"hub-labeling index built in {im.seconds:.2f}s (C++ reported {index_s:.2f}s) -> {cache_dir}")
        elif build_s is None:
            if timing_file.exists():
                build_s = float(json.loads(timing_file.read_text())["build_s"])
            print(f"hub-labeling index loaded from {cache_dir} in {im.seconds:.2f}s; construction time "
                  f"{'%.2fs (recorded when it was built)' % build_s if build_s is not None else 'unknown, use --rebuild-index'}")

        def loader(split, shuffle, std):
            return DataLoader(torch.as_tensor(splits[split]), batch_size=args.batch, shuffle=shuffle,
                              collate_fn=partial(collate_fetch, c_handler=py_pll, graph=graph, s_total=args.ss, std=std),
                              num_workers=workers, persistent_workers=bool(workers),
                              worker_init_fn=_worker_init if workers else None,
                              prefetch_factor=2 if workers else None, pin_memory=False)

        loaders = {"train": loader("train", True, args.perturb_std), "val": loader("val", False, 0.0),
                   "test": loader("test", False, 0.0)}
        model = GT(n_layers=args.n_layers, num_heads=args.num_heads, input_dim=x.shape[1], hidden_dim=args.hidden_dim,
                   output_dim=out_dim, attn_bias_dim=args.kbias, dp_attn=args.dp_attn, dp_ffn=args.dp_ffn,
                   dp_input=args.dp_input, dp_bias=args.dp_bias, ffn_dim=args.ffn_dim, num_nodes=n,
                   num_global_node=N_BPROOT, var_vfeat=bool(args.var_vfeat), aggr_output=bool(args.aggr_output)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
        scheduler = PolynomialDecayLR(optimizer, warmup=max(args.sched_epochs // 10, 50), tot=args.sched_epochs,
                                      lr=args.peak_lr, end_lr=args.end_lr, power=1.0)
        criterion = F.binary_cross_entropy_with_logits if binary else F.cross_entropy

        def train_epoch(epoch):
            model.train()
            total, count = 0.0, 0
            for batch in loaders["train"]:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                label = batch.y.view(-1)
                if binary:
                    label = label.unsqueeze(1).float()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * label.size(0)
                count += label.size(0)
            scheduler.step()  # main.py steps the scheduler once per epoch
            return total / max(count, 1)

        @torch.no_grad()
        def evaluate(split):
            model.eval()
            scores, labels = [], []
            for batch in loaders[split]:
                batch = batch.to(device)
                output = model(batch)
                labels.append(batch.y.view(-1).cpu())
                if binary:
                    scores.append(torch.sigmoid(output.squeeze(-1)).float().cpu())
                else:
                    scores.append(F.log_softmax(output, dim=1).float().cpu())
            return compute_metric(args.metric, torch.cat(scores), torch.cat(labels),
                                  kind="probs" if binary else "log_probs")

        tag = f"HubGT seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix,
                                 pre_time_s=build_s if build_s is not None else im.seconds,
                                 pre_peak_mem_mb=build_peak, params=count_parameters(model),
                                 extra={"index_load_or_build_s": im.seconds}, **res))
        del model, optimizer, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("HubGT", results, args, extra_config={"feature_norm": norm, "binary_single_logit": binary,
                                                 "index_cache": str(cache_dir), "num_workers": workers})


if __name__ == "__main__":
    main()
