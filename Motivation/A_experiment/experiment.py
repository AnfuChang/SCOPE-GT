"""A Motivation: paired Full / Partial / Center Query-set interventions.

Two-layer sampled sequence GT, center classification only, no structural encoding.
Every center uniformly samples at most K distinct contexts from its h-hop ball;
training resamples each epoch, validation/test use fixed paired samples. All modes
use the same token-wise FFN. Partial selects half of valid context Queries by an
independent, center-keyed random stream; there is no attention preview/selector.

Each mode is independently trained from matched initialization, batches, samples,
dropout masks, and maximum epoch budget. Select checkpoints by validation only.
The experiment tests whether context attention is necessary, not whether it is
universally harmful, and makes no efficiency claim. Protocol: A_query_v2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

if __package__ in (None, ""):
    import importlib
    import sys
    _folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(_folder.parent))
    __package__ = _folder.name
    importlib.import_module(__package__)

from .common import device_from, fit, seed_all
from .gated import GatedReader

PROTOCOL = "A_query_v2"
VARIANTS = ("full", "partial", "center")


def load_graph(root, feature_normalize="l1", split_index=0):
    root = Path(root)
    if (root / 'train_split0.npy').is_file():
        from SCOPE_GT.data import load_graph as load_scope_graph
        graph = load_scope_graph(root, f'_split{split_index}', feature_normalize)
        return graph.adj, graph.features, graph.labels, graph.splits
    adj = sp.load_npz(root / "adj.npz").tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    feat = np.load(root / "feat.npy").astype(np.float32)
    if feature_normalize == "l1":
        feat /= np.maximum(np.abs(feat).sum(1, keepdims=True), 1e-12)
    elif feature_normalize != "none":
        raise ValueError("feature_normalize must be l1 or none")
    label = np.load(root / "label.npy").astype(np.int64).reshape(-1)
    split = {}
    for name in ("train", "val", "test"):
        ids = np.load(root / f"{name}.npy")
        if ids.ndim == 2:
            if ids.dtype == np.bool_ and ids.shape[0] == adj.shape[0]:
                ids = ids[:, split_index]
            else:
                ids = ids[split_index]
        split[name] = np.flatnonzero(ids) if ids.dtype == np.bool_ else ids.astype(np.int64).reshape(-1)
    if adj.shape[0] != adj.shape[1] or feat.ndim != 2 or feat.shape[0] != adj.shape[0] or len(label) != adj.shape[0]:
        raise ValueError("adjacency, features, and labels must have matching node counts")
    if not np.isfinite(feat).all():
        raise ValueError("features must be finite")
    for name, ids in split.items():
        if not len(ids) or (ids < 0).any() or (ids >= len(label)).any() or len(np.unique(ids)) != len(ids):
            raise ValueError(f"invalid or empty {name} split")
        if (label[ids] < 0).any():
            raise ValueError(f"{name} contains unlabeled nodes")
    if any(np.intersect1d(split[a], split[b]).size for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/val/test splits must be disjoint")
    return adj, feat, label, split


def neighborhood(adj, hops):
    """Nodes within h hops (excluding the node itself), as CSR.

    float32 instead of float64: only indptr/indices are used, so the data array only costs memory.
    Each extra hop multiplies nnz by roughly the mean degree; use 2 hops on dense or large graphs.
    """
    if hops < 1:
        raise ValueError("hops must be positive")
    n = adj.shape[0]
    step = (adj + sp.eye(n, format="csr")).astype(np.float32).tocsr()
    reach = sp.eye(n, format="csr", dtype=np.float32)
    for _ in range(hops):
        reach = (reach @ step).tocsr()
        reach.data[:] = 1.0
    reach.setdiag(0)
    reach.eliminate_zeros()
    reach.sort_indices()
    return reach


def sample_contexts(reach, centers, k, rng):
    """Uniformly sample at most k neighbors per center without replacement -> (nodes [b, k], 0 for empty slots; mask [b, k])."""
    if k < 1:
        raise ValueError("K must be positive")
    centers = np.asarray(centers, dtype=np.int64)
    starts, ends = reach.indptr[centers], reach.indptr[centers + 1]
    counts = ends - starts
    rows = np.repeat(np.arange(len(centers)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(counts.sum()) - first                     # position among this row's candidates
    cand = reach.indices[np.repeat(starts, counts) + pos]
    order = np.lexsort((rng.random(len(cand)), rows))          # shuffle within rows; row order and length kept
    keep = pos < k
    nodes = np.zeros((len(centers), k), np.int64)
    mask = np.zeros((len(centers), k), bool)
    nodes[rows[keep], pos[keep]] = cand[order][keep]
    mask[rows[keep], pos[keep]] = True
    return nodes, mask


def make_gate(mode, mask, centers, seed, phase, epoch=0, ratio=0.5):
    """Independent center-keyed Query membership; never consumes sampling RNG."""
    mask = np.asarray(mask, dtype=bool)
    if mode == "full":
        return mask.copy()
    if mode == "center":
        return np.zeros_like(mask)
    if mode != "partial" or not 0 <= ratio <= 1:
        raise ValueError("invalid Query mode or partial ratio")
    gate = np.zeros_like(mask)
    for row, center in enumerate(centers):
        eligible = np.flatnonzero(mask[row])
        budget = int(np.rint(ratio * len(eligible)))
        rng = np.random.default_rng([seed, 17, phase, epoch, int(center)])
        chosen = rng.choice(eligible, size=budget, replace=False)
        gate[row, chosen] = True
    return gate


def run_one(variant, k, seed, data, reach, args, device):
    feat, label, split = data
    seed_all(seed)
    x = torch.as_tensor(feat, device=device)
    y_all = torch.as_tensor(label, device=device)
    model = GatedReader(feat.shape[1], int(label.max()) + 1, args.hidden_dim, args.heads, 2,
                        args.ff_dim, args.dropout).to(device)
    initial_digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        initial_digest.update(tensor.detach().cpu().numpy().tobytes())

    def inputs(centers, rng, phase, epoch=0):
        nodes, mask = sample_contexts(reach, centers, k, rng)
        gate = torch.as_tensor(make_gate(variant, mask, centers, seed, phase, epoch, args.partial_ratio), device=device)
        nodes, mask = torch.as_tensor(nodes, device=device), torch.as_tensor(mask, device=device)
        valid = torch.cat([torch.ones((len(centers), 1), dtype=torch.bool, device=device), mask], 1)
        c = torch.as_tensor(centers, device=device)
        cx, ux = x[c], x[nodes]
        return (cx, ux, valid, gate), y_all[c]

    def train_batches(epoch):
        order_rng = np.random.default_rng([seed, 11, epoch])
        order = split["train"][order_rng.permutation(len(split["train"]))]
        rng = np.random.default_rng([seed, 13, epoch])
        for s in range(0, len(order), args.batch_size):
            yield inputs(order[s:s + args.batch_size], rng, phase=0, epoch=epoch)

    @torch.no_grad()
    def accuracy(name):
        model.eval()
        rng = np.random.default_rng([seed, 23, 0 if name == "val" else 1])
        correct = 0
        for s in range(0, len(split[name]), args.eval_batch_size):
            (cx, ux, valid, g), y = inputs(split[name][s:s + args.eval_batch_size], rng, phase=1 if name == "val" else 2)
            correct += (model(cx, ux, valid, g).argmax(1) == y).sum().item()
        return correct / len(split[name])

    started = time.perf_counter()
    best = fit(model, train_batches, lambda: accuracy("val"), args.epochs, args.patience,
               lr=args.lr, weight_decay=args.weight_decay)
    return {"val_accuracy": best["acc"], "test_accuracy": accuracy("test"), "best_epoch": best["epoch"],
            "trained_epochs": best["trained_epochs"], "initial_state_sha256": initial_digest.hexdigest(),
            "seconds": time.perf_counter() - started}


def summarise(rows):
    out = []
    for k in sorted({r["K"] for r in rows}):
        for variant in VARIANTS:
            accs = [r["test_accuracy"] for r in rows if r["K"] == k and r["variant"] == variant]
            if accs:
                chosen = [r for r in rows if r["K"] == k and r["variant"] == variant]
                out.append({"dataset": chosen[0]["dataset"], "protocol": PROTOCOL, "K": k,
                            "mode": variant, "variant": variant, "test_mean": float(np.mean(accs)),
                            "test_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                            "seeds": len(accs), "mean_contexts": float(np.mean([r["mean_contexts"] for r in chosen]))})
    return out


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    device = device_from(args.device)
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "rows.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir / 'rows.csv'} already exists; choose a new --out or explicitly use --overwrite")
    if min(args.sizes) < 1 or len(set(args.sizes)) != len(args.sizes) or args.hops < 1 or args.epochs < 1:
        raise ValueError("sizes must be distinct and positive; hops and epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or len(set(args.variants)) != len(args.variants):
        raise ValueError("seeds/modes must be distinct; seeds must be nonnegative")
    if not 0 <= args.partial_ratio <= 1 or min(args.batch_size, args.eval_batch_size, args.threads) < 1:
        raise ValueError("invalid partial ratio or batch/thread count")
    if args.patience < 0 or args.split_index < 0:
        raise ValueError("patience and split_index must be nonnegative")
    adj, feat, label, split = load_graph(args.data_root, args.feature_normalize, args.split_index)
    started = time.perf_counter()
    reach = neighborhood(adj, args.hops)
    sizes = np.diff(reach.indptr)
    args.dataset = args.dataset or Path(args.data_root).name
    print(f"{args.dataset}: {adj.shape[0]} nodes, {int(adj.nnz // 2)} edges, {args.hops}-hop neighborhood "
          f"median {np.median(sizes):.0f} mean {sizes.mean():.1f} ({time.perf_counter() - started:.1f}s, "
          f"{reach.nnz / 1e6:.1f}M nnz)", flush=True)
    (out_dir / "config.json").write_text(json.dumps({"schema_version": 2, "protocol": PROTOCOL,
                                                     "args": vars(args), "dataset": args.dataset,
                                                     "paired_query_sets": True, "context_ffn": "all_modes",
                                                     "attention_layers": 2, "center_self_kv": True,
                                                     "selection_metric": "val_accuracy",
                                                     "neighborhood_median": float(np.median(sizes))},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for seed in args.seeds:
        for k in args.sizes:
            mean_ctx = float(np.minimum(sizes[split["test"]], k).mean())
            for variant in args.variants:
                result = run_one(variant, k, seed, (feat, label, split), reach, args, device)
                rows.append({"dataset": args.dataset, "protocol": PROTOCOL, "seed": seed, "K": k,
                             "mode": variant, "variant": variant,
                             "mean_contexts": mean_ctx, **result})
                print(f"  seed {seed}  K={k:<4d} {variant:<8s} test={result['test_accuracy']:.4f}  "
                      f"val={result['val_accuracy']:.4f}  epoch={result['best_epoch']}  "
                      f"{result['seconds']:.0f}s", flush=True)
                write_csv(out_dir / "rows.csv", rows)
                write_csv(out_dir / "summary.csv", summarise(rows))
    for r in summarise(rows):
        std = f" ± {100 * r['test_std']:.2f}" if r["test_std"] is not None else ""
        print(f"K={r['K']:<4d} {r['variant']:<8s} {100 * r['test_mean']:.2f}{std}")
    print(f"results: {out_dir.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", default="results/run_A_query_v2")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64], help="x-axis: number of contexts K")
    p.add_argument("--variants", "--modes", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--partial-ratio", type=float, default=0.5, help="fraction of contexts updated by partial")
    p.add_argument("--feature-normalize", choices=["l1", "none"], default="l1")
    p.add_argument("--split-index", type=int, default=0, help="split index in multi-split files; fixed across training seeds")
    p.add_argument("--overwrite", action="store_true", help="allow overwriting CSV files in the output directory")
    p.add_argument("--hops", type=int, default=3, help="hops of the sampling neighborhood; use 2 on dense or large graphs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0, help="0 = fixed training budget; >0 enables validation early stopping")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=6)
    run(p.parse_args())


if __name__ == "__main__":
    main()
