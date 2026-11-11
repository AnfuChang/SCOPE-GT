#!/usr/bin/env python3
"""Quick check of the compiled hub-labeling extension on one dataset, without training.

    python check_fetch.py --data-root /mnt/sdb/changaf25/sdust/datasets/minesweeper
    python check_fetch.py --data-root .../roman_empire --rounds 5 --workers 4

Builds the HubGT index (same edge list and degree order as hubgt_npz.py) in a temporary folder, then
fetches the subgraph of every node `--rounds` times with fetch_parallel -- the call that segfaulted in
collate_fetch -- first in this process (num_workers = 0 path) and then in spawned worker processes that
unpickle PyPLL and reload the stored index (num_workers > 0 path). Prints "OK" or dies with the native
stack trace (faulthandler).
"""
from __future__ import annotations

import argparse
import faulthandler
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "Precompute"))


def fetch_all(pll, n, batch, rounds, s_total, tag):
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for r in range(rounds):
        order = rng.permutation(n).astype(np.int32)
        for i in range(0, n, batch):
            ids, bias = pll.fetch_parallel(order[i:i + batch], s_total)
            ids = np.asarray(ids)
            if ids.size != len(order[i:i + batch]) * s_total or ids.min() < 0 or ids.max() >= n + 128:
                raise RuntimeError(f"{tag}: malformed fetch result (round {r}, offset {i})")
    print(f"[check] {tag}: fetched all {n} nodes x {rounds} rounds in {time.perf_counter() - start:.1f}s -- OK",
          flush=True)


def worker(pll, n, batch, rounds, s_total, wid):
    faulthandler.enable()
    fetch_all(pll, n, batch, rounds, s_total, f"spawned worker {wid}")


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--ss", type=int, default=48)
    p.add_argument("--s0", type=int, default=24)
    p.add_argument("--s0g", type=int, default=8)
    p.add_argument("--s1", type=int, default=12)
    args = p.parse_args()

    from hubgt_npz import check_extension
    from Precompute import PyPLL
    check_extension()

    adj = sparse.load_npz(Path(args.data_root) / "adj.npz").tocsr()
    adj = adj.maximum(adj.T).tocoo()
    keep = adj.row != adj.col
    edge_index = np.vstack([adj.row[keep], adj.col[keep]]).astype(np.uint32)
    n = adj.shape[0]
    deg = np.bincount(edge_index[0], minlength=n)
    id_map = np.argsort(deg, kind="stable").astype(np.uint32)
    print(f"[check] {args.data_root}: nodes={n} arcs={edge_index.shape[1]}", flush=True)

    with tempfile.TemporaryDirectory(prefix="hubgt_check_") as tmp:
        pll = PyPLL()
        pll.set_args(True, 42, args.ss, args.s0g, args.s0, args.s1)
        pll.get_index(edge_index, np.flip(id_map), str(Path(tmp) / "index"), True, False)
        fetch_all(pll, n, args.batch, args.rounds, args.ss, "main process")
        if args.workers:
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=worker, args=(pll, n, args.batch, args.rounds, args.ss, w))
                     for w in range(args.workers)]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()
            bad = [p.exitcode for p in procs if p.exitcode != 0]
            if bad:
                raise SystemExit(f"[check] worker exit codes {bad} (negative = killed by signal, -11 = SIGSEGV)")
    print("[check] all fetches completed: the extension is safe on this graph", flush=True)


if __name__ == "__main__":
    main()
