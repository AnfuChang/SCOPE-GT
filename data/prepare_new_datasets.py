#!/usr/bin/env python3
"""Convert LastFM-Asia and DBLP to the project NPZ layout (adj.npz, feat.npy, label.npy, train.npy, val.npy, test.npy,
metadata.json, label_mapping.json), following the existing homophilous datasets (e.g. amazon_computers):
binary symmetric float32 CSR adjacency without self loops, float32 dense features, int64 labels,
stratified random 60/20/20 split with seed 0.

    python -I prepare_new_datasets.py --lastfm-zip lastfm_asia.zip --dblp-npz dblp.npz --out-root <datasets dir>

Sources
  LastFM-Asia  https://snap.stanford.edu/data/lastfm_asia.zip (Rozemberczki & Sarkar, CIKM 2020): edges csv,
               target csv (country id), features json (liked artist ids) -> binary artist indicators, as the raw
               MUSAE features used for facebook_large
  DBLP         https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz (Bojchevski & Guennemann,
               ICLR 2018), the file PyG CitationFull("DBLP") downloads: CSR adjacency and bag-of-words attributes,
               binarised (x > 0) as PyG does; object arrays (idx_to_node, idx_to_attr) are never read
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def symmetric_binary(rows, cols, n):
    a = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    return a


def stratified_split(y, seed=0, ratios=(0.6, 0.2)):
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr, n_va = int(ratios[0] * len(idx)), int(ratios[1] * len(idx))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)).astype(np.int64) for k, v in parts.items()}


def save(out: Path, name, adj, x, y, meta):
    out.mkdir(parents=True, exist_ok=True)
    split = stratified_split(y)
    sp.save_npz(out / "adj.npz", adj)
    np.save(out / "feat.npy", x.astype(np.float32))
    np.save(out / "label.npy", y.astype(np.int64))
    for k, v in split.items():
        np.save(out / f"{k}.npy", v)
    rows, cols = adj.nonzero()
    meta = {"dataset": name, **meta, "format": "adj.npz, feat.npy, label.npy, train.npy, val.npy, test.npy",
            "num_nodes": int(adj.shape[0]), "num_edges_undirected": int(adj.nnz // 2),
            "num_edges_undirected_nnz": int(adj.nnz), "num_features": int(x.shape[1]), "num_classes": int(y.max() + 1),
            "feature_dtype": "float32", "edge_homophily": float((y[rows] == y[cols]).mean()),
            "isolated_nodes": int((np.diff(adj.indptr) == 0).sum()),
            "split_policy": "stratified_random_0.60_0.20_0.20", "num_train": int(len(split["train"])),
            "num_val": int(len(split["val"])), "num_test": int(len(split["test"])), "make_undirected": True, "seed": 0}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "label_mapping.json").write_text(json.dumps({str(i): i for i in range(int(y.max()) + 1)}, indent=2),
                                            encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def lastfm(zip_path: Path, out: Path):
    z = zipfile.ZipFile(zip_path)
    name = {Path(n).name: n for n in z.namelist()}
    edges = np.array([[int(r["node_1"]), int(r["node_2"])]
                      for r in csv.DictReader(io.StringIO(z.read(name["lastfm_asia_edges.csv"]).decode("utf-8")))])
    target = {int(r["id"]): int(r["target"])
              for r in csv.DictReader(io.StringIO(z.read(name["lastfm_asia_target.csv"]).decode("utf-8")))}
    feats = json.loads(z.read(name["lastfm_asia_features.json"]).decode("utf-8"))
    n = len(target)
    assert sorted(target) == list(range(n)) and sorted(map(int, feats)) == list(range(n))
    y = np.array([target[i] for i in range(n)], np.int64)
    dim = 1 + max(max(v) for v in feats.values() if v)
    x = np.zeros((n, dim), np.float32)
    for k, v in feats.items():
        x[int(k), v] = 1.0
    adj = symmetric_binary(edges[:, 0], edges[:, 1], n)
    save(out, "lastfm_asia", adj, x, y, {
        "source_url": "https://snap.stanford.edu/data/lastfm_asia.zip",
        "features": "binary indicators of liked artists (lastfm_asia_features.json)",
        "raw_edges": int(len(edges))})


def dblp(npz_path: Path, out: Path):
    f = np.load(npz_path, allow_pickle=False)     # object arrays are skipped: only numeric keys are read
    a = sp.csr_matrix((f["adj_data"], f["adj_indices"], f["adj_indptr"]), shape=tuple(f["adj_shape"]))
    x = sp.csr_matrix((f["attr_data"], f["attr_indices"], f["attr_indptr"]), shape=tuple(f["attr_shape"])).toarray()
    x = (x > 0).astype(np.float32)                # PyG CitationFull: x[x > 0] = 1
    y = f["labels"].astype(np.int64)
    coo = a.tocoo()
    adj = symmetric_binary(coo.row, coo.col, a.shape[0])
    save(out, "dblp", adj, x, y, {
        "source_url": "https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz",
        "features": "binary bag-of-words (attr_* arrays, x > 0), as torch_geometric CitationFull('DBLP')",
        "raw_adj_nnz": int(a.nnz)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lastfm-zip", type=Path)
    p.add_argument("--dblp-npz", type=Path)
    p.add_argument("--out-root", type=Path, required=True)
    args = p.parse_args()
    if args.lastfm_zip:
        lastfm(args.lastfm_zip, args.out_root / "lastfm_asia")
    if args.dblp_npz:
        dblp(args.dblp_npz, args.out_root / "dblp")


if __name__ == "__main__":
    main()
