"""Offline conversion of the LINKX benchmarks arxiv-year and pokec into the NPZ dataset layout.

No network access is needed: download the raw files on a machine with internet (URLs in RAW_FILES,
or run `python prepare_linkx.py --list-urls`), copy them into one folder, then run
    python prepare_linkx.py arxiv-year --raw-dir /path/to/raw
    python prepare_linkx.py pokec      --raw-dir /path/to/raw
Every raw file is checked for presence, size and SHA-256 before anything is converted.

Output directory (default <this folder>/<dataset>, same layout as datasets/wikics):
    adj.npz             scipy CSR, float32, binary, no self loops (symmetric unless --keep-direction)
    feat.npy            float32 [N, F]
    label.npy           int64 [N], contiguous 0..C-1, -1 = unlabeled (pokec only)
    train.npy / val.npy / test.npy          int64 node ids of LINKX fixed split 0
    train_k.npy / val_k.npy / test_k.npy    all 5 LINKX fixed splits (use --split-suffix _k)
    label_mapping.json  metadata.json

Sources (Lim et al., "Large Scale Learning on Non-Homophilous Graphs", NeurIPS 2021,
https://github.com/CUAI/Non-Homophily-Large-Scale):
    arxiv-year  graph + features + node_year from OGB ogbn-arxiv; label = year binned into 5
                even quantiles (LINKX even_quantile_labels); 5 fixed 50/25/25 splits from the LINKX repo.
    pokec       LINKX's own pokec.mat (Google Drive id 1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y) has been deleted
                (404, LINKX issue #20); its 5 fixed 50/25/25 splits are still downloadable. If you have a
                copy of pokec.mat, put it in --raw-dir and it is used as is. Otherwise the graph is rebuilt
                from SNAP soc-Pokec: identical nodes (user_id - 1), directed edges and gender labels (the
                163 empty profiles are exactly the nodes missing from the LINKX splits; directed edge
                homophily 0.445 as reported by LINKX), but LINKX never published its 65-dim feature code,
                so feat.npy is our own 19-dim numeric profile encoding (see load_pokec_snap).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy import sparse

# file name in --raw-dir: (download URL, size in bytes, sha256)
RAW_FILES = {
    "arxiv-year": {
        "arxiv.zip": (
            "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
            83058288, "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276"),
        "arxiv-year-splits.npy": (
            "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/arxiv-year-splits.npy",
            3579056, "a2d4c8b2ed621270642dcf4886c3f6e425da664b2d31c7a28fb11a6da763077a"),
    },
    "pokec": {
        "pokec-splits.npy": (
            "https://drive.usercontent.google.com/download?id=1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_&export=download&confirm=t",
            40176121, "2a6ceb92307ee911daa3428dc2491d83b5f7deefd7faa2dfe502b8f7c2947d1b"),
        "soc-pokec-relationships.txt.gz": (
            "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
            132454730, "1a23e0ec8a4e497752125f6b3f01696fea7fcdb696fa61d1e822faf4d0d69b14"),
        "soc-pokec-profiles.txt.gz": (
            "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
            435512787, "d4c6b0151cfa3571a7864ea12b06a7c5367e33ecbfe20358cf5420d76e40bad9"),
    },
}
POKEC_MAPPING = {"0 (female)": 0, "1 (male)": 1, "null (empty profile)": -1}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_raw(dataset, raw_dir, verify):
    """Fail early, with the download URL, on any missing / truncated / wrong raw file."""
    needed = dict(RAW_FILES[dataset])
    if dataset == "pokec" and (raw_dir / "pokec.mat").exists():
        needed = {"pokec-splits.npy": needed["pokec-splits.npy"]}  # SNAP files not needed
    problems = []
    for name, (url, size, sha) in needed.items():
        path = raw_dir / name
        if not path.exists():
            problems.append(f"missing  {name}\n         download: {url}")
            continue
        with path.open("rb") as f:
            if f.read(15).lower().lstrip().startswith((b"<!doctype html", b"<html")):
                problems.append(f"HTML     {name} is a web page, not the data file\n         download: {url}")
                continue
        if path.stat().st_size != size:
            problems.append(f"size     {name}: {path.stat().st_size} bytes, expected {size} (incomplete download?)"
                            f"\n         download: {url}")
            continue
        if verify:
            print("sha256", name, "...", flush=True)
            if sha256_file(path) != sha:
                problems.append(f"sha256   {name} does not match {sha} (rerun with --no-verify to skip)")
    if problems:
        sys.exit(f"raw files in {raw_dir.resolve()} are not ready:\n  " + "\n  ".join(problems))
    return needed


def read_table(f, dtype, sep=","):
    """Headerless numeric table from a binary file object (pandas if installed, else pure numpy)."""
    try:
        import pandas as pd
    except ImportError:
        return np.loadtxt(io.TextIOWrapper(f, encoding="ascii"), dtype=dtype, delimiter=sep, ndmin=2)
    return pd.read_csv(f, header=None, sep=sep, dtype=dtype).to_numpy()


def even_quantile_labels(vals, nclasses):
    """LINKX data_utils.even_quantile_labels, with np.int replaced by np.int64."""
    label = -np.ones(vals.shape[0], dtype=np.int64)
    intervals = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        intervals.append((lower, upper))
        label[(vals >= lower) * (vals < upper)] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    intervals.append((lower, np.inf))
    return label, intervals


def load_arxiv_year(raw_dir, nclasses=5):
    with zipfile.ZipFile(raw_dir / "arxiv.zip") as zf:
        def csv(member, dtype):
            with zf.open(member) as raw, gzip.open(raw) as f:
                return read_table(f, dtype)
        edge_index = csv("arxiv/raw/edge.csv.gz", np.int64).T
        feat = csv("arxiv/raw/node-feat.csv.gz", np.float32)
        year = csv("arxiv/raw/node_year.csv.gz", np.int64).reshape(-1)
    label, intervals = even_quantile_labels(year, nclasses)
    mapping = {f"year in [{lo:g}, {hi:g})": k for k, (lo, hi) in enumerate(intervals)}
    return edge_index, feat, label, mapping, {"label_meaning": "publication year, 5 even-quantile bins"}


def load_pokec_mat(raw_dir):
    mat = sio.loadmat(raw_dir / "pokec.mat")
    edge_index = np.asarray(mat["edge_index"], dtype=np.int64)
    feat = np.asarray(mat["node_feat"], dtype=np.float32)
    label = np.asarray(mat["label"]).reshape(-1).astype(np.int64)
    assert int(np.asarray(mat["num_nodes"]).item()) == feat.shape[0]
    return edge_index, feat, label, POKEC_MAPPING, {"label_meaning": "gender", "pokec_source": "LINKX pokec.mat"}


def _years(col):
    """'2012-05-25 11:20:00.0' -> fractional year; 'null' -> nan."""
    t = np.array([v[:19] if v != "null" else "NaT" for v in col], dtype="datetime64[s]")
    return 1970.0 + t.astype(np.float64) / (365.2425 * 86400) + np.where(np.isnat(t), np.nan, 0.0)


def load_pokec_snap(raw_dir):
    """Rebuild pokec from SNAP soc-Pokec. Node i = user_id i+1, arc u->v = "u has friend v".

    feat.npy, every column in [0, 1], a missing value becomes 0 plus an indicator column:
        public, completion_percentage/100, age/100, age_missing,
        registration, last_login (fractional year, min-max scaled), time_missing,
        one-hot of the top-level region (text before the first comma: the 8 Slovak kraje,
        "ceska republika", "zahranicie", ...; regions with < 100 users go to region=other), region_missing.
    gender is the label and is never used as a feature.
    """
    with gzip.open(raw_dir / "soc-pokec-relationships.txt.gz") as f:
        edge_index = read_table(f, np.int64, sep="\t").T - 1
    cols = [[] for _ in range(8)]  # user_id public completion gender region last_login registration AGE
    with gzip.open(raw_dir / "soc-pokec-profiles.txt.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            for k, v in enumerate(line.rstrip("\n").split("\t", 8)[:8]):
                cols[k].append(v)
    ids = np.asarray(cols[0], dtype=np.int64) - 1
    n = len(ids)
    assert np.array_equal(np.sort(ids), np.arange(n)), "user ids must be 1..N"
    order = np.argsort(ids)
    public, completion, gender, region, last_login, registration, age = (np.asarray(c)[order] for c in cols[1:])

    def num(col):
        return np.array([float(v) if v != "null" else np.nan for v in col])

    label = np.nan_to_num(num(gender), nan=-1).astype(np.int64)
    age_v = num(age)
    age_missing = np.isnan(age_v) | (age_v <= 0)
    times = np.stack([_years(registration), _years(last_login)], 1)
    time_missing = np.isnan(times).any(1)
    lo, hi = np.nanmin(times), np.nanmax(times)
    times = np.nan_to_num((times - lo) / (hi - lo), nan=0.0)

    top = np.array([r.split(",")[0].strip() if r != "null" else "" for r in region])
    names, counts = np.unique(top[top != ""], return_counts=True)
    names = sorted(names[counts >= 100].tolist())
    col_of = {r: k for k, r in enumerate(names)}
    region_col = np.array([col_of.get(r, len(names)) if r else len(names) + 1 for r in top])
    region_oh = np.zeros((n, len(names) + 2), dtype=np.float32)
    region_oh[np.arange(n), region_col] = 1.0

    feat = np.column_stack([
        np.nan_to_num(num(public), nan=0.0),
        np.nan_to_num(num(completion), nan=0.0) / 100.0,
        np.where(age_missing, 0.0, np.clip(np.nan_to_num(age_v), 0, 100) / 100.0),
        age_missing,
        times,
        time_missing,
        region_oh,
    ]).astype(np.float32)
    columns = (["public", "completion_percentage", "age", "age_missing", "registration", "last_login",
                "time_missing"] + [f"region={r}" for r in names] + ["region=other", "region_missing"])
    extra = {"label_meaning": "gender",
             "pokec_source": "SNAP soc-Pokec rebuild; graph/labels as LINKX, features are NOT the LINKX 65-dim ones",
             "feature_columns": columns, "time_scale_years": [float(lo), float(hi)]}
    return edge_index, feat, label, POKEC_MAPPING, extra


def build_adj(edge_index, n, keep_direction):
    row, col = edge_index
    keep = row != col
    row, col = row[keep], col[keep]
    if not keep_direction:
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
    adj = sparse.csr_matrix((np.ones(len(row), np.float32), (row, col)), shape=(n, n))
    adj.data[:] = 1.0  # duplicates were summed
    adj.sort_indices()
    return adj


def load_pickled_npy(path):
    """The LINKX split files are pickles written by numpy 1.x, which reference numpy.core.*; numpy 2 moved
    that to numpy._core and only keeps a shim, so alias the old names if the shim is missing."""
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError:
        import importlib
        for name in ("multiarray", "numeric", "_multiarray_umath"):
            sys.modules.setdefault(f"numpy.core.{name}", importlib.import_module(f"numpy._core.{name}"))
        return np.load(path, allow_pickle=True)


def load_splits(path, labels):
    splits = []
    for d in load_pickled_npy(path):
        s = {}
        for ours, theirs in (("train", "train"), ("val", "valid"), ("test", "test")):
            ids = np.asarray(d[theirs]).astype(np.int64).reshape(-1)
            assert (labels[ids] >= 0).all(), f"{theirs} split contains unlabeled nodes"
            s[ours] = ids
        joined = np.concatenate(list(s.values()))
        assert len(np.unique(joined)) == len(joined), "splits overlap"
        splits.append(s)
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", choices=sorted(RAW_FILES))
    p.add_argument("--raw-dir", type=Path, default=Path("raw"), help="folder holding the downloaded raw files")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <this folder>/<dataset>")
    p.add_argument("--keep-direction", action="store_true",
                   help="store the original directed arcs instead of the symmetrised graph (wikics is symmetrised)")
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check (size is still checked)")
    p.add_argument("--list-urls", action="store_true", help="print the raw files to download and exit")
    args = p.parse_args()
    if args.list_urls or args.dataset is None:
        for ds, files in RAW_FILES.items():
            print(ds)
            for name, (url, size, sha) in files.items():
                print(f"  {name:32s} {size:>11,d} B  {url}\n  {'':32s} sha256 {sha}")
        return
    out = args.out_dir or Path(__file__).resolve().parent / args.dataset

    needed = check_raw(args.dataset, args.raw_dir, not args.no_verify)
    if args.dataset == "arxiv-year":
        edge_index, feat, label, mapping, extra = load_arxiv_year(args.raw_dir)
    elif (args.raw_dir / "pokec.mat").exists():
        edge_index, feat, label, mapping, extra = load_pokec_mat(args.raw_dir)
    else:
        print("pokec.mat not in --raw-dir: rebuilding pokec from SNAP soc-Pokec (features differ from LINKX)")
        edge_index, feat, label, mapping, extra = load_pokec_snap(args.raw_dir)
    split_file = args.raw_dir / f"{args.dataset}-splits.npy"

    n = feat.shape[0]
    assert label.shape == (n,) and edge_index.shape[0] == 2 and edge_index.min() >= 0 and edge_index.max() < n
    classes = np.unique(label[label >= 0])
    assert np.array_equal(classes, np.arange(len(classes))), classes
    adj = build_adj(edge_index, n, args.keep_direction)
    splits = load_splits(split_file, label)

    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "adj.npz", adj)
    np.save(out / "feat.npy", feat)
    np.save(out / "label.npy", label)
    for k, s in enumerate(splits):
        for name, ids in s.items():
            np.save(out / f"{name}_{k}.npy", ids)
            if k == 0:
                np.save(out / f"{name}.npy", ids)
    (out / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    meta = {
        "dataset": args.dataset,
        "source_url": {k: v[0] for k, v in needed.items()},
        "format": "adj.npz, feat.npy, label.npy, train.npy, val.npy, test.npy",
        "num_nodes": int(n),
        "num_raw_edges": int(edge_index.shape[1]),
        ("num_edges_directed_nnz" if args.keep_direction else "num_edges_undirected_nnz"): int(adj.nnz),
        "num_features": int(feat.shape[1]),
        "num_classes": int(len(classes)),
        "num_unlabeled": int((label < 0).sum()),
        "feature_dtype": "float32",
        "split_policy": f"linkx_fixed_splits: train/val/test.npy = split 0, "
                        f"train_k/val_k/test_k.npy = split k (k=0..{len(splits) - 1})",
        "num_train": int(len(splits[0]["train"])),
        "num_val": int(len(splits[0]["val"])),
        "num_test": int(len(splits[0]["test"])),
        "num_splits": len(splits),
        "make_undirected": not args.keep_direction,
        "offline": True,
        **extra,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out.resolve())
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_columns"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
