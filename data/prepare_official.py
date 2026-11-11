#!/usr/bin/env python3
"""Preserve Platonov's ten fixed masks or the official OGB Products split.

No random splits are generated. Inputs must already be downloaded. Output must
be a new or empty directory, so existing experimental datasets are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

PLATONOV_DATASETS = ('minesweeper', 'roman_empire', 'amazon_ratings', 'tolokers')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parts(parts, n):
    joined = []
    for name in ('train', 'val', 'test'):
        ids = np.asarray(parts[name])
        if not np.issubdtype(ids.dtype, np.integer) or ids.ndim != 1 or not len(ids):
            raise ValueError(f'{name}: expected nonempty integer node IDs')
        if ids.min() < 0 or ids.max() >= n or len(np.unique(ids)) != len(ids):
            raise ValueError(f'{name}: duplicate or out-of-range node IDs')
        joined.append(ids)
    all_ids = np.concatenate(joined)
    if len(all_ids) != n or len(np.unique(all_ids)) != n:
        raise ValueError('Official split must partition all nodes without overlap')


def write_dataset(out, dataset, features, labels, edges, split_sets, metadata):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {out}')
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    edges = np.asarray(edges)
    n = len(labels)
    if features.ndim != 2 or features.shape[0] != n or not np.isfinite(features).all():
        raise ValueError('Invalid node features')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Official labels must be integers')
    labels = labels.astype(np.int64)
    if not np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))):
        raise ValueError('Expected contiguous official labels 0..C-1')
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or not len(edges) or edges.min() < 0 or edges.max() >= n):
        raise ValueError('Expected integer edge pairs with shape (E, 2)')
    for parts in split_sets.values():
        validate_parts(parts, n)
    adjacency = sparse.coo_matrix((np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])),
                                  shape=(n, n)).tocsr()
    adjacency.data.fill(1)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / 'adj.npz', adjacency)
    np.save(out / 'feat.npy', features)
    np.save(out / 'label.npy', labels)
    hashes = {}
    for suffix, parts in split_sets.items():
        for name, ids in parts.items():
            filename = f'{name}{suffix}.npy'
            np.save(out / filename, np.asarray(ids, dtype=np.int64))
            hashes[filename] = sha256(out / filename)
    metadata = {'dataset': dataset, **metadata, 'num_nodes': n,
                'num_features': features.shape[1], 'num_classes': int(labels.max()) + 1,
                'num_edges_undirected': adjacency.nnz // 2, 'make_undirected': True,
                'split_sha256': hashes,
                'split_sizes': {suffix: {k: len(v) for k, v in parts.items()}
                                for suffix, parts in split_sets.items()}}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'dataset': dataset, 'output': str(out.resolve()),
                      'split_policy': metadata['split_policy'], 'splits': list(split_sets)}, indent=2))


def platonov(dataset, raw_npz, out):
    with np.load(raw_npz, allow_pickle=False) as raw:
        features, labels, edges = raw['node_features'], raw['node_labels'], raw['edges']
        n = len(labels)
        masks = {}
        for name in ('train', 'val', 'test'):
            mask = raw[name + '_masks']
            if mask.shape != (10, n) or not np.isin(mask, (False, True)).all():
                raise ValueError(f'{name}_masks must contain ten binary masks with shape (10, {n})')
            masks[name] = mask.astype(bool)
        splits = {f'_split{i}': {name: np.flatnonzero(mask[i]) for name, mask in masks.items()}
                  for i in range(10)}
        for parts in splits.values():
            sizes = np.array([len(parts[k]) for k in ('train', 'val', 'test')])
            if np.any(np.abs(sizes - n * np.array([.5, .25, .25])) > 2):
                raise ValueError('Expected the official approximately 50/25/25 masks')
        # The ordinary six-file interface is an exact alias of official split 0.
        splits[''] = splits['_split0']
        write_dataset(out, dataset, features, labels, edges, splits,
                      {'split_policy': 'platonov_official_10_fixed_50_25_25',
                       'source_url': 'https://github.com/yandex-research/heterophilous-graphs',
                       'source_npz_sha256': sha256(raw_npz),
                       'official_split_ids': list(range(10)), 'default_official_split_id': 0})


def products(ogb_root, out):
    if Path(out).exists() and any(Path(out).iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {out}')
    try:
        import ogb
        import ogb.nodeproppred.dataset as ogb_dataset
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as error:
        raise SystemExit('Products conversion requires the optional package: pip install ogb') from error
    root = Path(ogb_root) / 'ogbn_products'
    # Stop before the upstream loader can prompt for an update or download.
    with Path(ogb_dataset.__file__).with_name('master.csv').open(newline='', encoding='utf-8') as handle:
        version = next(row['ogbn-products'] for row in csv.DictReader(handle)
                       if next(iter(row.values())) == 'version')
    if not (root / f'RELEASE_v{version}.txt').is_file():
        raise FileNotFoundError(f'Download the official OGB release first: {root / ("RELEASE_v" + version + ".txt")}')
    for name in ('train', 'valid', 'test'):
        path = root / 'split' / 'sales_ranking' / f'{name}.csv.gz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing official split: {path}')
    if not ((root / 'raw' / 'edge.csv.gz').is_file() or (root / 'processed' / 'data_processed').is_file()):
        raise FileNotFoundError(f'Missing downloaded OGB graph under {root}')
    data = NodePropPredDataset(name='ogbn-products', root=str(ogb_root))
    graph, labels = data[0]
    original = data.get_idx_split()
    parts = {name: np.asarray(original[key], dtype=np.int64)
             for name, key in (('train', 'train'), ('val', 'valid'), ('test', 'test'))}
    # OGB may prefer a cached split_dict.pt; verify it agrees with the release CSVs.
    for name, key in (('train', 'train'), ('val', 'valid'), ('test', 'test')):
        official_ids = np.loadtxt(root / 'split' / 'sales_ranking' / f'{key}.csv.gz', dtype=np.int64, ndmin=1)
        if not np.array_equal(parts[name], official_ids):
            raise ValueError(f'OGB cached {name} split disagrees with the official release CSV')
    write_dataset(out, 'ogbn_products', graph['node_feat'], labels, graph['edge_index'].T, {'': parts},
                  {'split_policy': 'ogb_official_sales_ranking', 'ogb_version': ogb.__version__,
                   'source_url': 'https://ogb.stanford.edu/docs/nodeprop/#ogbn-products',
                   'official_split_source_sha256': {name: sha256(root / 'split' / 'sales_ranking' / f'{key}.csv.gz')
                                                    for name, key in (('train', 'train'), ('val', 'valid'), ('test', 'test'))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='source', required=True)
    p = sub.add_parser('platonov')
    p.add_argument('dataset', choices=PLATONOV_DATASETS)
    p.add_argument('--raw-npz', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p = sub.add_parser('products')
    p.add_argument('--ogb-root', type=Path, required=True, help='parent folder containing downloaded ogbn_products/')
    p.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.source == 'platonov':
        platonov(args.dataset, args.raw_npz, args.out_dir)
    else:
        products(args.ogb_root, args.out_dir)


if __name__ == '__main__':
    main()
