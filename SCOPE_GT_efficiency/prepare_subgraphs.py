"""Shared, label-blind nested induced subgraphs; outside method timers."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from SCOPE_GT_budgets.common import dataset_manifest, write_json

DEFAULT_SIZES = ['2000','5000','10000','full']
DATASET_SIZES = dict(pubmed=DEFAULT_SIZES,arxiv_year=['10000','30000','60000','full'])


def split_ids(root, n):
    result = {}
    for name in ('train','val','test'):
        x = np.load(root / f'{name}.npy', allow_pickle=False)
        x = np.flatnonzero(x) if x.dtype == bool else x.astype(np.int64).reshape(-1)
        if not len(x) or x.min() < 0 or x.max() >= n:
            raise ValueError(f'Invalid {name} IDs')
        result[name] = x
    all_ids = np.concatenate(list(result.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError('Duplicate or overlapping splits')
    return result


def prepare(source, out, sizes=DEFAULT_SIZES, seed=2027):
    source, out = Path(source).resolve(), Path(out).resolve()
    fingerprint = dataset_manifest(source)
    adj = sparse.load_npz(source / 'adj.npz').tocsr()
    n = adj.shape[0]
    features = np.load(source / 'feat.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(source / 'label.npy', allow_pickle=False).reshape(-1)
    if adj.shape != (n,n) or len(features) != n or len(labels) != n:
        raise ValueError('Input graph/feature/label shapes disagree')
    splits = split_ids(source,n)
    requested = sorted(set(n if x == 'full' else int(x) for x in sizes))
    if not requested or min(requested) < 1 or max(requested) > n:
        raise ValueError(f'Sizes must lie in 1..{n}')
    # Merge balanced per-stratum priorities; every graph is a prefix of ONE order.
    covered = np.zeros(n,dtype=bool)
    covered[np.concatenate(list(splits.values()))] = True
    strata = list(splits.values()) + [np.flatnonzero(~covered)]
    rng = np.random.default_rng(seed)
    ordered, priority = [], []
    for ids in strata:
        if len(ids):
            ordered.append(rng.permutation(ids))
            priority.append((np.arange(len(ids))+.5)/len(ids))
    order = np.concatenate(ordered)[np.argsort(np.concatenate(priority),kind='stable')]
    graphs = []
    for count in requested:
        folder = out / f'N{count}'
        spec = dict(source_sha256=fingerprint, chain_seed=seed, n=count,
                    sampling='one split-stratified label-blind prefix chain; induced edges')
        marker = folder / 'graph.json'
        if marker.exists():
            old = json.loads(marker.read_text(encoding='utf-8'))
            if old['request'] != spec or dataset_manifest(folder) != old['data_sha256']:
                raise ValueError(f'Existing subgraph identity changed: {folder}; choose a fresh output')
            graphs.append(dict(n=count,root=str(folder),manifest=old))
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'Incomplete output requires inspection: {folder}')
        ids = np.sort(order[:count])
        inverse = np.full(n,-1,dtype=np.int64)
        inverse[ids] = np.arange(count)
        local_splits = {k:inverse[v][inverse[v]>=0] for k,v in splits.items()}
        if any(not len(x) for x in local_splits.values()):
            raise ValueError(f'Empty split at N={count}; do not silently resample')
        if not np.array_equal(np.unique(labels[ids][labels[ids]>=0]), np.unique(labels[labels>=0])):
            raise ValueError(f'Missing class at N={count}; choose sizes before running methods')
        if not np.array_equal(np.unique(labels[ids][local_splits['train']]), np.unique(labels[labels>=0])):
            raise ValueError(f'Training split lacks a class at N={count}')
        sub = adj[ids][:,ids].tocsr()
        sub.eliminate_zeros()
        folder.mkdir(parents=True,exist_ok=True)
        if count == n:
            # Preserve the original full-graph files (including masks and dtype) exactly.
            for filename in ('adj.npz','feat.npy','label.npy','train.npy','val.npy','test.npy'):
                shutil.copy2(source/filename,folder/filename)
        else:
            sparse.save_npz(folder/'adj.npz',sub)
            np.save(folder/'feat.npy',np.asarray(features[ids]))
            np.save(folder/'label.npy',labels[ids])
            for name,x in local_splits.items():
                np.save(folder/f'{name}.npy',x)
        np.save(folder/'node_ids.npy',ids)
        undirected = sub.maximum(sub.T)
        undirected.setdiag(0); undirected.eliminate_zeros()
        degree = np.diff(undirected.indptr)
        components, assignment = connected_components(undirected,directed=False)
        stats = dict(n=count, stored_entries=int(sub.nnz), undirected_edges=int(undirected.nnz//2),
            mean_undirected_degree=float(degree.mean()), isolated_fraction=float(np.mean(degree==0)),
            connected_components=int(components), largest_component=int(np.bincount(assignment).max()),
            split_sizes={k:len(x) for k,x in local_splits.items()},
            class_counts={str(k):int(v) for k,v in zip(*np.unique(labels[ids],return_counts=True))})
        manifest = dict(request=spec,statistics=stats,data_sha256=dataset_manifest(folder))
        write_json(marker,manifest)
        graphs.append(dict(n=count,root=str(folder),manifest=manifest))
        print(f'Prepared N={count}, stored edges={sub.nnz}',flush=True)
    write_json(out/'graphs.json',dict(source=str(source), chain_seed=seed,graphs=graphs))
    return graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sizes',nargs='+',default=DEFAULT_SIZES)
    p.add_argument('--graph-seed',type=int,default=2027)
    a=p.parse_args()
    prepare(a.source,a.out,a.sizes,a.graph_seed)


if __name__=='__main__':
    main()
