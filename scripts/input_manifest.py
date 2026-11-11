"""Identify processed graph inputs without loading a model or requiring SciPy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest(root, split_suffix=''):
    root = Path(root).resolve()
    names = ['adj.npz', 'feat.npy', 'label.npy'] + [f'{s}{split_suffix}.npy' for s in ('train', 'val', 'test')]
    files = {}
    for name in names:
        path = root / name
        files[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    features = np.load(root / 'feat.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root / 'label.npy', mmap_mode='r', allow_pickle=False)
    n = len(labels)
    if len(features) != n:
        raise ValueError('Feature/label row counts differ')
    for name, a in [('feat.npy', features), ('label.npy', labels)]:
        files[name].update(shape=list(a.shape), dtype=str(a.dtype))
    with np.load(root / 'adj.npz', allow_pickle=False) as graph:
        shape = graph['shape'].tolist()
        if shape != [n, n]:
            raise ValueError('Adjacency shape does not match labels')
        files['adj.npz'].update(shape=shape, stored_entries=len(graph['data']),
                                format=str(graph['format'].item()))
    split_ids = []
    for split in ('train', 'val', 'test'):
        name = f'{split}{split_suffix}.npy'
        a = np.load(root / name, allow_pickle=False)
        if a.dtype == bool:
            if a.shape != (n,):
                raise ValueError(f'{name}: boolean mask must have shape ({n},)')
            ids = np.flatnonzero(a)
        elif np.issubdtype(a.dtype, np.integer):
            ids = a.reshape(-1)
        else:
            raise ValueError(f'{name}: expected integer IDs or a boolean mask')
        if not len(ids) or ids.min() < 0 or ids.max() >= n or len(np.unique(ids)) != len(ids):
            raise ValueError(f'{name}: empty, duplicate or out-of-range IDs')
        files[name].update(shape=list(a.shape), dtype=str(a.dtype), centers=len(ids))
        split_ids.append(ids)
    joined = np.concatenate(split_ids)
    if len(np.unique(joined)) != len(joined):
        raise ValueError('Train/validation/test splits overlap')
    fingerprint = hashlib.sha256(json.dumps({k: v['sha256'] for k, v in files.items()}, sort_keys=True).encode()).hexdigest()
    return dict(schema=1, data_root=str(root), input_id=fingerprint, files=files,
                split_suffix=split_suffix, identity_scope='Exact processed file bytes, before method-specific transforms')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split-suffix', default='')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    value = json.dumps(manifest(a.data_root, a.split_suffix), indent=2, ensure_ascii=False) + '\n'
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(value, encoding='utf-8')
    else:
        print(value, end='')


if __name__ == '__main__':
    main()
