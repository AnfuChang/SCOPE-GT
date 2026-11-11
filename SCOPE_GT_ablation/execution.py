"""Center-query execution layouts of the same SCOPE-GT reader.

  sequential    process one sequence at a time, accumulate the logical-batch gradient

  padded        no cross-center sharing: each center has its own node pool (deduplicated only inside its
                sequence and evidence), so nodes shared by centers are encoded and projected again;
                the logical batch is padded to its longest sequence
  grouped       same per-center pools, split into groups of exactly equal length 1 + |S_c| (no padding);
                each group runs forward/backward on its own
  dedup_padded  one node table per logical batch (shared encoding and base K/V rows), padded layout
  dedup_packed  shared table and only valid (center, context) pairs with a segment softmax (final method)

Path and background records are sliced with their pairs; their number is not the token length.
"""
from __future__ import annotations

import numpy as np

from SCOPE_GT.batch import DenseBatch, PackedBatch
from SCOPE_GT.sampler import CTX_KEYS

LAYOUTS = ('sequential', 'padded', 'grouped', 'dedup_packed', 'dedup_padded', 'packed_unshared', 'shared_encoder')
STUDY_LAYOUTS = ('sequential', 'padded', 'grouped', 'dedup_packed')
RECORDS = {'moment': ('cols', 'nodes', 'weight', 'position', 'external'), 'context_bg': ('cols', 'nodes')}


def slice_draw(draw, rows):
    """Keep center rows `rows` (in that order) and remap their flat evidence records."""
    size = len(draw['ctx_valid'])
    rows = np.asarray(rows, np.int64)
    if len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= size).any():
        raise ValueError('rows must be unique center rows in range')
    remap = np.full(size, -1, np.int64)
    remap[rows] = np.arange(len(rows))
    out = dict(draw)
    for key in CTX_KEYS:
        out[key] = draw[key][rows]
    for prefix, fields in RECORDS.items():
        keep = remap[draw[prefix + '_rows']] >= 0
        out[prefix + '_rows'] = remap[draw[prefix + '_rows'][keep]]
        for field in fields:
            out[f'{prefix}_{field}'] = draw[f'{prefix}_{field}'][keep]
    # selected contexts are left-packed per row, so cropping to the longest kept row loses nothing
    width = max(int(out['ctx_valid'].sum(1).max()) if len(rows) else 0, 1)
    for key in CTX_KEYS:
        out[key] = out[key][:, :width]
    return out


def length_groups(draw):
    """Center rows grouped by exact sequence length (number of valid contexts)."""
    lengths = draw['ctx_valid'].sum(1)
    return [np.flatnonzero(lengths == length) for length in np.unique(lengths)]


def execution_rows(layout, draws):
    """Units of one logical batch; exact lengths in EVERY view for Grouped.

    Using a tuple of view lengths lets the multi-view consistency loss be
    differentiated within a unit and its graph released immediately afterwards.
    """
    n = len(draws[0]['ctx_valid'])
    if any(len(d['ctx_valid']) != n for d in draws):
        raise ValueError('views must contain the same prediction centers')
    if layout == 'sequential':
        return [np.asarray([i], dtype=np.int64) for i in range(n)]
    if layout == 'grouped':
        lengths = np.column_stack([d['ctx_valid'].sum(1) for d in draws])
        _, groups = np.unique(lengths, axis=0, return_inverse=True)
        return [np.flatnonzero(groups == i) for i in range(int(groups.max()) + 1)]
    if layout in ('padded', 'dedup_padded', 'dedup_packed', 'packed_unshared', 'shared_encoder'):
        return [np.arange(n)]
    raise ValueError(f'unknown layout: {layout}')


def make_unit_views(layout, centers, draws, features, rows):
    """Build one unit on CPU, retaining all pair/evidence rows for each view."""
    scope = 'batch' if layout in ('dedup_padded', 'dedup_packed', 'shared_encoder') else 'center'
    batch_type = PackedBatch if layout in ('dedup_packed', 'packed_unshared', 'shared_encoder') else DenseBatch
    batches = [batch_type(np.asarray(centers)[rows], slice_draw(draw, rows), features, 'cpu', scope=scope)
               for draw in draws]
    for batch in batches:
        batch.kv_per_read = layout in ('packed_unshared', 'shared_encoder')
    return batches


def make_units(layout, centers, draw, features, device='cpu'):
    """[(rows of the logical batch, batch)]: one unit, or one per exact-length group for `grouped`."""
    centers = np.asarray(centers, np.int64)
    every = np.arange(len(centers))
    if layout in ('packed_unshared', 'shared_encoder'):
        return [(every, make_unit_views(layout, centers, [draw], features, every)[0].to(device))]
    if layout == 'sequential':
        return [(rows, make_unit_views(layout, centers, [draw], features, rows)[0])
                for rows in execution_rows(layout, [draw])]
    if layout == 'padded':
        return [(every, DenseBatch(centers, draw, features, device, scope='center'))]
    if layout == 'grouped':
        return [(rows, DenseBatch(centers[rows], slice_draw(draw, rows), features, device, scope='center'))
                for rows in length_groups(draw)]
    if layout == 'dedup_padded':
        return [(every, DenseBatch(centers, draw, features, device, scope='batch'))]
    if layout == 'dedup_packed':
        return [(every, PackedBatch(centers, draw, features, device, scope='batch'))]
    raise ValueError(f'unknown layout {layout!r}; choose from {LAYOUTS}')


def unit_stats(units):
    """Diagnostics of one logical batch in one layout."""
    pairs = sum(b.pairs for _, b in units)
    slots = sum(getattr(b, 'slots', b.pairs) for _, b in units)
    encoded = sum(b.encoded_rows for _, b in units)
    unique = len(np.unique(np.concatenate([b.pool_ids for _, b in units])))
    centers = sum(b.size for _, b in units)
    return {'groups': len(units), 'centers': centers, 'pairs': pairs, 'slots': slots,
            'pad_fraction': 1. - pairs / max(slots, 1), 'encoded_rows': encoded, 'unique_nodes': unique,
            'occurrence_ratio': encoded / max(unique, 1), 'kv_rows': sum(b.size + b.pairs if getattr(b, 'kv_per_read', False) else b.kv_count for _, b in units),
            'path_records': sum(int(b.moment_owner.numel()) for _, b in units),
            'background_records': sum(int(b.context_bg_owner.numel()) for _, b in units),
            'token_length_mean': 1. + pairs / max(centers, 1),
            'token_length_max': 1 + max(int(np.bincount(b.pair_rows.cpu().numpy(), minlength=b.size).max()) for _, b in units),
            'padded_width': max((1 + b.slots // b.size) if hasattr(b, 'slots') else 0 for _, b in units)}
