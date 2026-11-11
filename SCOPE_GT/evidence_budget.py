"""Bound C attribute records after exact T/C statistics and B filtering.

The zero-budget/on path uses ContextSampler directly in train.make_sampler.
This adapter matches the independently tested tuning sampler, including its
per-center random keys and the scalar-only position-presence masks.
"""
import numpy as np

from .graph import CenterStream
from .sampler import ContextSampler


class EvidenceBudgetSampler(ContextSampler):
    def __init__(self, *args, record_budget=0, use_attributes=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= record_budget <= 64:
            raise ValueError('C record budget must be 0 (full) or 1..64 per pair/position')
        self.record_budget, self.use_attributes = int(record_budget), bool(use_attributes)

    @property
    def cache_signature(self):
        return (*super().cache_signature, 'attribute_pps_v1', self.record_budget, self.use_attributes)

    def relations(self, centers, sample, streams, relation_source=None, common_cache=None, acc=None):
        out = super().relations(centers, sample, streams, relation_source, common_cache, acc)
        out['full_witness_records'] = len(out['moment_nodes'])
        out['attribute_sampled_groups'] = 0
        limit = self.record_budget if self.use_attributes else 1
        if not limit or not len(out['moment_nodes']):
            return out
        keys = ((out['moment_rows'] * out['ctx'].shape[1] + out['moment_cols']) * 3
                + out['moment_position'])
        order = np.argsort(keys, kind='stable')
        ordered = keys[order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], len(order)]
        keep, weights = [], []
        stream = CenterStream(int(streams['select'].key), 28391)
        centers = np.asarray(centers, np.int64)
        for start, end in zip(starts, ends):
            index = order[start:end]
            if len(index) <= limit:
                keep.append(index)
                weights.append(out['moment_weight'][index])
                continue
            first = index[0]
            row, col, position = (int(out[key][first]) for key in ('moment_rows', 'moment_cols', 'moment_position'))
            weight = out['moment_weight'][index].astype(np.float64)
            cumulative = np.cumsum(weight)
            if cumulative[-1] <= 0 or not np.isfinite(cumulative[-1]):
                raise RuntimeError('nonpositive/nonfinite connection attribute weights')
            context = int(out['ctx'][row, col])
            slots = (context * 3 + position) * 257 + np.arange(limit)
            random = stream.random(centers[row:row+1], slots[None])[0]
            chosen = np.minimum(np.searchsorted(cumulative, random * cumulative[-1], side='right'), len(index)-1)
            chosen, multiplicity = np.unique(chosen, return_counts=True)
            keep.append(index[chosen])
            weights.append(multiplicity * (cumulative[-1] / limit))
            out['attribute_sampled_groups'] += 1
        keep = np.concatenate(keep)
        for key in ('moment_rows', 'moment_cols', 'moment_nodes', 'moment_position', 'moment_external'):
            out[key] = out[key][keep]
        out['moment_weight'] = np.concatenate(weights).astype(np.float32)
        out['witness_records'] = len(keep)
        out['witness_external_records'] = int(out['moment_external'].sum())
        return out
