"""Bounded CPU cache of deterministic evaluation samples, never learned states."""
from collections import OrderedDict
import time

import numpy as np

from .sampler import trim_padding


class EvaluationDrawCache:
    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError('cache capacity must be nonnegative')
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = self.oversize = 0
        self.build_seconds = 0.

    def get(self, sampler, centers, streams):
        centers = np.asarray(centers, np.int64)
        key = (sampler.cache_signature, centers.tobytes(),
               tuple((name, int(stream.key)) for name, stream in sorted(streams.items())))
        if key in self.entries:
            self.hits += 1
            draw, size = self.entries.pop(key)
            self.entries[key] = draw, size
            return draw, True
        self.misses += 1
        started = time.perf_counter()
        draw = trim_padding(sampler.draw(centers, streams))
        # Logical NumPy payload plus center IDs; not process RSS or GPU memory.
        size = centers.nbytes + sum(value.nbytes for value in draw.values() if isinstance(value, np.ndarray))
        if size > self.max_bytes:
            self.oversize += 1
            self.build_seconds += time.perf_counter() - started
            return draw, False
        # Trimmed arrays can be views of much larger padded allocations. Own
        # compact copies so the payload limit describes the retained storage.
        draw = {name: np.array(value, copy=True, order='C') if isinstance(value, np.ndarray) else value
                for name, value in draw.items()}
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
            self.evictions += 1
        self.entries[key] = draw, size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.build_seconds += time.perf_counter() - started
        return draw, False

    def statistics(self):
        return {'capacity_bytes': self.max_bytes, 'resident_bytes': self.bytes, 'peak_bytes': self.peak_bytes,
                'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'oversize_misses': self.oversize, 'build_seconds': self.build_seconds}
