"""Execution layouts of the batch-size study and the isolated timing worker.

SCOPE_GT_ablation/execution.py defines padded (independent node pools, padded to the longest sequence),
grouped, sequential and dedup_packed (shared node table, valid reads only). The batch-size study also needs

  packed   valid reads only, as dedup_packed, but one node pool per center (PackedBatch scope='center'):
           nodes used by several centers, or as their evidence, are encoded and projected once per center

so that Packed vs Dedup-Packed isolates node sharing, and Padded vs Packed isolates packing. The layout is
added here by wrapping the three functions that SCOPE_GT_ablation/benchmark_execution.py calls; no file of
SCOPE_GT or SCOPE_GT_ablation is modified, so the source fingerprints of earlier execution results stay valid.

    python -m SCOPE_GT_batch_size.layouts --worker <spec.json>      (started by run.py)
"""
from __future__ import annotations

import argparse

import numpy as np

from SCOPE_GT.batch import PackedBatch
from SCOPE_GT_ablation import benchmark_execution as be
from SCOPE_GT_ablation import execution as ex
from SCOPE_GT_ablation.execution import slice_draw

STUDY_LAYOUTS = ('dedup_packed', 'packed', 'padded')
_ORIG = dict(execution_rows=ex.execution_rows, make_unit_views=ex.make_unit_views, make_units=ex.make_units)


def execution_rows(layout, draws):
    if layout == 'packed':
        n = len(draws[0]['ctx_valid'])
        if any(len(d['ctx_valid']) != n for d in draws):
            raise ValueError('views must contain the same prediction centers')
        return [np.arange(n)]
    return _ORIG['execution_rows'](layout, draws)


def make_unit_views(layout, centers, draws, features, rows):
    if layout == 'packed':
        return [PackedBatch(np.asarray(centers)[rows], slice_draw(draw, rows), features, 'cpu', scope='center')
                for draw in draws]
    return _ORIG['make_unit_views'](layout, centers, draws, features, rows)


def make_units(layout, centers, draw, features, device='cpu'):
    if layout == 'packed':
        centers = np.asarray(centers, np.int64)
        return [(np.arange(len(centers)), PackedBatch(centers, draw, features, device, scope='center'))]
    return _ORIG['make_units'](layout, centers, draw, features, device)


def install():
    """Route benchmark_execution through the wrappers (idempotent)."""
    be.execution_rows, be.make_unit_views, be.make_units = execution_rows, make_unit_views, make_units


install()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='isolated epoch worker of the batch-size study')
    parser.add_argument('--worker', required=True)
    be.run_worker(parser.parse_args().worker)
