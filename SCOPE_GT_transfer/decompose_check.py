"""Equivalence self-check for the decomposition arms; run once before the full launch.

For each backbone, one batch of training centers is drawn once and fed to every arm around the
same backbone parameters in eval mode. Expected (up to float32 roundoff):
  NLGT/Gophormer  V1 native_packed == Native        (padding removal changes no computation)
  all backbones   V2 cq_noshare   == +SCMB          (cross-sequence sharing is exact)
                  V3 cq_sharedenc == +SCMB
HubGT's attention-bias dropout is applied by the native model even in eval mode; it is disabled
only inside this check so that outputs are deterministic. No training, no files kept.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from SCOPE_GT_efficiency.worker import runtime
from .adapters import TransferAdapter
from .batches import assemble, to_device
from .config import METHODS, canonical, config_for
from .decompose import (APPLICABLE, DecomposedExtraModel, DecomposedTransferModel, assemble_center_query,
                        assemble_extra_center_query, assemble_native_packed)
from .extra_batches import assemble_extra
from .extra_models import ExtraTransferModel
from .model import TransferModel


def check_method(method, dataset, root, seed, device, centers, atol, pad_multiple):
    cfg = config_for(method, dataset)
    folder = Path(tempfile.mkdtemp(prefix='scope_decompose_check_'))
    try:
        adapter = TransferAdapter(method, 'native', cfg, root, seed, device, folder)
        adapter.load(); adapter.prepare()
        backbone = adapter.model.backbone.eval()
        ids = adapter.splits['train'][:centers]
        extra = method in ('hubgt', 'swapgt')
        if method == 'nlgt':
            graphs = adapter.graphs(ids, np.arange(len(ids)), seed)
        elif method == 'gophormer':
            graphs = adapter.graphs(ids, None, None, np.random.RandomState(seed), cfg['train_samples'])
        else:
            graphs = adapter.graphs(ids)
        if method == 'hubgt':
            for layer in backbone.layers:
                layer.self_attention.dp_bias = 0.
        classes = adapter.classes

        def build(variant):
            if extra:
                labels = adapter.labels[[g['center'] for g in graphs]]
                if variant in ('native', 'scmb'):
                    data, stats = assemble_extra(graphs, method, variant, cfg, labels, classes, adapter.data.num_nodes)
                    return ExtraTransferModel(backbone, method, variant), data, stats
                data, stats = assemble_extra_center_query(graphs, method, variant, cfg, labels, classes,
                                                          adapter.data.num_nodes)
                return DecomposedExtraModel(backbone, method, variant), data, stats
            if variant in ('native', 'scmb'):
                data, stats = assemble(graphs, method, variant, cfg, classes)
                return TransferModel(backbone, method, variant), data, stats
            if variant == 'native_packed':
                data, stats = assemble_native_packed(graphs, method, cfg, classes, pad_multiple)
            else:
                data, stats = assemble_center_query(graphs, method, variant, cfg, classes)
            return DecomposedTransferModel(backbone, method, variant), data, stats

        outputs, work = {}, {}
        for variant in ('native', 'scmb') + APPLICABLE[method]:
            model, data, stats = build(variant)
            with torch.no_grad():
                output = model.eval()(to_device(data, device))
            outputs[variant] = (output[0] if method == 'swapgt' else output).float().cpu()
            work[variant] = stats
        comparisons = [('native', 'native_packed')] if 'native_packed' in APPLICABLE[method] else []
        comparisons += [('scmb', 'cq_noshare'), ('scmb', 'cq_sharedenc')]
        ok = True
        print(f'== {method}: {len(graphs)} sequences from {len(ids)} centers')
        for ref, arm in comparisons:
            diff = (outputs[ref] - outputs[arm]).abs().max().item()
            passed = diff <= atol
            ok &= passed
            print(f'  {arm:<14} vs {ref:<7} max|diff| = {diff:.3e}  {"PASS" if passed else "FAIL"}')
        for variant, stats in work.items():
            print(f'  {variant:<14} reads={stats.get("valid_reads")} encoder_rows={stats.get("encoder_rows", stats.get("memory_rows"))} '
                  f'kv_rows/layer={stats.get("projection_rows_per_layer")} padding={stats.get("allocated_padding_slots")} '
                  f'groups={stats.get("forward_groups", 1)}')
        return ok
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-base', type=Path, required=True)
    p.add_argument('--dataset', type=canonical, default='lastfm_asia')
    p.add_argument('--data-root', type=Path, help='exact dataset folder (default: DATA_BASE/dataset)')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--centers', type=int, default=64, help='training centers in the checked batch')
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--v1-pad-multiple', type=int, default=1)
    a = p.parse_args(argv)
    runtime(a.device)  # same float32 / no-TF32 settings as the measured runs
    root = (a.data_root or a.data_base / a.dataset).resolve()
    device = torch.device(a.device)
    results = {m: check_method(m, a.dataset, root, a.seed, device, a.centers, a.atol, a.v1_pad_multiple)
               for m in a.methods}
    failed = [m for m, ok in results.items() if not ok]
    print('ALL PASS' if not failed else f'FAILED: {" ".join(failed)}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
