"""One training or checkpoint-only inference process. Called by run.py."""
from __future__ import annotations

import time
BOOT = time.perf_counter()

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
import sklearn

from . import PROTOCOL
from .common import Meter, dataset_manifest, identity, rss_bytes, sha, source_manifest, sync, write_json
from .adapters import Adapter, seed_native


def scores(logits, labels, metric):
    from SCOPE_GT.train import scores as compute
    result = compute(logits, labels, metric)
    value = result['auc' if metric == 'auc' else 'accuracy']
    if not math.isfinite(value):
        raise ValueError('Nonfinite selection/test metric; inspect class coverage and logits')
    return result


def selection(metrics, metric):
    return metrics['auc' if metric == 'auc' else 'accuracy']


def max_peak(stages, key):
    values = [s[key] for s in stages if s.get(key) is not None]
    return max(values) if values else None


def checkpoint_save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(job, adapter, device, output, start):
    stages = []
    with Meter(device) as load:
        adapter.load()
    stages.append(load.result)
    with Meter(device) as prep:
        adapter.prepare()
        opt = adapter.optimizer()
    stages.append(prep.result)
    ready = time.perf_counter()
    ids = adapter.splits['val']
    if job['val_limit'] and len(ids) > job['val_limit']:
        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], 31337]))
        ids = ids[np.sort(rng.choice(len(ids), job['val_limit'], replace=False))]
    common = {'protocol': PROTOCOL, 'job_id': job['job_id'], 'config': job['config'],
              'data_sha256': job['data_sha256'], 'source_sha256': job['source_sha256'],
              'val_ids_sha256': identity(ids.tolist()), 'seed': job['seed'], 'method': job['method'],
              'dataset': job['dataset']}
    write_json(output / 'validation.json', {'count': len(ids), 'ids_sha256': common['val_ids_sha256'],
                                           'rule': 'same seed=SeedSequence([seed,31337]); limit=0 means all'})
    best, best_epoch, bad, best_ready_s = -math.inf, 0, 0, None
    history, peak_stages, exposure, steps_total = [], [], 0, 0
    fit_start = time.perf_counter()
    with (output / 'history.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, adapter.a.epochs + 1):
            with Meter(device) as tm:
                row = {'epoch': epoch, **adapter.epoch(epoch, opt)}
            if not math.isfinite(row['loss']):
                raise RuntimeError(f'Nonfinite train loss at epoch {epoch}')
            row['train'] = tm.result
            exposure += row['supervised_centers']
            steps_total = steps_total + row['optimizer_steps'] if steps_total is not None and row['optimizer_steps'] is not None else None
            peak_stages.append(tm.result)
            if adapter.eval_due(epoch):
                with Meter(device) as vm:
                    logits, _ = adapter.predict(ids, epoch, validation=True)
                    result = scores(logits, adapter.labels[ids], adapter.a.metric)
                del logits
                row.update(val=result, validation=vm.result)
                peak_stages.append(vm.result)
                score = selection(result, adapter.a.metric)
                epsilon = 1e-6 if job['method'] == 'scope_gt' else 0.
                if score > best + epsilon:
                    best, best_epoch, bad = score, epoch, 0
                    checkpoint_save(output / 'checkpoint.pt', {**common, 'best_epoch': epoch,
                        'best_val': score, 'extra': adapter.state_extra,
                        'state_dict': {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}})
                    # Includes graph preparation and each preceding validation/save; actual timestamp.
                    best_ready_s = time.perf_counter() - start
                else:
                    bad += adapter.a.eval_every if job['method'] == 'nlgt' else 1
            history.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            print(f"epoch={epoch} loss={row['loss']:.6f} train_s={tm.seconds:.3f} best={best:.6f} best_epoch={best_epoch}", flush=True)
            if adapter.a.patience and bad >= adapter.a.patience:
                break
    fit_stop = time.perf_counter() - fit_start
    if not best_epoch:
        raise RuntimeError('No validation checkpoint recorded')
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    adapter.model.load_state_dict(checkpoint['state_dict'])
    with Meter(device) as final:
        test_logits, diagnostic = adapter.predict(adapter.splits['test'], best_epoch)
    metrics = scores(test_logits, adapter.labels[adapter.splits['test']], adapter.a.metric)
    np.savez_compressed(output / 'test_predictions.npz', centers=adapter.splits['test'],
                        labels=adapter.labels[adapter.splits['test']], logits=test_logits)
    static = output / 'tokens.pt'
    artifacts = {'checkpoint.pt': sha(output / 'checkpoint.pt')}
    if static.exists():
        artifacts['tokens.pt'] = sha(static)
    write_json(output / 'artifacts.json', artifacts)
    result = {**common, 'status': 'complete', 'metric': adapter.a.metric,
        'test': metrics, 'best_val': best, 'best_epoch': best_epoch, 'epochs_run': len(history),
        'load': load.result, 'preprocess': prep.result, 'time_ready_best_seconds': best_ready_s,
        'fit_stop_seconds': fit_stop, 'job_seconds': time.perf_counter() - start,
        'train_seconds': sum(r['train']['seconds'] for r in history),
        'validation_seconds': sum(r.get('validation', {}).get('seconds', 0.) for r in history),
        'final_test': final.result, 'test_diagnostics': diagnostic,
        'supervised_center_exposures': exposure, 'optimizer_steps': steps_total,
        'train_selection_gpu_peak_allocated_bytes': max_peak(peak_stages, 'gpu_peak_allocated_bytes'),
        'train_selection_gpu_peak_reserved_bytes': max_peak(peak_stages, 'gpu_peak_reserved_bytes'),
        'complete_gpu_peak_allocated_bytes': max_peak(stages + peak_stages + [final.result], 'gpu_peak_allocated_bytes'),
        'parameters': sum(p.numel() for p in adapter.model.parameters() if p.requires_grad),
        'feature_location': adapter.feature_location, 'storage': adapter.storage(),
        'artifact_bytes': sum((output / name).stat().st_size for name in artifacts),
        'training_protocol': 'method-specific optimizer/loss; validation only; test once after best checkpoint',
        'timing_scope': 'after imports and integrity preflight; includes input, prepare, training, selection, final test and required export',
        'metric_calculation': 'included in validation; excluded from final prediction latency'}
    return result


def infer(job, adapter, device, output, start):
    checkpoint_dir = Path(job['train_dir'])
    with Meter(device) as startup:
        checkpoint = torch.load(checkpoint_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if checkpoint['job_id'] != job['job_id']:
            raise RuntimeError('Checkpoint/config identity mismatch')
        adapter.load()
        adapter.prepare(checkpoint)
        best_epoch = checkpoint['best_epoch']
        del checkpoint
    startup_seconds = time.perf_counter() - start
    stages, tests, reference = [], [], None
    for request in range(job['warm_repeats'] + 1):
        with Meter(device) as meter:
            logits, diagnostic = adapter.predict(adapter.splits['test'], best_epoch)
        if reference is None:
            reference = logits
            first_metric = scores(logits, adapter.labels[adapter.splits['test']], adapter.a.metric)
            # Compare with this very checkpoint's original final prediction, not another run's score.
            with np.load(checkpoint_dir / 'test_predictions.npz', allow_pickle=False) as old:
                if not np.array_equal(old['centers'], adapter.splits['test']):
                    raise RuntimeError('Test target order changed')
                if not np.allclose(old['logits'], logits, rtol=2e-4, atol=2e-5):
                    raise RuntimeError('Checkpoint-only inference differs from training result; do not pair quality and time')
                expected = scores(old['logits'], old['labels'], adapter.a.metric)
                if abs(selection(expected, adapter.a.metric) - selection(first_metric, adapter.a.metric)) > 1e-12:
                    raise RuntimeError('Classification quality changed at deployment; do not pair old quality with new latency')
        elif not np.allclose(reference, logits, rtol=2e-4, atol=2e-5):
            raise RuntimeError('Repeated deterministic request changed prediction')
        stages.append(meter.result)
        tests.append({'request': request, 'kind': 'first' if request == 0 else 'repeat',
                      **meter.result, 'diagnostics': diagnostic,
                      'ms_per_target_all_views': 1000. * meter.seconds / len(adapter.splits['test'])})
        del logits
    return {'protocol': PROTOCOL, 'job_id': job['job_id'], 'status': 'complete',
        'method': job['method'], 'dataset': job['dataset'], 'seed': job['seed'], 'metric': adapter.a.metric,
        'test': first_metric, 'startup': startup.result, 'startup_seconds': startup_seconds,
        'first_query_seconds': tests[0]['seconds'], 'cold_total_seconds': startup_seconds + tests[0]['seconds'],
        'requests': tests, 'feature_location': adapter.feature_location, 'actual_device': str(device),
        'inference_gpu_peak_allocated_bytes': max_peak([startup.result] + stages, 'gpu_peak_allocated_bytes'),
        'inference_gpu_peak_reserved_bytes': max_peak([startup.result] + stages, 'gpu_peak_reserved_bytes'),
        'storage': adapter.storage(), 'optimizer_loaded': False,
        'cold_definition': 'new checkpoint-only process; query draw cache empty; static NAG tokens persisted; OS page cache not flushed',
        'timing_scope': 'after imports and integrity preflight; all startup preparation plus complete test logits returned on CPU'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.spec.read_text(encoding='utf-8'))
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        audit_start = time.perf_counter()
        if dataset_manifest(job['data_root']) != job['data_sha256']:
            raise RuntimeError('Dataset changed since plan was created')
        current_source = source_manifest(job['method'])
        if current_source != job['source_sha256']:
            changed = sorted(name for name in set(current_source) | set(job['source_sha256'])
                             if current_source.get(name) != job['source_sha256'].get(name))
            raise RuntimeError('Source changed since plan was created: ' + ', '.join(changed))
        if job['phase'] == 'infer':
            artifacts = json.loads((Path(job['train_dir']) / 'artifacts.json').read_text(encoding='utf-8'))
            for name, expected in artifacts.items():
                if sha(Path(job['train_dir']) / name) != expected:
                    raise RuntimeError(f'Artifact changed: {name}')
        audit_seconds = time.perf_counter() - audit_start
        start = time.perf_counter()
        torch.set_num_threads(job['threads'])
        if job['device'].startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable; no silent CPU fallback')
        device = torch.device(job['device'])
        if job['phase'] == 'infer' and job['method'] == 'sgformer' and job['config']['eval_mode'] == 'full_cpu':
            device = torch.device('cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        seed_native(job['method'], job['seed'], job['config'])
        adapter = Adapter(job['method'], job['config'], job['data_root'], job['seed'], device,
                          Path(job['train_dir']) if job['phase'] == 'infer' else output)
        result = (train if job['phase'] == 'train' else infer)(job, adapter, device, output, start)
        result.update(cpu_process_peak_rss_bytes=rss_bytes()[1], integrity_preflight_seconds=audit_seconds,
            python_worker_seconds=time.perf_counter() - BOOT,
            environment={'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                         'sklearn': sklearn.__version__, 'platform': platform.platform(),
                         'threads': job['threads'], 'cuda': torch.version.cuda,
                         'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
            cpu_memory_scope='single process, workers=0; OS lifetime RSS peak includes imports and integrity checking; stage RSS is sampled')
        write_json(output / 'metrics.json', result)
        print(json.dumps({'status': 'complete', 'output': str(output), 'test': result['test']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        status = 'OOM_explicit' if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)) else 'failed'
        write_json(output / 'metrics.json', {'status': status, 'job_id': job['job_id'],
            'error': str(error), 'traceback': traceback.format_exc(), 'phase': job['phase']})
        raise


if __name__ == '__main__':
    main()
