"""Collect the batch-size study into three CSV tables (raw per-seed values stay in the run folders).

  python -m SCOPE_GT_batch_size.collect [--out-root SCOPE_GT_batch_size/results]

summary_accuracy.csv  dataset, B, completed seeds, test/val metric mean +- std (selection metric of the preset),
                      best epoch, epochs run, optimizer steps per epoch, epoch time (training only, mean over
                      seeds), train peak GPU memory (max over seeds), training and inference seconds
summary_cost.csv      per-seed epoch-time means and per-seed GPU peaks aggregated across independent seeds;
                      sample SD is across seeds, not across epochs; per_seed_cost.csv retains each run
summary_sharing.csv   per-seed sharing statistics aggregated across seeds; per_seed_sharing.csv retains runs
Rows are written only for finished work; missing seeds are counted, never filled.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIB = 2. ** 20


def rows_to_csv(path, rows):
    if not rows:
        print(f'{path.name}: no finished results')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'{path} ({len(rows)} rows)')


def mean_std(values):
    a = np.asarray(values, float)
    return (float(a.mean()) if len(a) else None, float(a.std(ddof=1)) if len(a) > 1 else None)


def batch_size(folder):
    return int(folder.name[1:])


def accuracy(root):
    out = []
    for folder in sorted(root.glob('*/accuracy/B*'), key=lambda p: (p.parts[-3], batch_size(p))):
        dataset = folder.parts[-3]
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        metrics, n_train = [], None
        for m in sorted(folder.glob('s*/seed*/metrics.json')):
            metrics.append(json.loads(m.read_text(encoding='utf-8')))
            if n_train is None:
                n_train = json.loads((m.parent.parent / 'config.json').read_text(encoding='utf-8'))['splits']['train']
        if not metrics:
            continue
        b = request['batch_size']
        metric = 'auc' if 'test_auc' in metrics[0] else 'accuracy'
        epoch_centers = request['preset'].get('centers_per_epoch') or n_train
        epoch_centers = min(epoch_centers, n_train)
        row = dict(dataset=dataset, batch_size=b, metric=metric, seeds=len(metrics),
                   seed_ids=' '.join(str(m['seed']) for m in metrics),
                   optimizer_steps_per_epoch=math.ceil(epoch_centers / b))
        for key in (f'test_{metric}', f'val_{metric}', 'best_epoch', 'epochs_run', 'epoch_seconds',
                    'epoch_seconds_all', 'train_seconds', 'inference_seconds', 'time_to_best_seconds'):
            row[key + '_mean'], row[key + '_std'] = mean_std([m[key] for m in metrics])
        peaks = [m['train_peak_allocated_bytes'] for m in metrics if m.get('train_peak_allocated_bytes')]
        row['train_peak_alloc_mib_max'] = max(peaks) / MIB if peaks else None
        out.append(row)
    return out


def cost_runs(root):
    out = []
    for folder in sorted(root.glob('*/cost/B*'), key=lambda p: (p.parts[-3], batch_size(p))):
        dataset, b = folder.parts[-3], batch_size(folder)
        runs = sorted(folder.glob('seed*'))
        if runs and (folder / 'request.json').exists():
            raise ValueError(f'Legacy and multi-seed cost outputs coexist in {folder}; collect them separately')
        runs = runs or [folder]
        identity = None
        for run in runs:
            request_file = run / 'request.json'
            if not request_file.exists():
                continue
            request = json.loads(request_file.read_text(encoding='utf-8'))
            seed = request['seed']
            common = {k: v for k, v in request.items() if k != 'seed'}
            if identity is not None and common != identity:
                raise ValueError(f'Cost run identities differ in {folder}; do not combine protocols')
            identity = common
            eq_path = run / 'equivalence.json'
            eq = json.loads(eq_path.read_text(encoding='utf-8')) if eq_path.exists() else []
            for path in sorted(run.glob('epoch_*.json')):
                if path.name.endswith('_spec.json'):
                    continue
                result = json.loads(path.read_text(encoding='utf-8'))
                layout = path.stem[len('epoch_'):]
                checks = [r for r in eq if r['layout'] == layout]
                row = dict(dataset=dataset, batch_size=b, seed=seed, layout=layout,
                           requested_runs=len(request.get('requested_seeds', [seed])), status=result.get('status'),
                           equivalence_passed=None if layout == 'dedup_packed'
                           else bool(checks) and all(r['passed'] for r in checks))
                measured = [r for r in result.get('rows') or [] if not r['warmup_epoch']]
                if measured:
                    t = [r['epoch_seconds'] for r in measured]
                    mem = [r['train_peak_alloc_mib'] for r in measured if r['train_peak_alloc_mib'] is not None]
                    row.update(measured_epochs=len(measured), epoch_seconds_mean=float(np.mean(t)),
                               epoch_seconds_within_run_sd=float(np.std(t, ddof=1)) if len(t) > 1 else None,
                               train_peak_alloc_mib=max(mem) if mem else None,
                               optimizer_steps=measured[0]['optimizer_steps'], centers=measured[0]['centers'],
                               sample_s=float(np.mean([r['sample_s'] for r in measured])),
                               assemble_s=float(np.mean([r['assemble_s'] for r in measured])),
                               compute_s=float(np.mean([r['compute_s'] for r in measured])))
                out.append(row)
    return out


def cost(root):
    grouped = defaultdict(list)
    for run in cost_runs(root):
        grouped[(run['dataset'], run['batch_size'], run['layout'])].append(run)
    out = []
    for (dataset, b, layout), runs in sorted(grouped.items()):
        if len({r['seed'] for r in runs}) != len(runs):
            raise ValueError(f'Duplicate seeds in cost study {dataset}/B{b}/{layout}')
        good = [r for r in runs if r['status'] == 'ok' and r.get('epoch_seconds_mean') is not None
                and (layout == 'dedup_packed' or r['equivalence_passed'])]
        expected = max(r['requested_runs'] for r in runs)
        row = dict(dataset=dataset, batch_size=b, layout=layout,
                   status='complete' if len(good) == expected else 'incomplete',
                   requested_runs=expected, completed_runs=len(good), seed_ids=' '.join(str(r['seed']) for r in good),
                   sd_scope='sample SD across per-seed epoch means')
        for key in ('epoch_seconds_mean', 'train_peak_alloc_mib', 'sample_s', 'assemble_s', 'compute_s'):
            values = [r[key] for r in good if r.get(key) is not None]
            avg, std = mean_std(values)
            row[key] = avg
            row['epoch_seconds_sd' if key == 'epoch_seconds_mean' else key + '_sd'] = std
        for key in ('measured_epochs', 'optimizer_steps', 'centers'):
            values = {r[key] for r in good}
            row[key] = next(iter(values)) if len(values) == 1 else None
        row['epoch_seconds_within_run_sd_mean'], _ = mean_std(
            [r['epoch_seconds_within_run_sd'] for r in good if r['epoch_seconds_within_run_sd'] is not None])
        out.append(row)
    index = {(r['dataset'], r['batch_size'], r['layout']): r for r in out}
    for row in out:
        if row['layout'] != 'dedup_packed':
            continue
        for ref in ('packed', 'padded'):
            other = index.get((row['dataset'], row['batch_size'], ref), {})
            if row['seed_ids'] != other.get('seed_ids'):
                continue
            for key, name in (('epoch_seconds_mean', 'time'), ('train_peak_alloc_mib', 'memory')):
                if row.get(key) and other.get(key):
                    row[f'{name}_reduction_vs_{ref}'] = 1. - row[key] / other[key]
    return out


def sharing_runs(root):
    out = []
    for folder in sorted(root.glob('*/sharing')):
        files = sorted(folder.glob('seed*/summary.csv'))
        if files and (folder / 'summary.csv').exists():
            raise ValueError(f'Legacy and multi-seed sharing outputs coexist in {folder}; collect them separately')
        files = files or ([folder / 'summary.csv'] if (folder / 'summary.csv').exists() else [])
        identity = None
        for path in files:
            request = json.loads((path.parent / 'request.json').read_text(encoding='utf-8'))
            common = {k: v for k, v in request.items() if k != 'seed'}
            if identity is not None and common != identity:
                raise ValueError(f'Sharing run identities differ in {folder}; do not combine protocols')
            identity = common
            with path.open(encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    out.append(dict(row, seed=int(request['seed']),
                                    requested_runs=len(request.get('requested_seeds', [request['seed']]))))
    return out


def sharing(root):
    grouped = defaultdict(list)
    for run in sharing_runs(root):
        grouped[(run['dataset'], int(run['batch_size']))].append(run)
    out = []
    for (dataset, b), runs in sorted(grouped.items()):
        if len({r['seed'] for r in runs}) != len(runs):
            raise ValueError(f'Duplicate seeds in sharing study {dataset}/B{b}')
        expected = max(r['requested_runs'] for r in runs)
        row = dict(dataset=dataset, batch_size=b, requested_runs=expected, completed_runs=len(runs),
                   status='complete' if len(runs) == expected else 'incomplete',
                   seed_ids=' '.join(str(r['seed']) for r in runs),
                   inputs_identical_to_first_B=all(str(r['inputs_identical_to_first_B']).lower() == 'true' for r in runs))
        for key in ('batches', 'epoch_centers', 'centers', 'reads', 'unique_tokens', 'encoded_occurrences',
                    'unique_encoded', 'token_repeat_fraction', 'encoded_repeat_fraction',
                    'reads_per_unique_token', 'occurrences_per_unique_encoded', 'seconds'):
            values = [float(r[key]) for r in runs]
            row[key], row[key + '_sd'] = mean_std(values)
        out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-root', type=Path, default=HERE / 'results')
    args = p.parse_args(argv)
    root = args.out_root.resolve()
    rows_to_csv(root / 'summary_accuracy.csv', accuracy(root))
    rows_to_csv(root / 'per_seed_cost.csv', cost_runs(root))
    rows_to_csv(root / 'summary_cost.csv', cost(root))
    rows_to_csv(root / 'per_seed_sharing.csv', sharing_runs(root))
    rows_to_csv(root / 'summary_sharing.csv', sharing(root))


if __name__ == '__main__':
    main()
