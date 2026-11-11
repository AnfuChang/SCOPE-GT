"""Summarize V1-V3 runs and, with --reference, merge them with the existing Table IX runs.

Outputs (in --out):
  per_run.csv          one row per run (decomposition roots and reference root)
  summary.csv          mean/sample-SD per dataset/backbone/variant, plus per-epoch work counters
  pairing.csv          each V1-V3 run checked against the reference Native run of the same seed:
                       identical initial parameters, configuration, data and per-epoch sampled inputs
  decomposition.csv    per backbone: V0..V4 means and the share of the Native->+SCMB change
                       contributed by each step, in percent of V0
  table_ix_extended.tex / .md   Table IX rows extended with the new arms
Measured values only; missing arms stay empty.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from SCOPE_GT_budgets.common import write_json
from SCOPE_GT_efficiency.run import read_history, write_csv
from .decompose import LABELS, ORDER
from .report import FIELDS, moments

COUNTERS = ('valid_reads', 'encoder_rows', 'projection_rows_per_layer', 'attention_pairs_per_layer',
            'allocated_padding_slots', 'forward_groups')
STEP_NAMES = {('native', 'native_packed'): 'padding removal',
              ('native_packed', 'cq_noshare'): 'center-query architecture',
              ('native', 'cq_noshare'): 'padding removal + center-query architecture',
              ('cq_noshare', 'cq_sharedenc'): 'encoder sharing',
              ('cq_sharedenc', 'scmb'): 'K/V sharing',
              ('cq_noshare', 'scmb'): 'encoder + K/V sharing'}
COST_FIELDS = (('train_peak_allocated_bytes', 'gpu_memory', 1 / 2 ** 20),
               ('epoch_seconds_mean', 'epoch_time', 1.),
               ('job_seconds', 'total_time', 1.))


def load_runs(root, variants=None):
    root = Path(root)
    specs = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    runs = []
    for s in specs:
        if variants and s['variant'] not in variants:
            continue
        path = Path(s['directory']) / 'metrics.json'
        r = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(status='pending')
        if path.exists() and r.get('job_id') != s['job_id']:
            raise ValueError(f'Metrics identity mismatch: {path}')
        runs.append((s, r))
    return runs


def counters(spec):
    """Work counters summed over one training epoch (identical for every epoch of a seed)."""
    history = read_history(spec['directory'])
    if not history:
        return {}
    totals = history[0].get('batch_totals', {})
    result = {name: totals.get(name) for name in COUNTERS}
    if result['encoder_rows'] is None:  # existing Native/+SCMB records: encoder rows = memory rows
        result['encoder_rows'] = totals.get('memory_rows')
    if result['forward_groups'] is None and totals:
        result['forward_groups'] = 1
    return result


def paired(spec, result, native_spec, native):
    if result.get('status') != 'complete' or native.get('status') != 'complete':
        return 'incomplete', 0
    hn = {r['epoch']: r for r in read_history(native_spec['directory'])}
    ht = {r['epoch']: r for r in read_history(spec['directory'])}
    common = sorted(set(hn) & set(ht))
    matched = (native['initial_state_sha256'] == result['initial_state_sha256'] and
               native['parameters'] == result['parameters'] and spec['config'] == native_spec['config'] and
               spec['data_sha256'] == native_spec['data_sha256'] and
               all(hn[e]['input_sha256'] == ht[e]['input_sha256'] for e in common))
    return ('complete' if matched and common else 'pair_validation_failed'), len(common)


def summarize(results, reference=None, out=None):
    results = [Path(r) for r in results]
    out = Path(out) if out else results[0]
    new = [run for root in results for run in load_runs(root)]
    # Use only the reference seeds that the decomposition runs cover, so every arm in a backbone's
    # row is averaged over the same seeds (e.g. seed 1 only, not seed 1 against five-seed means).
    covered = {(s['dataset'], s['method'], s['seed']) for s, _ in new}
    ref = [(s, r) for s, r in load_runs(reference, variants=('native', 'scmb'))
           if (s['dataset'], s['method'], s['seed']) in covered] if reference else []
    rows, mapping = [], {}
    for source, runs in (('decomposition', new), ('reference', ref)):
        for s, r in runs:
            key = (s['dataset'], s['method'], s['seed'], s['variant'])
            if key in mapping:
                raise ValueError(f'Duplicate run for {key}; pass each result root once')
            row = {k: s[k] for k in ('dataset', 'method', 'variant', 'seed', 'job_id', 'directory')}
            row['source'] = source
            row.update({k: v for k, v in r.items() if not isinstance(v, (dict, list)) and k != 'traceback'})
            if r.get('status') == 'complete':
                row.update(counters(s))
            mapping[key] = (s, r)
            rows.append(row)

    grouped = []
    for d, m, v in dict.fromkeys((r['dataset'], r['method'], r['variant']) for r in rows):
        items = [r for r in rows if (r['dataset'], r['method'], r['variant']) == (d, m, v)]
        good = [r for r in items if r['status'] == 'complete']
        row = dict(dataset=d, method=m, variant=v, label=LABELS.get(v, v), requested_runs=len(items),
                   completed_runs=len(good), status='complete' if len(good) == len(items) else 'incomplete')
        for field in FIELDS:
            moments(row, field, [r.get(field) for r in good], len(items))
        for name in COUNTERS:
            values = [r.get(name) for r in good if r.get(name) is not None]
            row[name + '_per_epoch'] = float(np.mean(values)) if values else None
        grouped.append(row)

    pairs = []
    for s, r in new:
        key = (s['dataset'], s['method'], s['seed'], 'native')
        pair = dict(dataset=s['dataset'], method=s['method'], seed=s['seed'], variant=s['variant'])
        if key not in mapping:
            pair.update(status='no_reference_native')
        else:
            native_spec, native = mapping[key]
            status, epochs = paired(s, r, native_spec, native)
            pair.update(status=status, matched_epochs=epochs)
            if status == 'complete':
                pair.update(accuracy_delta_vs_native_pp=100 * (r['test_accuracy'] - native['test_accuracy']),
                            gpu_reduction_vs_native_percent=100 * (1 - r['train_peak_allocated_bytes'] /
                                                                   native['train_peak_allocated_bytes'])
                            if r.get('train_peak_allocated_bytes') and native.get('train_peak_allocated_bytes') else None,
                            epoch_speedup_vs_native=native['epoch_seconds_mean'] / r['epoch_seconds_mean']
                            if r.get('epoch_seconds_mean') and native.get('epoch_seconds_mean') else None)
        pairs.append(pair)

    means = {(g['dataset'], g['method'], g['variant']): g for g in grouped}
    decomposition = []
    for d, m in dict.fromkeys((g['dataset'], g['method']) for g in grouped):
        present = [v for v in ORDER if (d, m, v) in means and means[d, m, v]['status'] == 'complete']
        row = dict(dataset=d, method=m, arms=' '.join(present))
        for v in present:
            g = means[d, m, v]
            row[f'{v}_accuracy_pct'] = 100 * g['test_accuracy_mean'] if g['test_accuracy_mean'] is not None else None
            for field, name, scale in COST_FIELDS:
                value = g.get(field + '_mean')
                row[f'{v}_{name}'] = value * scale if value is not None else None
        for field, name, scale in COST_FIELDS:
            base = row.get(f'native_{name}')
            if not base:
                continue
            for prev, cur in zip(present, present[1:]):
                a, b = row.get(f'{prev}_{name}'), row.get(f'{cur}_{name}')
                if a is not None and b is not None:
                    label = STEP_NAMES.get((prev, cur), f'{prev}->{cur}')
                    row[f'{name}: {label} (% of V0)'] = 100 * (a - b) / base
            if 'scmb' in present and row.get(f'scmb_{name}') is not None:
                row[f'{name}: total Native->+SCMB (% of V0)'] = 100 * (base - row[f'scmb_{name}']) / base
        decomposition.append(row)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_run.csv', rows)
    write_csv(out / 'summary.csv', grouped)
    write_json(out / 'summary.json', grouped)
    write_csv(out / 'pairing.csv', pairs)
    write_csv(out / 'decomposition.csv', decomposition)
    write_csv(out / 'failures.csv', [r for r in rows if r['status'] not in ('complete', 'pending')] +
              [p for p in pairs if p['status'] == 'pair_validation_failed'])

    def cell(g, field, scale=1., sd=True):
        x, s = g.get(field + '_mean'), g.get(field + '_sd')
        if x is None:
            return '--'
        return f'{x * scale:.2f}' if (s is None or not sd) else f'{x * scale:.2f} $\\pm$ {s * scale:.2f}'

    ordered = sorted(grouped, key=lambda g: (g['dataset'], g['method'], ORDER.index(g['variant'])
                                             if g['variant'] in ORDER else 99))
    tex = ['% Measured values only; accuracy mean $\\pm$ sample SD over seeds, other columns are means.',
           r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Backbone & Variant & Accuracy (\%) & Epoch time (s) & GPU memory (MiB) & Total time (s) \\', r'\midrule']
    md = ['| Backbone | Variant | Accuracy (%) | Epoch time (s) | GPU memory (MiB) | Total time (s) | Encoder rows/epoch | K/V rows/epoch | Padding slots/epoch | Forward groups/epoch |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in ordered:
        values = [g['method'].upper(), LABELS.get(g['variant'], g['variant']), cell(g, 'test_accuracy', 100),
                  cell(g, 'epoch_seconds_mean', sd=False), cell(g, 'train_peak_allocated_bytes', 1 / 2 ** 20, sd=False),
                  cell(g, 'job_seconds', sd=False)]
        tex.append(' & '.join(values) + r' \\')

        def number(name):
            x = g.get(name + '_per_epoch')
            return '--' if x is None else f'{x:,.0f}'
        md.append('| ' + ' | '.join([v.replace(' $\\pm$ ', ' ± ') for v in values] +
                                    [number('encoder_rows'), number('projection_rows_per_layer'),
                                     number('allocated_padding_slots'), number('forward_groups')]) + ' |')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (out / 'table_ix_extended.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    (out / 'table_ix_extended.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    write_json(out / 'report_metadata.json', dict(
        gpu_unit='MiB (2^20 bytes), peak allocated during training epochs', accuracy_unit='fraction in CSV; percent in tables',
        across_seed_sd='sample SD; absent for one seed', epoch_scope='mean training epoch after warmup epochs',
        total_time='load through prediction export', counters='summed over the first training epoch of each seed; per-layer for K/V rows and attention pairs',
        decomposition='step shares are (previous arm - next arm) / Native, in percent; they add up to the Native->+SCMB change',
        pairing='V1-V3 runs must share initial parameters, configuration, data and per-epoch sampled inputs with the reference Native run',
        reference_seeds='reference Native/+SCMB runs are restricted to the dataset/backbone/seed combinations of the decomposition runs'))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, nargs='+', required=True, help='one or more decompose_run --out roots')
    p.add_argument('--reference', type=Path, help='existing Table IX result root (run_transfer.sh OUT) with native/scmb runs')
    p.add_argument('--out', type=Path, help='output folder (default: first --results root)')
    a = p.parse_args(argv)
    summarize(a.results, a.reference, a.out)
    print(f'Wrote summary to {(a.out or a.results[0]).resolve()}')


if __name__ == '__main__':
    main()
