"""Plan and launch isolated training/inference jobs for both research experiments."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import PROTOCOL
from .common import CODE, dataset_manifest, identity, lock, source_manifest, write_json


DEFAULT_K_VALUES = (8, 16, 24, 32, 48, 64)
DEFAULT_W_VALUES = (12, 24, 36, 48, 72, 96)


def budget_points(cfg, axes, k_values=None, walk_values=None, record_values=None):
    """Use common scan coordinates and keep each dataset's other budget fixed.

    WikiCS and DBLP each share one production point between their two sweeps.
    Other presets share a point only if both production budgets are in the grid.
    """
    k, w, r = cfg['context_budget'], cfg['walks'] or cfg['context_budget'], cfg['connection_record_budget']
    points = {}
    def add(kk, ww, rr, axis):
        key = (kk, ww, rr)
        points.setdefault(key, [])
        if axis not in points[key]:
            points[key].append(axis)
    for axis in axes:
        if axis == 'k':
            for kk in k_values or DEFAULT_K_VALUES:
                add(kk, w, r, axis)
        elif axis == 'w':
            for ww in walk_values or DEFAULT_W_VALUES:
                add(k, ww, r, axis)
        else:
            if cfg['connection_attributes'] != 'on':
                raise ValueError('r sweep requires C attributes ON; current Tolokers is OFF')
            for rr in record_values or [4, 16, 0]:
                add(k, w, rr, axis)
    return [{'context_budget': kk, 'walks': ww, 'connection_record_budget': rr, 'axes': names}
            for (kk, ww, rr), names in points.items()]


def build_plan(args):
    from .adapters import config_for
    overrides = json.loads(args.overrides.read_text(encoding='utf-8')) if args.overrides else {}
    unknown = set(overrides) - {'all', 'datasets'}
    if unknown:
        raise ValueError(f'Override top-level keys must be all/datasets: {unknown}')
    jobs = []
    for dataset in args.datasets:
        manifest = None
        if not args.dry_run:
            print(f'[audit] hashing dataset {dataset}', flush=True)
            manifest = dataset_manifest(args.data_base / dataset)
        methods = args.methods if args.suite == 'efficiency' else ['scope_gt']
        for method in methods:
            update = {**overrides.get('all', {}).get(method, {}),
                      **overrides.get('datasets', {}).get(dataset, {}).get(method, {})}
            if args.epochs is not None:
                update['epochs'] = args.epochs
            if args.eval_every is not None:
                update['eval_every'] = args.eval_every
            cfg = config_for(method, dataset, update)
            points = budget_points(cfg, args.axes, args.k_values, args.walk_values, args.record_values) if args.suite == 'budget' else [{}]
            sources = source_manifest(method) if not args.dry_run else {}
            for point in points:
                effective = {**cfg, **{k: v for k, v in point.items() if k != 'axes'}}
                effective = config_for(method, dataset, effective)
                config_id = identity(effective)[:10]
                tag = (f"K{effective['context_budget']}_W{effective['walks'] or effective['context_budget']}_r{effective['connection_record_budget']}_"
                       if method == 'scope_gt' else 'final_') + config_id
                val_limit = args.val_limit
                if val_limit is None:
                    val_limit = effective.get('val_max_centers', 0) if args.suite == 'budget' else 20000
                for seed in args.seeds:
                    job = {'protocol': PROTOCOL, 'suite': args.suite, 'dataset': dataset, 'method': method,
                        'config': effective, 'seed': seed, 'data_root': str((args.data_base / dataset).resolve()),
                        'data_sha256': manifest, 'source_sha256': sources, 'threads': args.threads,
                        'device': args.device, 'val_limit': val_limit, 'axes': point.get('axes', []),
                        'requested_cold_repeats': args.cold_repeats, 'warm_repeats': args.warm_repeats}
                    job['is_default_budget'] = method == 'scope_gt' and (
                        effective['context_budget'] == cfg['context_budget'] and
                        (effective['walks'] or effective['context_budget']) == (cfg['walks'] or cfg['context_budget']) and
                        effective['connection_record_budget'] == cfg['connection_record_budget'])
                    job['job_id'] = identity(job)
                    job['directory'] = str((args.out / dataset / method / tag / f'seed{seed}').resolve())
                    jobs.append(job)
    return jobs


def launch(job, phase, output, args):
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / 'metrics.json'
    with lock(output / '.lock'):
        if metrics.exists():
            previous = json.loads(metrics.read_text(encoding='utf-8'))
            if previous.get('status') == 'complete':
                if not args.resume:
                    raise RuntimeError(f'Completed output exists (use --resume or a new --out): {output}')
                if previous.get('job_id') != job['job_id']:
                    raise RuntimeError(f'Output identity mismatch; choose a new --out: {output}')
                if phase == 'train':
                    manifest = output / 'artifacts.json'
                    if not manifest.exists() or any(not (output / name).is_file() for name in
                            json.loads(manifest.read_text(encoding='utf-8'))):
                        raise RuntimeError(f'Completed training artifacts are missing: {output}; restore them or use a fresh output')
                print(f'[skip] {phase}: {output}', flush=True)
                return True
        spec = {**job, 'phase': phase, 'output': str(output), 'train_dir': str(Path(job['directory']) / 'train')}
        write_json(output / 'spec.json', spec)
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED='1', PYTHONUTF8='1')
        command = [sys.executable, '-m', 'SCOPE_GT_budgets.worker', '--spec', str(output / 'spec.json')]
        print(f"[start] {job['dataset']} {job['method']} seed={job['seed']} {phase} -> {output}", flush=True)
        started = time.perf_counter()
        with (output / 'stdout.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.timeout or None)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                process.kill()
                process.wait()
                status = 'timeout_explicit' if isinstance(error, subprocess.TimeoutExpired) else 'interrupted'
                write_json(metrics, {'job_id': job['job_id'], 'status': status, 'limit_seconds': args.timeout})
                if isinstance(error, KeyboardInterrupt):
                    raise
                code = -1
        write_json(output / 'process.json', {'command': command, 'exit_code': code,
            'wall_seconds': time.perf_counter() - started, 'scope': 'full subprocess including imports, integrity checking and export'})
        if not metrics.exists():
            write_json(metrics, {'job_id': job['job_id'], 'status': 'failed_no_metrics', 'exit_code': code})
        result = json.loads(metrics.read_text(encoding='utf-8'))
        ok = code == 0 and result.get('status') == 'complete'
        print(f"[{'done' if ok else 'FAILED'}] {phase}: {output}", flush=True)
        if not ok:
            print(f"  reason: {result.get('error', result.get('status', 'unknown'))}; "
                  f"details: {output / 'stdout.log'}", flush=True)
        if not ok and args.fail_fast:
            raise RuntimeError(f'Worker failed: inspect {output / "stdout.log"}')
        return ok


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        'Optional backbone methods need a compatible external NPZ adapter tree in '
        'SCOPE_GT_BASELINES_ROOT. SCOPE-GT and --dry-run do not need this tree.'))
    p.add_argument('suite', choices=['efficiency', 'budget'])
    p.add_argument('--data-base', type=Path, default=CODE / 'datasets')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--methods', nargs='+', choices=['scope_gt', 'nlgt', 'nagphormer', 'sgformer'],
                   default=['scope_gt', 'nlgt', 'nagphormer', 'sgformer'])
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--cold-repeats', type=int, default=5,
                   help='checkpoint-only cold processes per trained seed (default: 5)')
    p.add_argument('--warm-repeats', type=int, default=3)
    p.add_argument('--timeout', type=float, default=0, help='seconds per isolated subprocess; 0=no limit')
    p.add_argument('--val-limit', type=int, default=None, help='0=all; efficiency default 20000 shared per-seed; budget default native')
    p.add_argument('--axes', nargs='+', choices=['k', 'w', 'r'], default=['k', 'w'])
    p.add_argument('--k-values', type=int, nargs='+')
    p.add_argument('--walk-values', type=int, nargs='+')
    p.add_argument('--record-values', type=int, nargs='+')
    p.add_argument('--overrides', type=Path, help='JSON {all:{method:{...}},datasets:{dataset:{method:{...}}}}')
    p.add_argument('--epochs', type=int, help='explicit training cap override (recorded); use 2 for smoke only')
    p.add_argument('--eval-every', type=int, help='explicit validation interval override, recorded')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='print effective jobs; do not hash data, create files or initialize CUDA')
    p.add_argument('--fail-fast', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.datasets = args.datasets or (['wikics', 'dblp', 'tolokers', 'coauthor_cs'] if args.suite == 'efficiency' else ['wikics', 'dblp'])
    args.seeds = args.seeds or [0, 1, 2, 3, 4]
    for values in (args.datasets, args.methods, args.seeds, args.axes):
        if len(set(values)) != len(values):
            p.error('dataset/method/seed/axis lists must not contain duplicates')
    if min(args.seeds) < 0 or args.threads < 1 or args.cold_repeats < 1 or args.warm_repeats < 0 or args.timeout < 0:
        p.error('invalid seed/thread/repetition/timeout settings')
    if args.val_limit is not None and args.val_limit < 0:
        p.error('val-limit must be nonnegative')
    for values in (args.k_values, args.walk_values):
        if values and min(values) < 1:
            p.error('K/W must be positive')
    if args.record_values and (min(args.record_values) < 0 or max(args.record_values) > 64):
        p.error('record values must be in 0..64; 0=full')
    if 'r' in args.axes and len(args.axes) > 1:
        p.error('Run optional r sweep separately with --axes r; do not mix its axis with K/W')
    jobs = build_plan(args)
    if args.dry_run:
        print(json.dumps({'protocol': PROTOCOL, 'jobs': jobs, 'training_jobs': len(jobs),
            'inference_processes': len(jobs) * args.cold_repeats}, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    write_json(args.out / f'launch_{stamp}.json', {'protocol': PROTOCOL, 'arguments': vars(args), 'jobs': jobs})
    failed = 0
    for job in jobs:
        root = Path(job['directory'])
        root.mkdir(parents=True, exist_ok=True)
        job_file = root / 'job.json'
        if job_file.exists() and json.loads(job_file.read_text(encoding='utf-8'))['job_id'] != job['job_id']:
            raise RuntimeError(f'Existing job identity differs: {root}; choose a fresh output directory')
        write_json(job_file, job)
        if not launch(job, 'train', root / 'train', args):
            failed += 1
            continue
        for repeat in range(args.cold_repeats):
            if not launch(job, 'infer', root / f'infer_cold{repeat}', args):
                failed += 1
    print(f'Finished: {len(jobs)} training configurations/seeds, {failed} failed phases. Raw output: {args.out}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
