"""Run the query-role x attention-depth grid in sequential worker processes."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from SCOPE_GT.data import resolve_split_suffix


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--data-base', type=Path, default=Path(os.environ.get('DATA_BASE', 'datasets')))
    p.add_argument('--datasets', nargs='+', default=['wikics', 'minesweeper'])
    p.add_argument('--layers', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    p.add_argument('--roles', nargs='+', choices=['center', 'partial', 'full'], default=['center', 'partial', 'full'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--partial-ratio', type=float, default=.5)
    p.add_argument('--out-root', '--out', type=Path, default=None)
    p.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--resume', action='store_true', help='skip completed matching seeds; restart an interrupted seed from initialization')
    p.add_argument('--dry-run', action='store_true', help='print commands without loading data or CUDA or creating results')
    p.add_argument('--fail-fast', action='store_true', help='stop on the first failed worker; otherwise attempt remaining jobs')
    p.add_argument('--diagnostic-centers', type=int, default=4096)
    p.add_argument('--profile-batches', type=int, default=3)
    p.add_argument('--profile-warmup', type=int, default=5)
    p.add_argument('--profile-repeats', type=int, default=20)
    return p


def make_jobs(args, extra):
    presets = json.loads((Path(__file__).resolve().parent.parent / 'SCOPE_GT' / 'presets.json').read_text(encoding='utf-8'))['datasets']
    if len(set(args.datasets)) != len(args.datasets) or set(args.datasets) - presets.keys():
        raise ValueError(f'datasets must be distinct known presets: {sorted(presets)}')
    if len(set(args.layers)) != len(args.layers) or min(args.layers) < 1:
        raise ValueError('layers must be distinct positive integers')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('seeds must be distinct nonnegative integers')
    if len(set(args.roles)) != len(args.roles) or not 0 <= args.partial_ratio <= 1:
        raise ValueError('roles must be distinct; partial ratio must be in [0,1]')
    protected = {'--preset', '--dataset', '--data-root', '--out-dir', '--query', '--execution', '--variant'}
    if any(x.split('=')[0] in protected for x in extra):
        raise ValueError('preset/dataset/data/output/query/execution are controlled by this grid runner')
    jobs = []
    for dataset in args.datasets:
        for layers in args.layers:
            # All three methods are identical at L=1: one training per seed.
            roles = ['center'] if layers == 1 else args.roles
            for role in roles:
                out = args.out_root / dataset / f'L{layers}' / role
                command = [sys.executable, '-u', '-m', 'SCOPE_GT_depth.train',
                    '--preset', dataset, '--data-root', str(args.data_base / dataset), '--out-dir', str(out),
                    '--layers', str(layers), '--role', role, '--partial-ratio', str(args.partial_ratio),
                    '--seeds', *map(str, args.seeds), '--device', args.device, '--threads', str(args.threads),
                    '--diagnostic-centers', str(args.diagnostic_centers), '--profile-batches', str(args.profile_batches),
                    '--profile-warmup', str(args.profile_warmup), '--profile-repeats', str(args.profile_repeats)]
                if args.resume:
                    command.append('--resume')
                command += extra
                jobs.append({'dataset': dataset, 'layers': layers, 'role': role, 'out': str(out), 'command': command})
    return jobs


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.out_root is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_root = Path(__file__).resolve().parent / 'results' / f'depth_{stamp}_{uuid4().hex[:6]}'
    args.out_root = args.out_root.resolve()
    args.data_base = args.data_base.resolve()
    try:
        jobs = make_jobs(args, extra)
    except ValueError as error:
        parser.error(str(error))
    print(f'Results: {args.out_root}\nWorkers: {len(jobs)}; independent seed trainings: {len(jobs)*len(args.seeds)}', flush=True)
    print('L=1 is one shared Center/Partial/Full anchor. Remaining depths are independently trained.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job['command']))
        return
    for name in args.datasets:
        root = args.data_base / name
        suffix, split_index = None, None
        for index, value in enumerate(extra):
            if value == '--split-suffix' and index + 1 < len(extra):
                suffix = extra[index + 1]
            elif value.startswith('--split-suffix='):
                suffix = value.split('=', 1)[1]
            elif value == '--split-index' and index + 1 < len(extra):
                split_index = int(extra[index + 1])
            elif value.startswith('--split-index='):
                split_index = int(value.split('=', 1)[1])
        try:
            suffix = resolve_split_suffix(name, suffix, split_index)
        except ValueError as error:
            parser.error(str(error))
        required = ('adj.npz', 'feat.npy', 'label.npy') + tuple(f'{split}{suffix}.npy' for split in ('train', 'val', 'test'))
        if not all((root / file).is_file() for file in required):
            parser.error(f'missing graph/split files under {root}; set --data-base to their parent')
    args.out_root.mkdir(parents=True, exist_ok=True)
    queue_key = '_'.join(args.datasets)
    (args.out_root / f'plan_{queue_key}.json').write_text(json.dumps({'jobs': jobs, 'seeds': args.seeds,
        'datasets': args.datasets, 'layers': args.layers, 'roles': args.roles,
        'partial_ratio': args.partial_ratio, 'single_layer_shared_anchor': True}, indent=2), encoding='utf-8')
    logs = args.out_root / 'logs'
    logs.mkdir(exist_ok=True)
    code_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env['PYTHONPATH'] = str(code_root) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONUTF8'] = '1'
    failed = []
    for i, job in enumerate(jobs, 1):
        label = f'{job["dataset"]}_L{job["layers"]}_{job["role"]}'
        log = logs / f'{label}.log'
        print(f'[{i}/{len(jobs)}] {label}; log={log}', flush=True)
        with log.open('a' if args.resume else 'w', encoding='utf-8') as handle:
            completed = subprocess.run(job['command'], cwd=code_root, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            failed.append({'job': label, 'returncode': completed.returncode, 'log': str(log)})
            print(f'[FAILED] {label}; inspect {log}', flush=True)
            if args.fail_fast:
                break
        else:
            print(f'[completed] {label}', flush=True)
    (args.out_root / f'failures_{queue_key}.json').write_text(json.dumps(failed, indent=2), encoding='utf-8')
    print(f'Finished. Failed jobs: {len(failed)}. Raw results: {args.out_root}', flush=True)
    print(f'Summarize: python -m SCOPE_GT_depth.collect --results {shlex.quote(str(args.out_root))}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
