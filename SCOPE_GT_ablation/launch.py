"""Accuracy ablations of SCOPE-GT: every variant is trained from scratch by SCOPE_GT/train.py.

    python -m SCOPE_GT_ablation.launch wikics --variant Full --data-root /path/to/datasets/wikics
    python -m SCOPE_GT_ablation.launch wikics --variant No-R No-S Full-query --seeds 0 1 2 3 4

--variant takes ids, names or groups from variants.py. The default trains the three controls.
Settings come from SCOPE_GT/presets.json; arguments after `--` are shared by all requested variants and
may not change a variant-defining switch. Each run keeps its raw training outputs and a manifest
(protocol, variant, command, seeds, code fingerprint, exit status). No baseline result is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent / 'SCOPE_GT'
sys.path.insert(0, str(HERE))
from variants import GROUPS, PROTOCOL_ID, VARIANTS, VARIANT_ARGS, expand, flags  # noqa: E402

PROTECTED = {'--' + name.replace('_', '-') for name in VARIANT_ARGS}
PROTECTED |= {'--execution', '--variant', '--preset', '--data-root', '--out-dir', '--dataset',
              '--split-index', '--split-suffix'}
OFFICIAL_MASK_DATASETS = ('minesweeper', 'roman_empire', 'amazon_ratings', 'tolokers')


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(SCOPE.glob('*.py')) + [SCOPE / 'presets.json', HERE / 'variants.py']:
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def presets():
    return json.loads((SCOPE / 'presets.json').read_text(encoding='utf-8'))['datasets']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset')
    p.add_argument('--variant', nargs='+', default=['controls'], help='ids, names or groups: ' + ', '.join(list(GROUPS) + list(VARIANTS)))
    p.add_argument('--data-root', type=Path, default=None, help='dataset directory (default: DATA_BASE/DATASET)')
    p.add_argument('--out-root', type=Path, default=HERE / 'results' / 'accuracy')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--threads', type=int, default=6)
    split = p.add_mutually_exclusive_group()
    split.add_argument('--split-index', type=int, choices=range(10), default=None,
                       help='one official Platonov fixed mask; default 0 on its four datasets')
    split.add_argument('--split-suffix', default=None, help='explicit split-file suffix; empty string selects unsuffixed files')
    p.add_argument('--dry-run', action='store_true', help='print the commands only')
    args, extra = p.parse_known_args(argv)
    if extra[:1] == ['--']:
        extra = extra[1:]
    if args.dataset not in presets():
        p.error(f'no SCOPE_GT preset for {args.dataset!r}')
    try:
        selected = expand(args.variant)
    except KeyError as error:
        p.error(f'unknown variant or group {error}')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.threads < 1:
        p.error('seeds must be unique and nonnegative; threads must be positive')
    for token in extra:
        if token.split('=', 1)[0] in PROTECTED:
            p.error(f'{token} defines a variant or the data; it cannot be a shared override')
    data_root = (args.data_root or Path(os.environ.get('DATA_BASE', str(HERE.parent / 'datasets'))) / args.dataset).resolve()
    suffix = args.split_suffix
    split_index = args.split_index
    if split_index is None and suffix is None and args.dataset in OFFICIAL_MASK_DATASETS:
        split_index = 0
    if split_index is not None:
        suffix = f'_split{split_index}'
    if suffix is None:
        suffix = ''
    required = ('adj.npz', 'feat.npy', 'label.npy', *(f'{name}{suffix}.npy' for name in ('train', 'val', 'test')))
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing and not args.dry_run:
        p.error(f'{data_root} is missing {", ".join(missing)}')
    env = dict(os.environ, PYTHONPATH=str(HERE.parent) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
    fingerprint = code_fingerprint()
    run_root = args.out_root.resolve() / args.dataset
    failures = []
    for variant in selected:
        out_dir = run_root / f'{variant}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        cmd = [sys.executable, '-m', 'SCOPE_GT.train', '--preset', args.dataset, '--dataset', args.dataset,
               '--data-root', str(data_root), '--out-dir', str(out_dir), '--device', args.device,
               '--threads', str(args.threads), '--log-every', '20', '--seeds', *map(str, args.seeds),
               '--split-suffix', suffix, '--variant', variant, *flags(variant), *extra]
        print(f'[{variant}] {VARIANTS[variant][0]}: ' + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        started = datetime.now().isoformat(timespec='seconds')
        entry = {'protocol_id': PROTOCOL_ID, 'dataset': args.dataset, 'variant': variant, 'name': VARIANTS[variant][0],
                 'group': VARIANTS[variant][1], 'out_dir': str(out_dir), 'seeds': args.seeds, 'shared_overrides': extra,
                 'command': cmd, 'scope_code_sha256': fingerprint, 'data_root': str(data_root), 'split_suffix': suffix,
                 'official_split_index': split_index, 'started': started,
                 'finished': None, 'exit_code': None}
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / 'manifest.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry) + '\n')
        code = subprocess.run(cmd, cwd=HERE.parent, env=env).returncode
        entry.update(finished=datetime.now().isoformat(timespec='seconds'), exit_code=code)
        with (run_root / 'manifest.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry) + '\n')
        if code:
            failures.append(variant)
            print(f'[{variant}] failed with exit code {code}; see the run output above', file=sys.stderr, flush=True)
    if failures:
        raise SystemExit('Failed variants: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
