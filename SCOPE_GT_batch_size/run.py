"""Batch-size study of SCOPE-GT: the training batch size B varies, every other setting stays at the preset.

Run from the repository root (or through scripts/run_batch_size.sh):

  python -m SCOPE_GT_batch_size.run accuracy --datasets wikics corafull --data-base /data
  python -m SCOPE_GT_batch_size.run cost     --datasets wikics corafull --data-base /data
  python -m SCOPE_GT_batch_size.run sharing  --datasets wikics corafull arxiv_year pokec ogbn_products --data-base /data

accuracy  full SCOPE-GT training (SCOPE_GT.train, final packed execution) for each B and seed; one process per
          (B, seed) so that interrupted work resumes per seed. Only --batch-size differs from the main experiment.
cost      execution cost per B and seed: dedup_packed (SCOPE-GT), packed (same valid reads, one node pool per
          center) and padded (one node pool per center, padded) -- the epoch protocol of
          SCOPE_GT_ablation/benchmark_execution.py (fresh trajectory, warm-up epochs, measured epochs, isolated
          process per layout); a CPU logits/gradient check against dedup_packed precedes timing.
sharing   CPU only: all training mini-batches of epoch 0 (view 0) for each B and seed, in the training order;
          counts token reads M, unique token nodes, encoded-node occurrences without cross-center sharing and
          unique encoded nodes (tokens plus connection / background evidence nodes). Also checks that every
          center receives the same sampled inputs whatever B is.

All phases default to seeds 0--4. Cost and sharing store each seed separately; --cost-seed explicitly
selects one seed for reproducing a historical single-seed run.
Each output folder stores request.json (sources, data fingerprints, arguments); a rerun with an identical
request reuses finished work, a different request stops with an error instead of mixing protocols.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from SCOPE_GT import train as scope_gt  # noqa: E402
from SCOPE_GT.data import resolve_feature_normalize  # noqa: E402
from SCOPE_GT.sampler import CTX_KEYS  # noqa: E402

SCHEMA = 'scope_gt_batch_size_v2'
DATA_FILES = ('adj.npz', 'feat.npy', 'label.npy', 'train.npy', 'val.npy', 'test.npy')
DEFAULT_SIZES = {'wikics': [16, 32, 64, 128, 256], 'corafull': [16, 32, 64, 128, 256, 512, 1024]}
SHARING_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
STUDY_FILES = ('run.py', 'layouts.py')


# ---------------------------------------------------------------------------------------------- helpers
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    out = {'SCOPE_GT/' + n: sha256(CODE / 'SCOPE_GT' / n) for n in scope_gt.SOURCE_FILES}
    out['SCOPE_GT/presets.json'] = sha256(CODE / 'SCOPE_GT' / 'presets.json')
    for n in ('execution.py', 'benchmark_execution.py'):
        out['SCOPE_GT_ablation/' + n] = sha256(CODE / 'SCOPE_GT_ablation' / n)
    for n in STUDY_FILES:
        out['SCOPE_GT_batch_size/' + n] = sha256(HERE / n)
    return out


def data_fingerprint(root):
    return {n: sha256(root / n) for n in DATA_FILES}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bind(folder, request, dry_run):
    """Create folder/request.json, or check that an existing folder was produced by the same request."""
    marker = folder / 'request.json'
    if marker.exists():
        if read_json(marker) != request:
            raise SystemExit(f'{folder} holds results of a different request (sources, data or arguments); '
                             f'move it away or choose another --out-root')
        return
    if not dry_run:
        write_json(marker, request)


def sizes_for(dataset, args, default):
    if args.batch_sizes:
        return sorted(set(args.batch_sizes))
    return default.get(dataset, default.get('*')) if isinstance(default, dict) else list(default)


def scope_args(dataset, data_root, batch_size, device, threads, extra):
    argv = ['--preset', dataset, '--dataset', dataset, '--data-root', str(data_root), '--device', device,
            '--threads', str(threads), '--batch-size', str(batch_size), *extra]
    cargs = scope_gt.parse_args(argv)
    cargs.feature_normalize = resolve_feature_normalize(dataset, cargs.feature_normalize)
    cargs.metric = 'auc' if dataset in scope_gt.AUC_DATASETS else 'accuracy'
    if cargs.query != 'center' or cargs.execution != 'packed':
        raise SystemExit(f'{dataset}: the preset must use center queries and packed execution')
    return cargs


# ---------------------------------------------------------------------------------------------- accuracy
def run_accuracy(args, dataset, data_root, out_root, base):
    preset = scope_gt.load_preset(dataset)
    for b in sizes_for(dataset, args, DEFAULT_SIZES):
        folder = out_root / dataset / 'accuracy' / f'B{b}'
        request = dict(base, phase='accuracy', dataset=dataset, batch_size=b, preset=preset,
                       shared_overrides=args.extra)
        bind(folder, request, args.dry_run)
        for seed in args.seeds:
            run_dir = folder / f's{seed}'
            if (run_dir / f'seed{seed}' / 'metrics.json').exists():
                print(f'[accuracy] {dataset} B={b} seed={seed}: done', flush=True)
                continue
            if run_dir.exists() and not args.dry_run:          # interrupted run: keep it aside, never mix
                run_dir.rename(run_dir.with_name(f'{run_dir.name}.incomplete_{datetime.now():%Y%m%d_%H%M%S}'))
            cmd = [sys.executable, '-u', '-m', 'SCOPE_GT.train', '--preset', dataset, '--dataset', dataset,
                   '--data-root', str(data_root), '--out-dir', str(run_dir), '--device', args.device,
                   '--threads', str(args.threads), '--log-every', '20', '--seeds', str(seed),
                   '--variant', f'batch_size_{b}', '--execution', 'packed', '--batch-size', str(b), *args.extra]
            print(f'[accuracy] {dataset} B={b} seed={seed}: ' + shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            log = folder / f's{seed}.log'
            with log.open('w', encoding='utf-8') as handle:
                code = subprocess.run(cmd, cwd=CODE, env=child_env(), stdout=handle, stderr=subprocess.STDOUT).returncode
            if code:
                raise SystemExit(f'{dataset} B={b} seed={seed} failed (exit {code}); see {log}')


def child_env():
    path = str(CODE) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    return dict(os.environ, PYTHONPATH=path, PYTHONUNBUFFERED='1', PYTHONUTF8='1')


# ---------------------------------------------------------------------------------------------- cost
def run_cost(args, dataset, data_root, out_root, base):
    import torch
    from SCOPE_GT_batch_size import layouts  # noqa: F401  (installs the packed layout)
    from SCOPE_GT_ablation import benchmark_execution as be
    import SCOPE_GT_batch_size.layouts as lay
    layouts_ = args.layouts or list(lay.STUDY_LAYOUTS)
    if 'dedup_packed' not in layouts_:
        layouts_ = ['dedup_packed'] + layouts_
    for b in sizes_for(dataset, args, DEFAULT_SIZES):
        folder = out_root / dataset / 'cost' / f'B{b}' / f'seed{args.cost_seed}'
        if (folder.parent / 'request.json').exists():
            raise SystemExit(f'{folder.parent} contains a legacy single-seed study; choose a fresh --out-root')
        request = dict(base, phase='cost', dataset=dataset, batch_size=b, seed=args.cost_seed, layouts=layouts_,
                       epoch_warmup=args.epoch_warmup, epoch_repeats=args.epoch_repeats,
                       shared_overrides=args.extra,
                       cuda=torch.cuda.get_device_name(0) if args.device == 'cuda' and torch.cuda.is_available() else None)
        print(f'[cost] {dataset} B={b} layouts={layouts_} -> {folder}', flush=True)
        bind(folder, request, args.dry_run)
        if args.dry_run:
            continue
        cargs = scope_args(dataset, data_root, b, args.device, args.threads, args.extra)
        torch.set_num_threads(args.threads)
        graph, sampler = be.load_setup(cargs)
        scales = scope_gt.fit_scales(cargs, sampler, graph) if cargs.relations == 'on' else None
        cache_path = folder / 'cache.pkl'
        if cache_path.exists():
            with cache_path.open('rb') as f:                    # written by this runner for this request
                cache = pickle.load(f)
        else:
            cache = be.build_cache(cargs, graph, sampler, scales, args.check_batches, args.cost_seed)
            cache['args'] = {k: str(v) if isinstance(v, Path) else v for k, v in cache['args'].items()}
            with cache_path.open('wb') as f:
                pickle.dump(cache, f)
        eq_path = folder / 'equivalence.json'
        if eq_path.exists():
            checks = read_json(eq_path)
        else:
            checks = be.check_equivalence(cargs, graph, cache, [l for l in layouts_ if l != 'dedup_packed'],
                                          torch.device('cpu'))
            write_json(eq_path, checks)
        passed = {l: all(r['passed'] for r in checks if r['layout'] == l) for l in layouts_}
        passed['dedup_packed'] = True
        reference = be.checksum(be.make_model(cargs, graph, cache, torch.device('cpu')))
        order = layouts_[b % len(layouts_):] + layouts_[:b % len(layouts_)]   # rotate start order across B
        for layout in order:
            result_path = folder / f'epoch_{layout}.json'
            if result_path.exists() and read_json(result_path).get('status') == 'ok':
                print(f'  {layout:13s} (reuse)', flush=True)
                continue
            if not passed[layout]:
                write_json(result_path, dict(status='equivalence_failed', layout=layout, rows=[]))
                print(f'  {layout:13s} equivalence_failed', flush=True)
                continue
            spec = dict(cache=str(cache_path), protocol='epoch', layout=layout, device=args.device,
                        threads=args.threads, warmup=0, repeats=1, checksum=reference,
                        epoch_centers=cargs.centers_per_epoch, epoch_warmup=args.epoch_warmup,
                        epoch_repeats=args.epoch_repeats, result=str(result_path))
            spec_path = folder / f'epoch_{layout}_spec.json'
            write_json(spec_path, spec)
            with (folder / f'epoch_{layout}.log').open('w', encoding='utf-8') as log:
                code = subprocess.run([sys.executable, '-u', '-m', 'SCOPE_GT_batch_size.layouts', '--worker',
                                       str(spec_path)], cwd=CODE, env=child_env(), stdout=log,
                                      stderr=subprocess.STDOUT).returncode
            status = read_json(result_path).get('status') if result_path.exists() else f'no result (exit {code})'
            print(f'  {layout:13s} {status}', flush=True)


# ---------------------------------------------------------------------------------------------- sharing
def center_signatures(centers, draw):
    """Digest of each center's sampled inputs: every pair field of its contexts (in order) and its
    connection / background evidence records (with weights), independent of the record order."""
    out = {}
    valid = draw['ctx_valid']
    for i, c in enumerate(centers):
        h = hashlib.sha1()
        for key in CTX_KEYS:
            if key != 'ctx_valid':
                h.update(np.ascontiguousarray(draw[key][i, valid[i]]).tobytes())
        for prefix, extra in (('moment', ('moment_weight', 'moment_position')), ('context_bg', ())):
            sel = np.flatnonzero(draw[prefix + '_rows'] == i)
            cols, nodes = draw[prefix + '_cols'][sel], draw[prefix + '_nodes'][sel]
            idx = sel[np.lexsort((nodes, cols))]
            for key in (prefix + '_cols', prefix + '_nodes', *extra):
                h.update(np.ascontiguousarray(draw[key][idx]).tobytes())
        out[int(c)] = h.hexdigest()
    return out


def run_sharing(args, dataset, data_root, out_root, base):
    from SCOPE_GT.batch import PackedBatch
    from SCOPE_GT.sampler import trim_padding
    from SCOPE_GT_ablation import benchmark_execution as be
    folder = out_root / dataset / 'sharing' / f'seed{args.cost_seed}'
    if (folder.parent / 'request.json').exists():
        raise SystemExit(f'{folder.parent} contains legacy sharing statistics; choose a fresh --out-root')
    sizes = sizes_for(dataset, args, SHARING_SIZES)
    request = dict(base, phase='sharing', dataset=dataset, batch_sizes=sizes, seed=args.cost_seed,
                   shared_overrides=args.extra)
    print(f'[sharing] {dataset} B={sizes} -> {folder}', flush=True)
    bind(folder, request, args.dry_run)
    if args.dry_run or (folder / 'summary.csv').exists():
        if (folder / 'summary.csv').exists():
            print('  (reuse)', flush=True)
        return
    cargs = scope_args(dataset, data_root, sizes[0], 'cpu', args.threads, args.extra)
    graph, sampler = be.load_setup(cargs)
    seed = args.cost_seed
    rng = np.random.default_rng(np.random.SeedSequence([seed, scope_gt.SHUFFLE_TAG, 0]))
    pool = graph.splits['train']
    if cargs.centers_per_epoch and cargs.centers_per_epoch < len(pool):
        pool = rng.choice(pool, cargs.centers_per_epoch, replace=False)
    order = rng.permutation(len(pool))
    empty = np.zeros((len(graph.labels), 0), np.float32)   # zero-width rows: indices only, no feature values
    batch_rows, summary, reference = [], [], None
    for b in sizes:
        if b > len(pool):
            print(f'  B={b}: larger than the {len(pool)} training centers of an epoch; skipped', flush=True)
            continue
        streams = scope_gt.make_streams(seed, scope_gt.TRAIN_TAGS, 0, 0)
        signatures, tot = {}, dict.fromkeys(('centers', 'reads', 'unique_tokens', 'encoded_occurrences',
                                             'unique_encoded'), 0)
        started = time.perf_counter()
        for i, start in enumerate(range(0, len(pool), b)):
            centers = pool[order[start:start + b]]
            draw = trim_padding(sampler.draw(centers, streams))
            own = PackedBatch(centers, draw, empty, 'cpu', scope='center')
            shared = PackedBatch(centers, draw, empty, 'cpu', scope='batch')
            reads = own.pairs + own.size                      # contexts plus each center's self-read
            if own.kv_count != reads:
                raise AssertionError('per-center pools must hold one K/V row per token read')
            row = dict(dataset=dataset, seed=seed, batch_size=b, batch=i, centers=len(centers), reads=reads,
                       unique_tokens=shared.kv_count, encoded_occurrences=own.encoded_rows,
                       unique_encoded=shared.encoded_rows)
            row['token_repeat_fraction'] = 1. - row['unique_tokens'] / reads
            row['encoded_repeat_fraction'] = 1. - row['unique_encoded'] / row['encoded_occurrences']
            batch_rows.append(row)
            for k in tot:
                tot[k] += row[k]
            signatures.update(center_signatures(centers, draw))
        if reference is None:
            reference = signatures
        same = signatures == reference
        summary.append(dict(dataset=dataset, seed=seed, batch_size=b, batches=sum(r['batch_size'] == b for r in batch_rows),
                            epoch_centers=len(pool), **tot,
                            token_repeat_fraction=1. - tot['unique_tokens'] / tot['reads'],
                            encoded_repeat_fraction=1. - tot['unique_encoded'] / tot['encoded_occurrences'],
                            reads_per_unique_token=tot['reads'] / tot['unique_tokens'],
                            occurrences_per_unique_encoded=tot['encoded_occurrences'] / tot['unique_encoded'],
                            inputs_identical_to_first_B=same, seconds=time.perf_counter() - started))
        print(f'  B={b}: token repeats {100 * summary[-1]["token_repeat_fraction"]:.2f}%  '
              f'encoded repeats {100 * summary[-1]["encoded_repeat_fraction"]:.2f}%  '
              f'inputs identical: {same}', flush=True)
    write_rows(folder / 'batches.csv', batch_rows)
    write_rows(folder / 'summary.csv', summary)
    if not all(r['inputs_identical_to_first_B'] for r in summary):
        raise SystemExit(f'{dataset}: sampled inputs of some centers depend on B; see {folder / "summary.csv"}')


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument('phase', choices=['accuracy', 'cost', 'sharing'])
    p.add_argument('--datasets', nargs='+', default=['wikics', 'corafull'])
    p.add_argument('--data-base', type=Path, default=Path(os.environ.get('DATA_BASE', 'datasets')))
    p.add_argument('--out-root', type=Path, default=HERE / 'results')
    p.add_argument('--batch-sizes', nargs='+', type=int, default=None,
                   help='default: wikics 16..256, corafull 16..1024 (accuracy, cost); 16..2048 (sharing)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                   help='independent seeds for every phase (default: 0 1 2 3 4)')
    p.add_argument('--cost-seed', type=int, default=None,
                   help='explicit single-seed override for cost/sharing; otherwise use --seeds')
    p.add_argument('--layouts', nargs='+', choices=['dedup_packed', 'packed', 'padded'], default=None)
    p.add_argument('--epoch-warmup', type=int, default=3)
    p.add_argument('--epoch-repeats', type=int, default=10)
    p.add_argument('--check-batches', type=int, default=2, help='real batches of the CPU equivalence check')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--dry-run', action='store_true')
    args, extra = p.parse_known_args(argv)
    args.extra = extra[1:] if extra[:1] == ['--'] else extra
    for token in args.extra:
        if token.split('=', 1)[0] in ('--batch-size', '--preset', '--data-root', '--out-dir', '--execution',
                                      '--query', '--seeds', '--dataset'):
            p.error(f'{token} is set by the study and cannot be overridden')
    if args.batch_sizes and min(args.batch_sizes) < 1:
        p.error('batch sizes must be positive')
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or args.cost_seed is not None and args.cost_seed < 0):
        p.error('seeds must be unique nonnegative integers')
    if args.cost_seed is not None and args.phase == 'accuracy':
        p.error('--cost-seed applies only to cost/sharing')
    if args.threads < 1 or args.epoch_warmup < 0 or args.epoch_repeats < 1 or args.check_batches < 1:
        p.error('threads/check batches/measured epochs must be positive; warmup must be nonnegative')
    requested_seeds = args.seeds if args.cost_seed is None else [args.cost_seed]
    if args.phase == 'cost' and args.device == 'cuda' and not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA unavailable; cost measurements need the GPU (use --device cpu only for a smoke test)')
    out_root = args.out_root.resolve()
    src = sources()
    for dataset in args.datasets:
        data_root = (args.data_base / dataset).resolve()
        missing = [n for n in DATA_FILES if not (data_root / n).is_file()]
        if missing and not args.dry_run:
            raise SystemExit(f'{data_root} is missing {", ".join(missing)}')
        base = dict(schema=SCHEMA, source_sha256=src,
                    dataset_sha256=data_fingerprint(data_root) if not missing else None,
                    device=args.device, threads=args.threads, requested_seeds=requested_seeds)
        if args.phase == 'accuracy':
            run_accuracy(args, dataset, data_root, out_root, base)
        else:
            for seed in requested_seeds:
                seed_args = argparse.Namespace(**vars(args))
                seed_args.cost_seed = seed
                {'cost': run_cost, 'sharing': run_sharing}[args.phase](
                    seed_args, dataset, data_root, out_root, base)


if __name__ == '__main__':
    main()
