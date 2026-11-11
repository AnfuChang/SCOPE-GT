"""Isolated, matched mechanism experiments. See README.md in this directory."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from SCOPE_GT import train
from SCOPE_GT.data import resolve_feature_normalize
from SCOPE_GT.sampler import ACCESS_KEYS, DIAGNOSTIC_KEYS
from SCOPE_GT_ablation import benchmark_execution as bench
from SCOPE_GT_ablation.execution import make_units

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ('packed_unshared', 'shared_encoder', 'dedup_packed')
MODES = ('independent', 'shared')


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for folder in
            ('SCOPE_GT', 'SCOPE_GT_ablation', 'SCOPE_GT_mechanisms')
            for p in sorted((ROOT / folder).glob('*.py'))}


def compare_draw(a, b):
    errors = {}
    for key in a:
        if not isinstance(a[key], np.ndarray):
            continue  # access counters intentionally differ
        x, y = a[key], b[key]
        if x.shape != y.shape:
            raise AssertionError(f'{key}: shape mismatch {x.shape} != {y.shape}')
        if x.dtype.kind in 'fc':
            errors[key] = float(np.max(np.abs(x - y))) if x.size else 0.
            if not np.allclose(x, y, rtol=1e-6, atol=1e-8):
                raise AssertionError(f'{key}: max error {errors[key]}')
        elif not np.array_equal(x, y):
            raise AssertionError(f'{key}: exact mismatch')
    return errors


class PeakRSS:
    """Sample this isolated retrieval worker; preparation is outside the window."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = self.process.memory_info().rss
        self.baseline = self.peak
        def poll():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def profile_a(args, graph, cache, layout, device, out):
    model = bench.make_model(args, graph, cache, device, dropout_off=False).train()
    item = cache['batches'][0]
    units = bench.move(make_units(layout, item['centers'], item['draw'], graph.features), device)
    labels = torch.as_tensor(item['labels'], device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # Separate pass: profiler overhead never enters the epoch measurement.
    handles = []
    for name, module in [('encoder', model.encoder), ('relation', model.relation)] + [
            (f'layer{i}.{side}', getattr(block, side))
            for i, block in enumerate(model.blocks) for side in ('k_proj', 'v_proj')]:
        stack = []
        def pre(m, inputs, label=name, stack=stack):
            ctx = torch.profiler.record_function(label)
            ctx.__enter__()
            stack.append(ctx)
        def post(m, inputs, output, stack=stack):
            stack.pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            bench.step(model, units, labels, len(labels))
            bench.sync(device)
        prof.export_chrome_trace(str(out / 'profile.json'))
        (out / 'profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total'), encoding='utf-8')
    finally:
        for handle in handles:
            handle.remove()


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    out = Path(spec['out'])
    result = dict(status='failed', experiment=spec['experiment'], variant=spec['variant'],
                  seed=spec['seed'], dataset=spec['dataset'], identity=spec['identity'])
    try:
        if sources() != spec['sources']:
            raise ValueError('source changed after preparing this job')
        torch.set_num_threads(spec['threads'])
        if sha(spec['cache']) != spec['cache_sha256']:
            raise ValueError('cached workload changed after preparing this job')
        with open(spec['cache'], 'rb') as f:
            cache = pickle.load(f)
        args = argparse.Namespace(**cache['args'])
        train.seed_all(spec['seed'])
        graph, sampler = bench.load_setup(args)
        if graph.fingerprint != spec['data_sha256']:
            raise ValueError('input files changed after preparing this job')
        if spec['experiment'] == 'A':
            device = torch.device(spec['device'])
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            checks = bench.check_equivalence(args, graph, cache, [spec['variant']], torch.device('cpu'))
            if not all(r['passed'] for r in checks):
                raise AssertionError(f'logit/gradient check failed: {checks}')
            rows = bench.measure_epoch(spec['variant'], args, graph, sampler, cache, device, 0,
                                       spec['warmup'], spec['epochs'])
            if spec['profile']:
                profile_a(args, graph, cache, spec['variant'], device, out)
            result.update(rows=rows, checks=checks, correctness_scope='dropout-free CPU',
                          stochastic_scope='native dropout; no claim of identical stochastic trajectories')
        else:
            rows = []
            for i, item in enumerate(cache['requests']):
                acc = dict.fromkeys(ACCESS_KEYS + DIAGNOSTIC_KEYS, 0)
                with PeakRSS() as memory:
                    started = time.perf_counter()
                    draw = sampler.relations(item['centers'], item['sample'], item['streams'],
                                             common_cache=None, acc=acc, three_hop_mode=spec['variant'])
                    elapsed = time.perf_counter() - started
                arrays = {k: v for k, v in draw.items() if isinstance(v, np.ndarray)}
                np.savez(out / f'draw_{i:05d}.npz', **arrays)
                rows.append(dict(batch=i, centers=len(item['centers']), retrieval_seconds=elapsed,
                                 cpu_peak_rss_bytes=memory.peak, cpu_baseline_rss_bytes=memory.baseline,
                                 **{k: draw[k] for k in ACCESS_KEYS + DIAGNOSTIC_KEYS}))
                del draw, arrays
            if not sum(row['three_hop_pairs'] for row in rows):
                raise ValueError('fixed workload contains no three-hop requests; not an informative B comparison')
            result.update(rows=rows, scope='fixed-request exact evidence retrieval, not full-graph training',
                          rss_scope='isolated worker RSS, 20ms sampling; graph/features resident')
        result['status'] = 'complete'
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    write(out / 'result.json', result)
    return 0 if result['status'] == 'complete' else 1


def summarize(root):
    specs = sorted(root.glob('*/seed_*/*/spec.json'))
    entries, paired = [], []
    for path in specs:
        s = json.loads(path.read_text(encoding='utf-8'))
        rpath = path.with_name('result.json')
        r = json.loads(rpath.read_text(encoding='utf-8')) if rpath.exists() else {'status': 'missing'}
        row = {k: s[k] for k in ('dataset', 'seed', 'variant', 'experiment', 'identity')}
        row['status'] = r['status']
        if r.get('identity', s['identity']) != s['identity']:
            row['status'] = 'identity_mismatch'
        if s['experiment'] == 'B' and row['status'] == 'complete':
            eq = path.parent.parent / 'equivalence.json'
            if not eq.exists() or not json.loads(eq.read_text()).get('passed'):
                row['status'] = 'equivalence_pending'
        r['status'] = row['status']
        if r['status'] == 'complete':
            if s['experiment'] == 'A':
                rows = [x for x in r['rows'] if not x['warmup_epoch']]
                for key in ('epoch_seconds', 'train_peak_alloc_mib', 'train_peak_reserved_mib', 'encoded_rows', 'kv_projection_rows', 'valid_reads'):
                    values = [x[key] for x in rows if x.get(key) is not None]
                    row[key] = float(max(values) if key.startswith('train_peak') else np.mean(values)) if values else None
            else:
                row['retrieval_seconds'] = sum(x['retrieval_seconds'] for x in r['rows'])
                row['cpu_peak_rss_bytes'] = max(x['cpu_peak_rss_bytes'] for x in r['rows'])
                for key in ACCESS_KEYS + DIAGNOSTIC_KEYS:
                    row[key] = sum(x[key] for x in r['rows'])
        entries.append(row)
    study = json.loads((root / 'study.json').read_text(encoding='utf-8'))
    existing = {(e['dataset'], e['seed'], e['variant']) for e in entries}
    for seed in study['seeds']:
        for variant in study['variants']:
            if (study['dataset'], seed, variant) not in existing:
                entries.append(dict(dataset=study['dataset'], seed=seed, variant=variant,
                                    experiment=study['experiment'], status='missing'))
    for dataset in sorted({e['dataset'] for e in entries}):
        subset = [e for e in entries if e['dataset'] == dataset]
        for variant in sorted({e['variant'] for e in subset}):
            group = [e for e in subset if e['variant'] == variant]
            good = [e for e in group if e['status'] == 'complete']
            row = dict(dataset=dataset, variant=variant, planned=len(group), completed=len(good))
            for key in ('epoch_seconds', 'train_peak_alloc_mib', 'retrieval_seconds', 'cpu_peak_rss_bytes'):
                values = [e[key] for e in good if e.get(key) is not None]
                if len(values) == len(group) and values:
                    row[key + '_mean'] = float(np.mean(values))
                    row[key + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            paired.append(row)
    for name, rows in [('per_seed.csv', entries), ('summary.csv', paired)]:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (root / name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    differences = []
    metric = 'epoch_seconds' if study['experiment'] == 'A' else 'retrieval_seconds'
    comparisons = [('packed_unshared', 'shared_encoder'), ('shared_encoder', 'dedup_packed'),
                   ('packed_unshared', 'dedup_packed')] if study['experiment'] == 'A' else [('independent', 'shared')]
    for seed in study['seeds']:
        group = {e['variant']: e for e in entries if e['seed'] == seed}
        for left, right in comparisons:
            if left not in group or right not in group:
                continue
            a, b = group[left], group[right]
            if a['status'] == b['status'] == 'complete' and a['identity'] == b['identity']:
                differences.append(dict(seed=seed, reference=left, candidate=right, metric=metric,
                                        difference=b[metric]-a[metric], speedup=a[metric]/b[metric]))
    with (root / 'paired_differences.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['seed','reference','candidate','metric','difference','speedup'])
        w.writeheader()
        w.writerows(differences)
    return entries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('experiment', choices=['A', 'B', 'collect'])
    p.add_argument('--dataset', default='wikics')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--centers', type=int, default=2048, help='B microbenchmark centers only; A uses all training centers')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--layouts', nargs='+', choices=LAYOUTS, default=list(LAYOUTS),
                   help='A variants; select shared_encoder only when matched endpoints already exist')
    p.add_argument('--timeout', type=float, default=0, help='per isolated job seconds; 0 means no limit')
    args, extra = p.parse_known_args(argv)
    root = args.out.resolve()
    if args.experiment == 'collect':
        if extra:
            p.error('unexpected collect arguments')
        summarize(root)
        return
    if args.data_root is None or min(args.epochs, args.centers, args.batch_size, args.threads) < 1 or args.warmup < 0:
        p.error('data-root required; positive counts and nonnegative warmup required')
    if len(set(args.layouts)) != len(args.layouts):
        p.error('layouts must be unique')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.timeout < 0:
        p.error('unique nonnegative seeds and nonnegative timeout required')
    if extra[:1] == ['--']:
        extra = extra[1:]
    protected = {'preset', 'dataset', 'data-root', 'seeds', 'batch-size', 'query', 'execution', 'centers-per-epoch'}
    if any(x.split('=')[0] in {'--' + k for k in protected} for x in extra):
        p.error('shared overrides may not alter data, layout, query or training subset')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        p.error('output directory must be empty (use collect to recover completed records)')
    write(root / 'study.json', dict(experiment=args.experiment, dataset=args.dataset, seeds=args.seeds,
                                  variants=args.layouts if args.experiment == 'A' else MODES))
    failures = []
    for seed in args.seeds:
        cfg = train.parse_args(['--preset', args.dataset, '--data-root', str(args.data_root.resolve()),
                                '--batch-size', str(args.batch_size), '--query', 'center', *extra])
        cfg.feature_normalize = resolve_feature_normalize(args.dataset, cfg.feature_normalize)
        cfg.seeds = [seed]
        cfg.threads, cfg.device = args.threads, args.device
        cfg.centers_per_epoch = 0
        cfg.epochs = max(cfg.epochs, args.warmup + args.epochs)
        torch.set_num_threads(args.threads)
        graph, sampler = bench.load_setup(cfg)
        cache = dict(args=vars(cfg), seed=seed)
        if args.experiment == 'A':
            cache = bench.build_cache(cfg, graph, sampler, train.fit_scales(cfg, sampler, graph), 2, seed)
        else:
            if (cfg.max_hops != 3 or cfg.relations != 'on' or cfg.relation_source != 'graph'
                    or cfg.connection_record_budget or cfg.connection_attributes != 'on'
                    or cfg.external_evidence != 'all'):
                p.error('B requires --max-hops 3, relations on and original-graph evidence')
            centers = np.random.default_rng(seed).choice(graph.splits['train'],
                         min(args.centers, len(graph.splits['train'])), replace=False)
            streams = train.make_streams(seed, train.TRAIN_TAGS, 0, 0)
            requests = []
            for start in range(0, len(centers), args.batch_size):
                c = centers[start:start + args.batch_size]
                acc = dict.fromkeys(ACCESS_KEYS + DIAGNOSTIC_KEYS, 0)
                sample, _ = sampler._sample(c, streams, acc)
                requests.append(dict(centers=c, sample=sample, streams=streams))
            cache['requests'] = requests
        directory = root / args.dataset / f'seed_{seed}'
        directory.mkdir(parents=True)
        cache_path = directory / 'inputs.pkl'
        with cache_path.open('wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        identity = dict(sources=sources(), data_sha256=graph.fingerprint, cache_sha256=sha(cache_path),
                        config=vars(cfg), experiment=args.experiment, seed=seed,
                        warmup=args.warmup, epochs=args.epochs, device=args.device, threads=args.threads)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        del graph, sampler, cache
        gc.collect()
        for variant in args.layouts if args.experiment == 'A' else MODES:
            out = directory / variant
            out.mkdir()
            spec = dict(**identity, identity=digest, dataset=args.dataset, variant=variant, cache=str(cache_path),
                        out=str(out), profile=args.profile)
            write(out / 'spec.json', spec)
            command = [sys.executable, '-m', 'SCOPE_GT_mechanisms.run', '--worker', str(out / 'spec.json')]
            print(f'{args.experiment} {args.dataset} seed={seed} {variant}', flush=True)
            try:
                with (out / 'stdout.log').open('w', encoding='utf-8') as log:
                    code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=args.timeout or None).returncode
                if code:
                    if not (out / 'result.json').exists():
                        write(out / 'result.json', dict(status='failed_exit', identity=digest, exit_code=code))
                    failures.append(str(out))
            except subprocess.TimeoutExpired:
                write(out / 'result.json', dict(status='timeout', identity=digest))
                failures.append(str(out))
        if args.experiment == 'B' and all((directory / v / 'result.json').exists() and
                json.loads((directory / v / 'result.json').read_text())['status'] == 'complete' for v in MODES):
            try:
                errors = []
                for path in sorted((directory / 'shared').glob('draw_*.npz')):
                    with np.load(path) as a, np.load(directory / 'independent' / path.name) as b:
                        errors.append(compare_draw(dict(a), dict(b)))
                write(directory / 'equivalence.json', dict(passed=True, batches=errors))
            except Exception as error:
                write(directory / 'equivalence.json', dict(passed=False, error=str(error)))
                for variant in MODES:
                    path = directory / variant / 'result.json'
                    result = json.loads(path.read_text())
                    result['status'] = 'equivalence_failed'
                    write(path, result)
                failures.append(str(directory))
    summarize(root)
    if failures:
        raise SystemExit('Failed jobs (inspect logs): ' + ', '.join(failures))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        raise SystemExit(worker(sys.argv[2]))
    main()
