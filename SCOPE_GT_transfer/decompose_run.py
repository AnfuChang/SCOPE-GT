"""Launch the V1/V2/V3 decomposition arms that complement the existing Table IX runs.

Native and +SCMB are not rerun. Every arm keeps the backbone sampler, inputs, losses, optimizer,
schedule, validation selection, inference views/votes and initial parameters of Table IX, and is
measured by the same worker (complete training, checkpoint selection, one final test).
Inapplicable pairs (V1 for HubGT/SwapGT, whose inputs have no padding) are skipped with a note.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from SCOPE_GT_budgets.common import dataset_manifest, identity, lock, sha, write_json
from . import PROTOCOL
from .config import CODE, DATASETS, METHODS, canonical, config_for
from .decompose import APPLICABLE, DECOMPOSED_VARIANTS, DEFINITIONS, SKIP_REASON
from .decompose_report import summarize
from .run import sources as transfer_sources

HERE = Path(__file__).resolve().parent
STUDY = 'transfer_decompose'
WORKER = 'SCOPE_GT_transfer.decompose_worker'


def sources(method):
    result = transfer_sources(method)
    for name in ('decompose.py', 'decompose_worker.py', 'decompose_run.py', 'decompose_report.py'):
        path = HERE / name
        result[path.relative_to(CODE).as_posix()] = sha(path)
    return result


def launch(spec, python, args):
    """Same supervision as SCOPE_GT_transfer.run.launch, with the decomposition worker."""
    out = Path(spec['directory']); out.mkdir(parents=True, exist_ok=True)
    with lock(out / '.lock'):
        marker = out / 'spec.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['job_id'] != spec['job_id']:
            raise ValueError(f'Existing output has another source/config/data/runtime identity: {out}; use a fresh --out')
        metrics = out / 'metrics.json'
        if metrics.exists():
            old = json.loads(metrics.read_text(encoding='utf-8'))
            if old.get('status') == 'complete' and old.get('job_id') == spec['job_id']:
                if not all((out / name).exists() for name in ('history.jsonl', 'checkpoint.pt', 'test_predictions.npz')):
                    raise ValueError(f'Missing completed artifacts: {out}')
                print(f'[reuse] {spec["method"]} {spec["variant"]} seed={spec["seed"]}', flush=True)
                return old
        write_json(marker, spec)
        for name in ('progress.json', 'metrics.json', 'history.jsonl'):
            (out / name).unlink(missing_ok=True)
        command = [python, '-u', '-m', WORKER, '--spec', str(marker)]
        started = time.monotonic(); timeout_status = None; progress = {}
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1', OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
        print(f'[start] {spec["method"]} {spec["variant"]} seed={spec["seed"]} -> {out}', flush=True)
        with (out / 'stdout.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(.25)
                try:
                    progress = json.loads((out / 'progress.json').read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    pass
                now = time.monotonic()
                phase = progress.get('phase')
                limit = args.preprocess_timeout if phase in ('load', 'preprocess') else (
                    args.epoch_timeout if phase in ('train', 'validation') else 0)
                if limit and now - progress.get('started_monotonic', now) > limit:
                    timeout_status = 'epoch_timeout' if phase == 'train' else f'{phase}_timeout'
                if args.job_timeout and now - started > args.job_timeout:
                    timeout_status = timeout_status or 'job_timeout'
                if timeout_status:
                    process.kill(); process.wait(); break
        if timeout_status or not metrics.exists():
            write_json(metrics, dict(protocol=spec['protocol'], job_id=spec['job_id'], study=spec['study'],
                                     method=spec['method'], variant=spec['variant'], dataset=spec['dataset'],
                                     n=spec['n'], seed=spec['seed'], status=timeout_status or 'failed_no_metrics',
                                     failed_phase=progress.get('phase'), observed_process_s=time.monotonic() - started,
                                     exit_code=process.returncode))
        result = json.loads(metrics.read_text(encoding='utf-8'))
        write_json(out / 'process.json', dict(command=command, exit_code=process.returncode,
                                              wall_s=time.monotonic() - started))
        print(f'[{result["status"]}] {spec["method"]} {spec["variant"]} seed={spec["seed"]} '
              f'test={result.get("test_accuracy")} epoch_s={result.get("epoch_seconds_mean")} '
              f'gpu_peak={result.get("train_peak_allocated_bytes")}', flush=True)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog=(
        'Uses bundled NPZ adapters; SCOPE_GT_BASELINES_ROOT optionally overrides them.'))
    p.add_argument('--data-base', type=Path, required=True)
    p.add_argument('--out', type=Path, default=CODE / 'results' / STUDY)
    p.add_argument('--datasets', nargs='+', type=canonical, choices=DATASETS, default=['lastfm_asia'])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--variants', nargs='+', choices=DECOMPOSED_VARIANTS, default=list(DECOMPOSED_VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                   help='default: five seeds 0 through 4, matching the Native/+SCMB Table IX launcher; '
                        'reference runs must contain these seeds for complete pairing')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--epoch-timeout', type=float, default=0, help='train/validation watchdog seconds, 0 disables')
    p.add_argument('--preprocess-timeout', type=float, default=0, help='load/preprocessing watchdog seconds, 0 disables')
    p.add_argument('--job-timeout', type=float, default=0, help='whole-run watchdog seconds, 0 disables')
    p.add_argument('--v1-pad-multiple', type=int, default=1,
                   help='V1 groups sequences by length rounded up to this multiple; 1 = exact lengths, zero padding')
    p.add_argument('--overrides', type=Path, help='JSON {all:{method:{...}},datasets:{dataset:{method:{...}}}}; '
                                                   'use the same file as the reference Table IX runs, if any')
    p.add_argument('--dataset-map', type=Path, help='JSON canonical dataset -> exact directory')
    p.add_argument('--reference', type=Path, help='existing Table IX result root; adds pairing checks to the summary')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if (a.threads < 1 or a.warmup_epochs < 0 or a.v1_pad_multiple < 1 or any(s < 0 for s in a.seeds) or
            min(a.epoch_timeout, a.preprocess_timeout, a.job_timeout) < 0 or
            not np.isfinite([a.epoch_timeout, a.preprocess_timeout, a.job_timeout]).all()):
        p.error('Invalid seeds, counts, pad multiple or time budgets')
    for name in ('datasets', 'methods', 'variants', 'seeds'):
        if len(set(getattr(a, name))) != len(getattr(a, name)):
            p.error(f'Duplicate {name}')
    if a.reference and not (a.reference / 'plan.json').exists():
        p.error(f'--reference has no plan.json: {a.reference}')
    pairs =[(m, v) for m in a.methods for v in a.variants if v in APPLICABLE[m]]
    for m in a.methods:
        for v in a.variants:
            if v not in APPLICABLE[m]:
                print(f'[skip] {m} {v}: {SKIP_REASON[m]}', flush=True)
    if not pairs:
        p.error('No applicable backbone/variant pairs')
    overrides = json.loads(a.overrides.read_text(encoding='utf-8')) if a.overrides else {}
    roots = json.loads(a.dataset_map.read_text(encoding='utf-8')) if a.dataset_map else {}
    configs = {f'{d}/{m}': config_for(m, d, {**overrides.get('all', {}).get(m, {}),
                                             **overrides.get('datasets', {}).get(d, {}).get(m, {})})
               for d in a.datasets for m in a.methods}
    description = dict(protocol=PROTOCOL, study=STUDY, datasets=a.datasets, methods=a.methods, variants=a.variants,
                       pairs=[f'{m}/{v}' for m, v in pairs], seeds=a.seeds, configurations=configs,
                       jobs=len(a.datasets) * len(pairs) * len(a.seeds), warmup_epochs=a.warmup_epochs,
                       v1_pad_multiple=a.v1_pad_multiple, definitions=DEFINITIONS)
    if a.dry_run:
        print(json.dumps(description, ensure_ascii=False, indent=2))
        return
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    description.update(source_sha256={m: sources(m) for m in a.methods}, device=a.device, threads=a.threads,
                       data_base=str(a.data_base.resolve()), dataset_map=roots, python=sys.executable,
                       time_limits=dict(epoch=a.epoch_timeout, preprocess=a.preprocess_timeout, job=a.job_timeout))
    with lock(out / '.study.lock'):
        frozen = out / 'study_identity.json'
        if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != description:
            raise ValueError('Frozen study/source/config changed; use a fresh --out')
        write_json(frozen, description)
        probe = subprocess.run([sys.executable, '-m', WORKER, '--probe', a.device], cwd=CODE,
                               capture_output=True, text=True, encoding='utf-8', timeout=60)
        if probe.returncode:
            raise RuntimeError(probe.stderr or probe.stdout)
        runtime = json.loads(probe.stdout)
        specs = []
        for dataset in a.datasets:
            root = Path(roots.get(dataset, a.data_base / dataset)).resolve()
            fingerprints = dataset_manifest(root)
            n = len(np.load(root / 'label.npy', mmap_mode='r', allow_pickle=False))
            for method in a.methods:
                variants = [v for v in a.variants if v in APPLICABLE[method]]
                for seed in a.seeds:
                    # Rotate the arm order across seeds, as in the Table IX launcher.
                    order = variants[seed % len(variants):] + variants[:seed % len(variants)]
                    for variant in order:
                        spec = dict(protocol=PROTOCOL, study=STUDY, method=method, variant=variant, dataset=dataset,
                                    seed=seed, requested_seeds=a.seeds, n=n, config=configs[f'{dataset}/{method}'],
                                    data_root=str(root), data_sha256=fingerprints,
                                    source_sha256=description['source_sha256'][method], runtime=runtime,
                                    device=a.device, threads=a.threads, warmup_epochs=a.warmup_epochs,
                                    v1_pad_multiple=a.v1_pad_multiple if variant == 'native_packed' else 1,
                                    time_limits=description['time_limits'])
                        spec['job_id'] = identity(spec)
                        spec['directory'] = str(out / dataset / method / f'seed{seed}' / variant)
                        specs.append(spec)
        write_json(out / 'plan.json', specs)
        for spec in specs:
            launch(spec, sys.executable, a)
            summarize([out], a.reference, out)
        rows = summarize([out], a.reference, out)
        if any(r['status'] != 'complete' for r in rows if r.get('source') == 'decomposition'):
            raise SystemExit('Incomplete runs; inspect failures.csv and per-run stdout.log')


if __name__ == '__main__':
    main()
