"""Run a bundled classification recipe and retain source/input/runtime identities."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time

from input_manifest import manifest, sha256

ROOT = Path(__file__).resolve().parents[1]


def terminate_process_tree(process):
    """Stop the launcher's children as well as its shell after a deadline."""
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The shell can exit before a child that ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        # The bundled Bash baseline recipes target Linux. Avoid relying on
        # taskkill/WMI when testing a direct Python child on Windows.
        process.kill()
    process.wait()


def run_logged(command, cwd, env, log_path, timeout_seconds=0):
    """Stream output while a wall-clock watchdog also covers silent subprocesses."""
    timed_out = threading.Event()
    started = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', start_new_session=(os.name == 'posix'))

        def expire():
            if process.poll() is None:
                timed_out.set()
                terminate_process_tree(process)

        timer = threading.Timer(timeout_seconds, expire) if timeout_seconds else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            terminate_process_tree(process)
            raise
        finally:
            process.stdout.close()
            if timer:
                timer.cancel()
                timer.join()
    return (124 if timed_out.is_set() else code), timed_out.is_set(), time.monotonic() - started


def main():
    registry = json.loads((ROOT / 'Baselines/registry.json').read_text(encoding='utf-8'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=registry['methods'])
    p.add_argument('dataset')
    p.add_argument('--data-base', type=Path, default=os.environ.get('DATA_BASE'))
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default=os.environ.get('DEVICE', 'cuda:0'))
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=42, help='Classification recipe seed start; not the resource-study seed')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--bash', default='bash')
    p.add_argument('--timeout-seconds', type=float, default=0,
                   help='wall-clock limit for this complete launcher invocation; 0 disables. '
                        'Use --runs 1 for an independently timed seed.')
    p.add_argument('--epoch-timeout-seconds', type=float, default=None,
                   help='ANS-GT classification training-epoch limit; default 43200 (12 hours), '
                        '0 disables. This does not limit preprocessing or validation.')
    p.add_argument('--dry-run', action='store_true', help='Print the launcher plan; no training, imports of models, or file writes')
    a = p.parse_args()
    if not a.data_base and not a.data_dir:
        p.error('Set DATA_BASE, --data-base or --data-dir')
    if a.runs < 1 or a.seed_start < 0 or not math.isfinite(a.timeout_seconds) or a.timeout_seconds < 0:
        p.error('Invalid run count, seed, or wall-clock limit')
    if a.epoch_timeout_seconds is not None and a.method != 'ansgt':
        p.error('--epoch-timeout-seconds is currently implemented for ANS-GT classification only')
    epoch_timeout_seconds = (a.epoch_timeout_seconds if a.epoch_timeout_seconds is not None
                             else float(os.environ.get('EPOCH_TIMEOUT_SECONDS', 43200))) if a.method == 'ansgt' else 0
    if not math.isfinite(epoch_timeout_seconds) or epoch_timeout_seconds < 0:
        p.error('Invalid training-epoch limit')
    if os.environ.get('SPLIT_SUFFIXES', '').strip():
        p.error('Use one fixed processed split per invocation; do not rotate SPLIT_SUFFIXES')
    method = registry['methods'][a.method]
    folder = ROOT / 'Baselines' / method['folder']
    data = (a.data_dir or a.data_base / a.dataset).resolve()
    out = (a.out or ROOT / 'results/baselines' / a.method / a.dataset / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    env = dict(os.environ)
    # Some legacy launchers expect a dataset directory; all accept it directly.
    for key, value in dict(DATA_ROOT=data, DATA_DIR=data, OUT_DIR=out, OUT_ROOT=out,
                           PYTHON=a.python, DEVICE=a.device, RUNS=a.runs, NUM_RUNS=a.runs,
                           SEED=a.seed_start, BASE_SEED=a.seed_start, SEED_START=a.seed_start,
                           SCOPE_GT_RUN_MANIFEST_DIR=out / 'provenance').items():
        env[key] = str(value)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'Baselines'), str(folder), env.get('PYTHONPATH', '')])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONUTF8'] = '1'
    if a.method == 'ansgt':
        env['EPOCH_TIMEOUT_SECONDS'] = str(epoch_timeout_seconds)
    command = [a.bash, str(folder / method['launcher']), a.dataset]
    plan = dict(method=a.method, implementation=method['implementation'], cwd=str(folder),
                command=command, command_display=shlex.join(command), data_root=str(data), out=str(out),
                requested_runs=a.runs, seed_start=a.seed_start, timeout_seconds=a.timeout_seconds,
                epoch_timeout_seconds=epoch_timeout_seconds,
                environment={k: v for k, v in env.items() if k not in os.environ or v != os.environ[k]})
    inventory = ROOT / 'Baselines/source_inventory.json'
    if inventory.exists():
        plan['packaged_snapshot_sha256'] = json.loads(inventory.read_text(encoding='utf-8'))['snapshot_sha256']
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if out.exists() and any(out.iterdir()):
        p.error('Output is not empty; use a fresh --out to keep run identities separate')
    inputs = manifest(data)
    out.mkdir(parents=True, exist_ok=True)
    provenance = out / 'provenance'
    provenance.mkdir()
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'Cython', 'dgl', 'tqdm', 'numba'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Probe the selected interpreter, not the launcher's environment, if overridden.
    probe = subprocess.run([a.python, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (provenance / 'pip-freeze.txt').write_text(probe.stdout + probe.stderr, encoding='utf-8')
    for filename, value in [('launch.json', plan), ('inputs.json', inputs),
                            ('source.json', {str(f.relative_to(ROOT)): sha256(f) for f in (ROOT / 'Baselines').rglob('*')
                                             if f.is_file() and f.suffix in ('.py', '.pyx', '.h', '.hpp', '.c', '.cpp', '.json', '.sh', '.yaml', '.yml', '.toml', '.ini', '.cfg')
                                             and not any(part in ('__pycache__', 'build', '.git', 'cache', 'logs') or part.startswith('results')
                                                         for part in f.relative_to(ROOT / 'Baselines').parts[:-1])}),
                            ('launcher-runtime.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages))]:
        (provenance / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    code, timed_out, wall_seconds = run_logged(command, folder, env, out / 'stdout.log', a.timeout_seconds)
    epoch_records = sorted(provenance.glob('epoch_timeout_seed*_epoch*.json'))
    epoch_timed_out = code == 124 and not timed_out and bool(epoch_records)
    (provenance / 'exit.json').write_text(json.dumps(dict(
        exit_code=code, status='epoch_timeout' if epoch_timed_out else
        ('timeout' if timed_out else ('complete' if code == 0 else 'failed')),
        timeout_seconds=a.timeout_seconds, observed_wall_seconds=wall_seconds,
        epoch_timeout_seconds=epoch_timeout_seconds,
        epoch_timeout_records=[p.name for p in epoch_records],
        completed_utc=datetime.now(timezone.utc).isoformat())) + '\n', encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
