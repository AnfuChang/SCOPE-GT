"""Measurement/provenance utilities. No GPU work is performed by the launcher."""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
from SCOPE_GT.external_backbones import baseline_path, source_key

CODE = Path(__file__).resolve().parents[1]
FILES = ('adj.npz', 'feat.npy', 'label.npy', 'train.npy', 'val.npy', 'test.npy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def dataset_manifest(root):
    root = Path(root)
    names = list(FILES)
    metadata = root / 'metadata.json'
    if metadata.is_file():
        info = json.loads(metadata.read_text(encoding='utf-8'))
        if info.get('split_policy') in ('platonov_official_10_fixed_50_25_25', 'ogb_official_sales_ranking'):
            names.append('metadata.json')
            names.extend(f'{part}_split{i}.npy' for i in info.get('official_split_ids', [])
                         for part in ('train', 'val', 'test'))
    return {name: sha(root / name) for name in names}


def source_manifest(method):
    """Fingerprint execution and model sources, excluding presentation artifacts.

    Explicit package files keep plotting, tests, examples and generated tables
    from invalidating pending workers. Changes to measurement code or models
    still invalidate the plan; this scope change requires a fresh result root.
    """
    package = CODE / 'SCOPE_GT_budgets'
    files = [package / name for name in (
        '__init__.py', 'adapters.py', 'common.py', 'run.py', 'worker.py', 'profiles.json')]
    files.append(CODE / 'SCOPE_GT/external_backbones.py')
    folders = [CODE / 'SCOPE_GT'] if method == 'scope_gt' else [baseline_path({
        'nlgt': 'NLGT', 'nagphormer': 'NAGphormer', 'sgformer': 'SGFormer'}[method]),
        baseline_path('common')]
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if any(part in ('results', '__pycache__', '.git', 'figures', 'logs', 'checkpoints')
                   for part in p.relative_to(folder).parts):
                continue
            if p.is_file() and p.suffix in ('.py', '.json', '.sh'):
                files.append(p)
    result = {source_key(p): sha(p) for p in files}
    return result


def rss_bytes():
    """Current RSS, and OS lifetime high-water RSS; byte units on both supported OSes."""
    if sys.platform.startswith('linux'):
        values = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, *_ = line.split()
                values[key] = int(value) * 1024
        return values.get('VmRSS:', 0), values.get('VmHWM:', 0)
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        p = Counters()
        p.cb = ctypes.sizeof(p)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(p), p.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(p.WorkingSetSize), int(p.PeakWorkingSetSize)
    raise RuntimeError('RSS measurement currently supports Linux and Windows')


class RSSMonitor:
    """No psutil dependency; workers are deliberately single-process (DataLoader workers=0)."""
    def __init__(self, interval=.02):
        self.interval, self.peak, self.stop_event = interval, 0, threading.Event()

    def __enter__(self):
        self.peak = rss_bytes()[0]
        def poll():
            while not self.stop_event.wait(self.interval):
                self.peak = max(self.peak, rss_bytes()[0])
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, rss_bytes()[0])


def sync(device):
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


class Meter:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        import torch
        sync(self.device)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.rss = RSSMonitor().__enter__()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        sync(self.device)
        self.seconds = time.perf_counter() - self.start
        self.rss.__exit__(*exc)
        self.result = {'seconds': self.seconds, 'rss_sampled_peak_bytes': self.rss.peak,
                       'gpu_peak_allocated_bytes': None, 'gpu_peak_reserved_bytes': None}
        if self.device.type == 'cuda':
            self.result.update(gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                               gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)))


def array_bytes(*objects):
    """Unique underlying ndarray/tensor storage bytes, not Python object overhead or RSS."""
    import numpy as np
    import scipy.sparse as sp
    import torch
    seen, storage = set(), {}
    def visit(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, np.ndarray):
            base = x
            while isinstance(base.base, np.ndarray):
                base = base.base
            storage[('cpu', int(base.__array_interface__['data'][0]))] = int(base.nbytes)
        elif torch.is_tensor(x):
            if x.layout == torch.strided:
                s = x.untyped_storage()
                storage[(str(x.device), s.data_ptr())] = s.nbytes()
            elif x.layout == torch.sparse_coo:
                visit(x._indices())
                visit(x._values())
        elif sp.issparse(x):
            for name in ('data', 'indices', 'indptr', 'row', 'col'):
                if hasattr(x, name):
                    visit(getattr(x, name))
        elif isinstance(x, dict):
            for item in x.values():
                visit(item)
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        elif hasattr(x, '__dict__') and not isinstance(x, torch.nn.Module):
            visit(vars(x))
    for obj in objects:
        visit(obj)
    return sum(storage.values())


@contextlib.contextmanager
def lock(path):
    """Never break somebody else's lock automatically, including after a crashed job."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Active/stale lock exists: {path}; inspect its PID before removing it') from None
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'host': platform.node(), 'started': time.time()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)
