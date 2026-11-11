"""Save parsed adapter arguments without changing the training protocol."""
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import sys


def record_config(args, source):
    """Called after argument parsing; enabled by scripts/run_baseline.py."""
    directory = os.environ.get('SCOPE_GT_RUN_MANIFEST_DIR')
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'torch-geometric', 'Cython', 'dgl', 'numba'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch = sys.modules.get('torch')
    value = dict(entry=str(Path(source).resolve()), argv=sys.argv,
                 recorded_utc=datetime.now(timezone.utc).isoformat(),
                 runtime=dict(python=sys.version, executable=sys.executable, packages=packages,
                              torch_cuda=getattr(getattr(torch, 'version', None), 'cuda', None)),
                 config=vars(args))
    target = root / f'effective_config_{os.getpid()}.json'
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
