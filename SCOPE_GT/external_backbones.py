"""Locate bundled NPZ backbone adapters, with an optional external override.

Set SCOPE_GT_BASELINES_ROOT to replace the bundled Baselines tree with, for example,
NLGT/nlgt_npz.py and common/data.py. These experiments use the NPZ adapter
interfaces; an unmodified upstream model repository is not a substitute.
SCOPE-GT training and model-specific ablations do not require this setting.
"""
from __future__ import annotations

import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'SCOPE_GT_BASELINES_ROOT'
EXTERNAL_PREFIX = 'external-baselines/'


def baseline_root(root=None):
    value = root if root is not None else os.environ.get(ENVIRONMENT) or CODE / 'Baselines'
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'{ENVIRONMENT} is not an adapter directory: {folder}')
    return folder


def baseline_path(relative, root=None):
    folder = baseline_root(root)
    path = (folder / relative).resolve()
    path.relative_to(folder)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing NPZ adapter path: {path}. {ENVIRONMENT}, if set, must point to '
            'the compatible adapter tree, not just an upstream model checkout.')
    return path


def source_key(path):
    """Use location-independent manifest keys for local and external sources."""
    path = Path(path).resolve()
    try:
        return path.relative_to(CODE).as_posix()
    except ValueError:
        return EXTERNAL_PREFIX + path.relative_to(baseline_root()).as_posix()


def source_path(key):
    if key.startswith(EXTERNAL_PREFIX):
        return baseline_path(key[len(EXTERNAL_PREFIX):])
    path = (CODE / key).resolve()
    path.relative_to(CODE)
    return path
