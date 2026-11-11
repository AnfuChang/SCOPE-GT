"""Worker for the decomposition arms: the unchanged Table IX worker with decomposed adapters.

Training loop, validation selection, checkpointing, timing and memory measurement are those of
SCOPE_GT_transfer.worker, so V1-V3 are measured exactly like the existing Native/+SCMB runs.
"""
import argparse
import json
from pathlib import Path

from . import worker as base
from .decompose import make_decomposed_adapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--probe')
    a = p.parse_args()
    if a.probe:
        print(json.dumps(base.runtime(a.probe)))
        return
    if not a.spec:
        p.error('--spec required')
    spec = json.loads(a.spec.read_text(encoding='utf-8'))
    base.TransferAdapter = make_decomposed_adapter(int(spec.get('v1_pad_multiple', 1)))
    if not base.run(spec):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
