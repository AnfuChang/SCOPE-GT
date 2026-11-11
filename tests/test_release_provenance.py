"""CPU checks for input identity and packaged adapter discovery; no training."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from input_manifest import manifest
from run_baseline import run_logged
from SCOPE_GT.external_backbones import baseline_root, baseline_path
sys.path.insert(0, str(ROOT / 'Baselines'))
from common.epoch_deadline import EpochDeadline


class InputIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        np.save(self.root / 'feat.npy', np.arange(12, dtype=np.float32).reshape(6, 2))
        np.save(self.root / 'label.npy', np.array([0, 1, 0, 1, 0, 1]))
        # Actual SciPy CSR NPZ fields, generated without a SciPy dependency.
        np.savez(self.root / 'adj.npz', format=np.array(b'csr'), shape=[6, 6],
                 data=np.ones(2), indices=[1, 0], indptr=[0, 1, 2, 2, 2, 2, 2])
        for split, ids in [('train', [0, 1]), ('val', [2, 3]), ('test', [4, 5])]:
            np.save(self.root / f'{split}.npy', ids)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_changes_when_actual_split_changes(self):
        first = manifest(self.root)
        np.save(self.root / 'train.npy', [1, 0])
        second = manifest(self.root)
        self.assertNotEqual(first['input_id'], second['input_id'])
        self.assertEqual(first['files']['feat.npy']['sha256'], second['files']['feat.npy']['sha256'])

    def test_boolean_masks_and_explicit_split_suffix(self):
        for split in ('train', 'val', 'test'):
            ids = np.load(self.root / f'{split}.npy')
            mask = np.zeros(6, dtype=bool)
            mask[ids] = True
            np.save(self.root / f'{split}_split3.npy', mask)
        value = manifest(self.root, '_split3')
        self.assertEqual(value['files']['train_split3.npy']['centers'], 2)
        self.assertNotIn('train.npy', value['files'])

    def test_rejects_overlapping_splits(self):
        np.save(self.root / 'val.npy', [1, 3])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            manifest(self.root)

    def test_rejects_noninteger_ids(self):
        np.save(self.root / 'test.npy', [4.0, 5.0])
        with self.assertRaisesRegex(ValueError, 'integer'):
            manifest(self.root)

    def test_rejects_invalid_graph_shape(self):
        np.savez(self.root / 'adj.npz', format=np.array(b'csr'), shape=[5, 5], data=[], indices=[], indptr=[0]*6)
        with self.assertRaisesRegex(ValueError, 'Adjacency shape'):
            manifest(self.root)


class AdapterReleaseTests(unittest.TestCase):
    def test_bundled_lookup_and_optional_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(baseline_root(), ROOT / 'Baselines')
            self.assertTrue(baseline_path('Gophormer/gophormer_npz.py').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'SCOPE_GT_BASELINES_ROOT': tmp}):
                self.assertEqual(baseline_root(), Path(tmp).resolve())

    def test_registry_entrypoints_and_no_training_launch_plans(self):
        registry = json.loads((ROOT / 'Baselines/registry.json').read_text(encoding='utf-8'))
        self.assertEqual(len(registry['methods']), 14)
        for method, row in registry['methods'].items():
            with self.subTest(method=method):
                for name in ('entry', 'launcher'):
                    self.assertTrue((ROOT / 'Baselines' / row['folder'] / row[name]).is_file())
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_baseline.py'), method,
                                         'wikics', '--data-base', str(ROOT / 'data'), '--dry-run'],
                                        capture_output=True, text=True, check=True)
                plan = json.loads(result.stdout)
                self.assertEqual(plan['requested_runs'], 5)
                self.assertEqual(plan['epoch_timeout_seconds'], 43200 if method == 'ansgt' else 0)
                self.assertTrue(plan['environment']['OUT_ROOT'].endswith(plan['out']))

    def test_effective_configuration_is_preserved(self):
        path = ROOT / 'Baselines/common/provenance.py'
        spec = importlib.util.spec_from_file_location('release_provenance', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'SCOPE_GT_RUN_MANIFEST_DIR': tmp}):
                module.record_config(argparse.Namespace(batch_size=128, metric='auc'), __file__)
            result = json.loads(next(Path(tmp).glob('effective_config_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(result['config'], dict(batch_size=128, metric='auc'))
            self.assertEqual(result['runtime']['executable'], sys.executable)


class BaselineDeadlineTests(unittest.TestCase):
    def test_training_deadline_excludes_work_outside_each_epoch(self):
        # Total process time exceeds the limit; each individually timed epoch does not.
        import time
        with tempfile.TemporaryDirectory() as tmp:
            time.sleep(1.02)
            for epoch in (1, 2):
                with EpochDeadline(1, seed=42, epoch=epoch, record_dir=tmp):
                    time.sleep(0.02)
            self.assertEqual(list(Path(tmp).glob('epoch_timeout*.json')), [])

    def test_silent_training_epoch_records_timeout_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = ('import sys,time,threading; sys.path.insert(0,sys.argv[1]); '
                     'real_timer=threading.Timer; '
                     'threading.Timer=lambda interval,callback: real_timer(0.001,callback); '
                     'from common.epoch_deadline import EpochDeadline; '
                     'deadline=EpochDeadline(0.1,seed=43,epoch=2,record_dir=sys.argv[2]); '
                     'deadline.__enter__(); time.sleep(60)')
            result = subprocess.run([sys.executable, '-c', child, str(ROOT / 'Baselines'), tmp],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 124)
            record = json.loads((Path(tmp) / 'epoch_timeout_seed43_epoch2.json').read_text(encoding='utf-8'))
            self.assertEqual(record['scope'], 'training_epoch')
            self.assertEqual(record['epoch'], 2)
            self.assertGreaterEqual(record['observed_epoch_seconds'], 0.1)

    def test_completed_process_keeps_output_and_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'stdout.log'
            code, timed_out, wall = run_logged(
                [sys.executable, '-c', "print('completed'); raise SystemExit(3)"],
                ROOT, dict(os.environ), log, timeout_seconds=10)
            self.assertEqual(code, 3)
            self.assertFalse(timed_out)
            self.assertIn('completed', log.read_text(encoding='utf-8'))
            self.assertGreater(wall, 0)

    def test_silent_process_is_stopped_at_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, timed_out, wall = run_logged(
                [sys.executable, '-c', 'import time; time.sleep(60)'],
                ROOT, dict(os.environ), Path(tmp) / 'stdout.log', timeout_seconds=0.1)
            self.assertEqual(code, 124)
            self.assertTrue(timed_out)
            self.assertLess(wall, 15)


if __name__ == '__main__':
    unittest.main()
