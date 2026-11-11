"""CPU-only checks that epochs and independent seeds are aggregated separately."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from SCOPE_GT_batch_size.collect import cost, cost_runs, sharing


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class SeedAggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_cost(self, seed, requested=tuple(range(5)), legacy=False):
        folder = self.root / 'wikics' / 'cost' / 'B256'
        run = folder if legacy else folder / f'seed{seed}'
        save_json(run / 'request.json', dict(seed=seed, requested_seeds=list(requested)))
        save_json(run / 'equivalence.json', [{'layout': 'packed', 'passed': True}])
        for layout, scale in [('dedup_packed', 1), ('packed', 2)]:
            rows = [dict(warmup_epoch=True, epoch_seconds=10000, train_peak_alloc_mib=10000)]
            rows += [dict(warmup_epoch=False, epoch_seconds=scale * (seed + offset),
                          train_peak_alloc_mib=scale * (10 * seed + memory), optimizer_steps=3,
                          centers=580, sample_s=.1, assemble_s=.2, compute_s=.3)
                     for offset, memory in [(.5, 15), (1.5, 20)]]
            save_json(run / f'epoch_{layout}.json', dict(status='ok', rows=rows))

    def test_cost_uses_per_seed_means_and_peaks(self):
        for seed in range(5):
            self.make_cost(seed)
        rows = {r['layout']: r for r in cost(self.root)}
        row = rows['dedup_packed']
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['completed_runs'], 5)
        self.assertEqual(row['measured_epochs'], 2)
        self.assertAlmostEqual(row['epoch_seconds_mean'], 3.)
        self.assertAlmostEqual(row['epoch_seconds_sd'], np.std([1, 2, 3, 4, 5], ddof=1))
        self.assertAlmostEqual(row['train_peak_alloc_mib'], 40.)
        self.assertAlmostEqual(row['time_reduction_vs_packed'], .5)
        self.assertEqual(len(cost_runs(self.root)), 10)

    def test_missing_seed_is_counted_without_filling(self):
        for seed in range(4):
            self.make_cost(seed)
        for row in cost(self.root):
            self.assertEqual(row['status'], 'incomplete')
            self.assertEqual(row['requested_runs'], 5)
            self.assertEqual(row['completed_runs'], 4)

    def test_legacy_one_seed_keeps_within_run_sd_separate(self):
        self.make_cost(0, requested=(0,), legacy=True)
        row = next(r for r in cost(self.root) if r['layout'] == 'dedup_packed')
        self.assertEqual(row['completed_runs'], 1)
        self.assertIsNone(row['epoch_seconds_sd'])
        self.assertAlmostEqual(row['epoch_seconds_within_run_sd_mean'], np.std([.5, 1.5], ddof=1))

    def test_legacy_and_multiseed_outputs_are_not_pooled(self):
        self.make_cost(0, requested=(0,), legacy=True)
        self.make_cost(0)
        with self.assertRaisesRegex(ValueError, 'Legacy and multi-seed'):
            cost(self.root)

    def test_sharing_averages_seed_statistics(self):
        for seed in range(5):
            run = self.root / 'wikics' / 'sharing' / f'seed{seed}'
            save_json(run / 'request.json', dict(seed=seed, requested_seeds=list(range(5))))
            row = dict(dataset='wikics', seed=seed, batch_size=256, batches=3, epoch_centers=580,
                       centers=580, reads=100, unique_tokens=80-seed*10, encoded_occurrences=200,
                       unique_encoded=100, token_repeat_fraction=.2+seed*.1, encoded_repeat_fraction=.5,
                       reads_per_unique_token=100/(80-seed*10), occurrences_per_unique_encoded=2,
                       inputs_identical_to_first_B=True, seconds=seed+1)
            with (run / 'summary.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        row, = sharing(self.root)
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['completed_runs'], 5)
        self.assertAlmostEqual(row['token_repeat_fraction'], .4)
        self.assertAlmostEqual(row['token_repeat_fraction_sd'], np.std([.2, .3, .4, .5, .6], ddof=1))
        self.assertTrue(row['inputs_identical_to_first_B'])


if __name__ == '__main__':
    unittest.main()
