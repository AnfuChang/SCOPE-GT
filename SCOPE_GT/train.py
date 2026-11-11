"""Train and evaluate SCOPE-GT for node classification.

    python -m SCOPE_GT.train --preset roman_empire --data-root DATA/roman_empire --seeds 0 1 2 3 4

--preset loads starting settings of a dataset from presets.json (explicit arguments
override them). Checkpoints are selected on the validation split only. Every random choice is keyed by
(seed, stream, epoch, view, center[, node]), so a center's sample does not depend on batch composition.
Outputs: config.json, training_summary.csv (per seed + mean / sample std) and, per seed, history.csv,
metrics.json, checkpoint.pt and test_predictions.npz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ''):
    import importlib
    _folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(_folder.parent))
    __package__ = _folder.name
    importlib.import_module(__package__)

from .batch import DenseBatch, PackedBatch
from .cache import EvaluationDrawCache
from .data import load_graph, resolve_feature_normalize, resolve_split_suffix, OFFICIAL_MASK_DATASETS
from .graph import DIRECTION_MODES, CenterStream, DirectionIndex, GraphIndex, load_raw_adjacency
from .model import SCOPEGT
from .relation import TYPES, fit_statistic_scales
from .sampler import ContextSampler, trim_padding, ACCESS_KEYS, DIAGNOSTIC_KEYS
from .evidence_budget import EvidenceBudgetSampler
from .external_evidence import ExternalPathStatisticsSampler

PROTOCOL_ID = 'scope_gt_walk_tcb_v4_20261021'
PRESETS = Path(__file__).resolve().with_name('presets.json')
# Each stream is independent of batch composition and the relation-source switch.
SHUFFLE_TAG = 7801
TRAIN_TAGS = {'walk_first': 7802, 'walk_step': 7804, 'candidate': 7805, 'select': 7803, 'context_background': 7811}
EVAL_TAGS = {'walk_first': 5202, 'walk_step': 5204, 'candidate': 5205, 'select': 5203, 'context_background': 5211}
FIT_SEED, FIT_TAGS = 20261021, {'walk_first': 9102, 'walk_step': 9104, 'candidate': 9105,
                             'select': 9103, 'context_background': 9111}
AUC_DATASETS = ('questions', 'minesweeper', 'tolokers')
SOURCE_FILES = ('train.py', 'model.py', 'relation.py', 'sampler.py', 'cache.py', 'batch.py', 'graph.py', 'data.py',
                'packed_ops.py', 'presets.json', 'evidence_budget.py', 'external_evidence.py')


def make_streams(seed, tags, *parts):
    return {name: CenterStream(seed, tag, *parts) for name, tag in tags.items()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(results):
    """mean and sample-std rows appended to training_summary.csv."""
    keys = []
    for r in results:
        keys += [k for k in r if k not in keys]
    rows = []
    for name in ('mean', 'std'):
        row = {}
        for key in keys:
            values = [r.get(key) for r in results]
            numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)]
            if numeric:
                row[key] = float(np.mean(numeric)) if name == 'mean' else (
                    float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None)
            else:
                row[key] = values[0] if len(set(map(str, values))) == 1 else None
        row['seed'] = name
        rows.append(row)
    return rows


def claim_out_dir(path):
    """Create a fresh result directory; never overwrite (adds _1, _2, ...)."""
    path = Path(path)
    for index in range(1000):
        candidate = path if index == 0 else path.with_name(f'{path.name}_{index}')
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f'cannot create a result directory under {path.parent}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(logits, labels, metric):
    x = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    out = {'accuracy': float((x.argmax(-1) == y).double().mean()), 'loss': float(F.cross_entropy(x, y))}
    if metric == 'auc':
        from sklearn.metrics import roc_auc_score
        prob = torch.softmax(x, -1).numpy()
        try:
            out['auc'] = float(roc_auc_score(labels, prob[:, 1]) if prob.shape[1] == 2
                               else roc_auc_score(labels, prob, multi_class='ovr'))
        except ValueError:                             # split with a single class
            out['auc'] = float('nan')
    return out


def selection_score(metrics, metric):
    value = metrics['auc'] if metric == 'auc' else metrics['accuracy']
    return value if np.isfinite(value) else -float('inf')


class Tally:
    """Sampling / assembly cost of every view: graph accesses, contexts, evidence and selection diagnostics."""
    SUM_KEYS = ACCESS_KEYS + DIAGNOSTIC_KEYS + ('full_witness_records', 'attribute_sampled_groups')

    def __init__(self, hops, degree):
        self.hops, self.degree = hops, degree
        self.total = dict.fromkeys(self.SUM_KEYS + ('centers', 'contexts', 'context_degree', 'background_pairs',
                                                    'encoded_rows', 'kv_rows', 'slots'), 0.)
        self.hop_counts = np.zeros(hops + 1, np.int64)
        self.sample_seconds = self.assemble_seconds = 0.
        self.cache_hits = self.cache_misses = 0

    def add(self, draw, batch, sample_seconds, assemble_seconds, cache_hit=False):
        t, valid = self.total, draw['ctx_valid']
        for name in dict.fromkeys(self.SUM_KEYS):
            if not cache_hit or name not in ACCESS_KEYS:
                value = (draw.get(name, draw['witness_records'] if name == 'full_witness_records' else 0)
                         if name in ('full_witness_records', 'attribute_sampled_groups') else draw[name])
                t[name] += float(value)
        self.cache_hits += int(cache_hit)
        self.cache_misses += int(not cache_hit)
        t['centers'] += batch.size
        t['contexts'] += int(valid.sum())
        t['context_degree'] += float(self.degree[draw['ctx'][valid]].sum())
        t['background_pairs'] += int((draw['context_bg_count'] > 0).sum())
        t['encoded_rows'] += batch.encoded_rows
        t['kv_rows'] += batch.kv_count
        t['slots'] += getattr(batch, 'slots', batch.pairs)
        self.hop_counts += np.bincount(draw['sample_hop'][valid], minlength=self.hops + 1)[:self.hops + 1]
        self.sample_seconds += sample_seconds
        self.assemble_seconds += assemble_seconds

    def summary(self):
        t, c, contexts = self.total, max(self.total['centers'], 1), max(self.total['contexts'], 1)
        row = {'graph_access_per_center': sum(t[k] for k in ACCESS_KEYS) / c,
               'candidates_per_center': t['candidates'] / c, 'contexts_per_center': t['contexts'] / c,
               'context_degree_mean': t['context_degree'] / contexts,
               'witnesses_per_center': t['witness_records'] / c,
               'full_witnesses_per_center': t['full_witness_records'] / c,
               'attribute_sampled_groups_per_center': t['attribute_sampled_groups'] / c,
               'witness_external_share': t['witness_external_records'] / max(t['witness_records'], 1),
               'background_pair_coverage': t['background_pairs'] / contexts,
               'candidate_reads_per_center': t['candidate_reads'] / c,
               'distance_reads_per_center': t['distance_reads'] / c,
               'witness_reads_per_center': t['witness_reads'] / c,
               'edge_queries_per_center': t['edge_queries'] / c,
               'background_reads_per_center': t['context_background_reads'] / c,
               'candidate_truncated_share': t['candidate_truncated_centers'] / c,
               'background_records_per_center': t['background_records'] / c,
               'background_proposals_per_center': t['background_proposal_records'] / c,
               'three_hop_pair_share': t['three_hop_pairs'] / contexts,
               'encoded_rows_per_center': t['encoded_rows'] / c, 'kv_rows_per_center': t['kv_rows'] / c,
               'pad_fraction': 1. - t['contexts'] / max(t['slots'], 1),
               'sample_seconds': self.sample_seconds, 'assemble_seconds': self.assemble_seconds,
               'cache_hits': self.cache_hits, 'cache_misses': self.cache_misses}
        shares = self.hop_counts / max(self.hop_counts.sum(), 1)
        for hop in range(1, self.hops + 1):
            row[f'hop{hop}_share'] = float(shares[hop])
        return row


def draw_batch(sampler, centers, streams, features, device, execution, tally=None, cache=None):
    started = time.perf_counter()
    if cache is None:
        draw, cache_hit = trim_padding(sampler.draw(centers, streams)), False
    else:
        draw, cache_hit = cache.get(sampler, centers, streams)
    sampled = time.perf_counter()
    batch = (PackedBatch if execution == 'packed' else DenseBatch)(centers, draw, features, device)
    if tally is not None:
        tally.add(draw, batch, sampled - started, time.perf_counter() - sampled, cache_hit)
    return batch


@torch.no_grad()
def predict(model, sampler, centers, args, device, seed, views, features, tally=None, cache=None):
    """Average the logits of `views` fixed, seed-keyed samples of every center's context."""
    model.eval()
    total = None
    for view in range(views):
        streams = make_streams(seed, EVAL_TAGS, view)
        chunks = [model(draw_batch(sampler, centers[s:s + args.eval_batch_size], streams, features, device,
                                   args.execution, tally, cache)).float()
                  for s in range(0, len(centers), args.eval_batch_size)]
        logits = torch.cat(chunks)
        total = logits if total is None else total + logits
    return (total / views).cpu().numpy()


def make_sampler(args, graph, index, direction):
    record_budget = getattr(args, 'connection_record_budget', 0)
    use_attributes = args.connection_attributes == 'on'
    sampler_class = EvidenceBudgetSampler if record_budget or not use_attributes else ContextSampler
    extra = dict(record_budget=record_budget, use_attributes=use_attributes) if sampler_class is EvidenceBudgetSampler else {}
    sampler = sampler_class(index, direction, budget=args.context_budget, hops=args.max_hops,
                          hop_shares=args.hop_shares, background=args.background_size, walks=args.walks,
                          candidate_multiplier=args.candidate_multiplier,
                          relation_source=args.relation_source if args.relations == 'on' else 'none',
                          sampling=getattr(args, 'sampling', 'structured'), **extra)
    return (ExternalPathStatisticsSampler(sampler)
            if getattr(args, 'external_evidence', 'all') == 'structure_off' else sampler)


def build_model(args, graph, scales=None):
    model = SCOPEGT(graph.features.shape[1], int(graph.labels.max() + 1), dim=args.hidden_dim, heads=args.heads,
                   layers=args.layers, ff_dim=args.ff_dim, dropout=args.dropout, input_dropout=args.input_dropout,
                   enc_layers=args.enc_layers, rel_dim=args.rel_dim, hops=args.max_hops,
                   relation_scale_init=args.relation_scale_init, background_scale_init=args.background_scale_init,
                   topology_similarity=args.topology_similarity == 'on',
                   connection_attributes=args.connection_attributes == 'on',
                   connection_position=args.connection_position == 'on',
                   learn_scales=args.relation_scales == 'learned',
                   relations=args.relations == 'on', relation_drop=tuple(args.relation_drop), query=args.query,
                   external_attributes=getattr(args, 'external_evidence', 'all') != 'attributes_off')
    if scales is not None and model.relations:
        model.relation.set_scales(scales)
    return model


def fit_scales(args, sampler, graph):
    """Statistic scales of the connection evidence from a fixed sample of training centers (no labels).
    Uses the selected production sampler. Each relation-source
    control calibrates only inside its own graph; no labels are consulted."""
    train = graph.splits['train']
    rng = np.random.default_rng(np.random.SeedSequence([FIT_SEED, len(train)]))
    centers = np.sort(rng.choice(train, min(args.stat_fit_centers, len(train)), replace=False))
    streams = make_streams(FIT_SEED, FIT_TAGS)
    draws = [sampler.draw(centers[s:s + args.eval_batch_size], streams, relation_source=args.relation_source)
             for s in range(0, len(centers), args.eval_batch_size)]
    scales = fit_statistic_scales(draws, args.max_hops)
    scales['fit_centers'] = int(len(centers))
    scales.update(source=args.relation_source,
                  candidate_source='uniform_rw' if sampler.sampling == 'uniform_rw' else 'balanced_walk',
                  selection='uniform_pool' if sampler.sampling == 'uniform_rw' else 'uniform_strata')
    return scales


def train_one_seed(args, graph, sampler, features, seed, device, out_dir, scales, preprocess_seconds):
    seed_all(seed)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args, graph, scales).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    train_centers, val_centers, test_centers = graph.splits['train'], graph.splits['val'], graph.splits['test']
    epoch_val = val_centers
    if args.val_max_centers and args.val_max_centers < len(val_centers):   # per-epoch validation only
        pick = np.random.default_rng(np.random.SeedSequence([seed, 31337])).choice(
            len(val_centers), args.val_max_centers, replace=False)
        epoch_val = val_centers[np.sort(pick)]
    labels = graph.labels
    n_views = 2 if args.consistency > 0 else 1
    eval_cache = EvaluationDrawCache(int(args.eval_cache_mib * 2 ** 20)) if args.eval_cache == 'on' else None
    history, best, bad, epoch_seconds = [], {'score': -float('inf'), 'epoch': 0, 'state': None}, 0, []
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed, SHUFFLE_TAG, epoch]))
        view_streams = [make_streams(seed, TRAIN_TAGS, epoch, v) for v in range(n_views)]
        pool = train_centers
        if args.centers_per_epoch and args.centers_per_epoch < len(train_centers):
            pool = rng.choice(train_centers, args.centers_per_epoch, replace=False)
        order = rng.permutation(len(pool))
        tally = Tally(args.max_hops, sampler.graph.degree)
        loss_sum = torch.zeros((), device=device)
        sync(device)
        started = time.perf_counter()
        for start in range(0, len(pool), args.batch_size):
            centers = pool[order[start:start + args.batch_size]]
            batches = [draw_batch(sampler, centers, s, features, device, args.execution, tally) for s in view_streams]
            y = torch.as_tensor(labels[centers], device=device)
            outputs = [model(b) for b in batches]
            loss = sum(F.cross_entropy(o, y, label_smoothing=args.label_smoothing) for o in outputs) / len(outputs)
            if len(outputs) > 1:                       # consistency between two independent context samples
                p0, p1 = (torch.softmax(o, -1) for o in outputs)
                loss = loss + args.consistency * (p0 - p1).pow(2).sum(-1).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()
            loss_sum += loss.detach() * len(centers)
        sync(device)
        epoch_seconds.append(time.perf_counter() - started)
        schedule.step()
        train_loss = float(loss_sum) / len(pool)
        if not np.isfinite(train_loss):
            raise RuntimeError(f'nonfinite training loss at epoch {epoch + 1}')
        row = {'epoch': epoch + 1, 'train_loss': train_loss, f'val_{args.metric}': None}
        last = epoch == args.epochs - 1
        if last or (epoch + 1) % args.eval_every == 0:
            eval_start = time.perf_counter()
            val_tally = Tally(args.max_hops, sampler.graph.degree)
            val = scores(predict(model, sampler, epoch_val, args, device, seed, args.train_eval_contexts,
                                 features, val_tally, eval_cache),
                         labels[epoch_val], args.metric)
            row['val_seconds'] = time.perf_counter() - eval_start
            val_cost = val_tally.summary()
            for key in ('sample_seconds', 'assemble_seconds', 'graph_access_per_center', 'cache_hits', 'cache_misses'):
                row['val_' + key] = val_cost[key]
            row.update({'val_loss': val['loss'], 'val_accuracy': val['accuracy'], f'val_{args.metric}': val.get(args.metric)})
            score = selection_score(val, args.metric)
            if score > best['score'] + 1e-6 or best['state'] is None:
                best = {'score': score, 'epoch': epoch + 1,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
        row.update(tally.summary())
        row.update(supervised_centers=len(pool), training_centers_available=len(train_centers),
                   optimizer_steps=(len(pool) + args.batch_size - 1) // args.batch_size,
                   epoch_scope='complete_training_split' if len(pool) == len(train_centers) else 'budgeted_training_subset')
        row['epoch_seconds'] = epoch_seconds[-1]
        history.append(row)
        write_csv(out_dir / 'history.csv', history)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or last:
            shown = row.get(f'val_{args.metric}')
            print(f'seed={seed} epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.4f} '
                  f'val_{args.metric}={100 * shown if shown is not None else float("nan"):.2f} '
                  f'ctx={row["contexts_per_center"]:.1f} t={row["epoch_seconds"]:.1f}s '
                  f'(sample {row["sample_seconds"]:.1f}s, assemble {row["assemble_seconds"]:.1f}s) '
                  f'val_t={row.get("val_seconds", 0.):.1f}s '
                  f'val_cache={row.get("val_cache_hits", 0)}hit/{row.get("val_cache_misses", 0)}miss', flush=True)
        if args.patience and bad >= args.patience:
            print(f'seed={seed} early stop at epoch {epoch + 1}, best epoch {best["epoch"]}', flush=True)
            break
    train_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    model.load_state_dict(best['state'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    val = scores(predict(model, sampler, val_centers, args, device, seed, args.eval_contexts, features, cache=eval_cache),
                 labels[val_centers], args.metric)
    test_tally = Tally(args.max_hops, sampler.graph.degree)
    sync(device)
    inference_start = time.perf_counter()
    logits = predict(model, sampler, test_centers, args, device, seed, args.eval_contexts, features, test_tally, eval_cache)
    sync(device)
    inference_seconds = time.perf_counter() - inference_start
    inference_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    test = scores(logits, labels[test_centers], args.metric)
    torch.save({'state_dict': best['state'], 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'seed': seed, 'best_epoch': best['epoch'], 'statistic_scales': scales, 'protocol_id': PROTOCOL_ID,
                'dataset_sha256': graph.fingerprint}, out_dir / 'checkpoint.pt')
    np.savez_compressed(out_dir / 'test_predictions.npz', centers=test_centers, labels=labels[test_centers], logits=logits)
    steady = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {'seed': seed, 'best_epoch': best['epoch'], 'epochs_run': len(history),
              'val_accuracy': val['accuracy'], 'val_loss': val['loss'],
              'test_accuracy': test['accuracy'], 'test_loss': test['loss']}
    if args.metric == 'auc':
        result.update(val_auc=val['auc'], test_auc=test['auc'])
    result.update({
        'parameters': parameters, 'preprocess_seconds': preprocess_seconds,
        'epoch_seconds': float(np.mean(steady)), 'epoch_seconds_all': float(np.mean(epoch_seconds)),
        # first epoch through the best epoch: training plus per-epoch validation
        'time_to_best_seconds': float(sum(r['epoch_seconds'] + (r.get('val_seconds') or 0.)
                                          for r in history if r['epoch'] <= best['epoch'])),
        'train_seconds': float(np.sum(epoch_seconds)), 'inference_seconds': inference_seconds,
        'train_peak_allocated_bytes': train_peak, 'inference_peak_allocated_bytes': inference_peak})
    for key in history[0]:
        if key.endswith(('_per_center', '_share', '_coverage', '_mean', '_fraction')):
            values = [r[key] for r in history if r[key] is not None]
            result[f'train_{key}'] = float(np.mean(values)) if values else None
    result['train_sample_seconds'] = float(sum(r['sample_seconds'] for r in history))
    result['train_assemble_seconds'] = float(sum(r['assemble_seconds'] for r in history))
    result['validation_seconds'] = float(sum(r.get('val_seconds', 0.) for r in history))
    if eval_cache is not None:
        result.update({'eval_cache_' + key: value for key, value in eval_cache.statistics().items()})
    result['eval_cache_cost_scope'] = 'Build/lookup included in the first applicable validation or inference; CPU arrays only.'
    for key, value in test_tally.summary().items():
        result[f'test_{key}'] = value
    diagnostic_start = time.perf_counter()
    if args.relations == 'on':
        calibration = train_centers[:min(64, len(train_centers))]
        diagnostic_batch = draw_batch(sampler, calibration, make_streams(FIT_SEED, FIT_TAGS),
                                      features, device, args.execution)
        result.update(model.relation_diagnostics(diagnostic_batch))
        result['relation_diagnostic_seconds'] = time.perf_counter() - diagnostic_start
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default=None, help='load the settings of this dataset from presets.json')
    p.add_argument('--data-root', type=Path, required=True, help='directory with adj.npz feat.npy label.npy train/val/test.npy')
    p.add_argument('--dataset', default=None, help='dataset name for outputs (default: --preset, else folder name)')
    p.add_argument('--out-dir', type=Path, default=None)
    split = p.add_mutually_exclusive_group()
    split.add_argument('--split-suffix', default=None,
                       help='read train<suffix>.npy, val<suffix>.npy, test<suffix>.npy; explicit empty string selects legacy unsuffixed files')
    split.add_argument('--split-index', type=int, choices=range(10), default=None,
                       help='select one Platonov official fixed mask (0..9); default 0 for the four Platonov presets')
    p.add_argument('--feature-normalize', choices=['auto', 'l1', 'none'], default='auto',
                   help='auto: none for dense / mixed features, l1 for sparse bag-of-words')
    p.add_argument('--metric', choices=['auto', 'acc', 'auc'], default='auto',
                   help='auto: ROC-AUC for questions / minesweeper / tolokers, accuracy otherwise; also the selection metric')
    p.add_argument('--direction', choices=DIRECTION_MODES, default='auto',
                   help='auto: use edge direction when the stored adjacency is asymmetric')
    g = p.add_argument_group('contexts and their selection')
    g.add_argument('--sampling', choices=['structured', 'uniform_rw'], default='structured',
                   help='uniform_rw: ordinary unbiased walks and uniform pooled selection (No-S ablation); ignores hop shares')
    g.add_argument('--context-budget', type=int, default=32, help='K: maximum number of contexts per center')
    g.add_argument('--max-hops', type=int, choices=[1, 2, 3], default=2)
    g.add_argument('--hop-shares', type=float, nargs='+', default=None, help='quota share per true-distance bucket')
    g.add_argument('--walks', type=int, default=None, help='W: batch walks per center; default K, no retry or refill')
    g.add_argument('--candidate-multiplier', type=float, default=2., help='candidate limit / K')
    g = p.add_argument_group('relation evidence')
    g.add_argument('--background-size', type=int, default=4, help='proposals per unique context node BEFORE filtering; no refill')
    g.add_argument('--stat-fit-centers', type=int, default=1024)
    g.add_argument('--relation-source', choices=['graph', 'sequence'], default='graph',
                   help='sequence: recompute T/C/B in the center sequence induced subgraph')
    g.add_argument('--relations', choices=['on', 'off'], default='on',
                   help='off: identical selected tokens, no explicit relations or external evidence rows')
    g.add_argument('--relation-drop', nargs='*', choices=list(TYPES), default=[])
    g.add_argument('--external-evidence', choices=['all', 'attributes_off', 'structure_off'], default='all',
                   help='mask external attributes or external supporting-path statistics; retain tokens and endpoint topology')
    g.add_argument('--topology-similarity', choices=['on', 'off'], default='on')
    g.add_argument('--connection-attributes', choices=['on', 'off'], default='on')
    g.add_argument('--connection-record-budget', type=int, default=0,
                   help='C attribute records per pair/position; 0=full, 1..64=weighted samples; exact statistics retained')
    g.add_argument('--connection-position', choices=['on', 'off'], default='on')
    g.add_argument('--relation-scales', choices=['learned', 'fixed'], default='learned')
    g = p.add_argument_group('model')
    g.add_argument('--hidden-dim', type=int, default=128)
    g.add_argument('--rel-dim', type=int, default=64)
    g.add_argument('--heads', type=int, default=4)
    g.add_argument('--layers', type=int, default=2, help='center-query attention layers')
    g.add_argument('--ff-dim', type=int, default=256)
    g.add_argument('--enc-layers', type=int, default=2, help='residual MLP blocks of the token encoder')
    g.add_argument('--dropout', type=float, default=0.3)
    g.add_argument('--input-dropout', type=float, default=0.0, help='dropout on raw node features')
    g.add_argument('--relation-scale-init', type=float, default=0.05, help='initial lambda of the topology / connection residuals')
    g.add_argument('--background-scale-init', type=float, default=0.05, help='initial lambda of the background residual')
    g.add_argument('--query', choices=['center', 'context'], default='center',
                   help='context: contexts also query and update in the first L-1 layers (ablation; needs --execution dense)')
    g.add_argument('--execution', choices=['packed', 'dense'], default='packed', help='packed (final) or dense reference layout')
    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--patience', type=int, default=30, help='early stopping on validation; 0 = off')
    g.add_argument('--lr', type=float, default=0.001)
    g.add_argument('--weight-decay', type=float, default=0.0001, help='AdamW (decoupled) weight decay')
    g.add_argument('--label-smoothing', type=float, default=0.0)
    g.add_argument('--consistency', type=float, default=0.0, help='weight of the two-sample consistency loss; 0 = one sample')
    g.add_argument('--grad-clip', type=float, default=1.0)
    g.add_argument('--batch-size', type=int, default=64)
    g.add_argument('--eval-batch-size', type=int, default=256)
    g.add_argument('--centers-per-epoch', type=int, default=0, help='training centers drawn per epoch; 0 = all')
    g.add_argument('--eval-every', type=int, default=1)
    g.add_argument('--val-max-centers', type=int, default=0, help='subsample per-epoch validation (large graphs); 0 = all')
    g.add_argument('--train-eval-contexts', type=int, default=3, help='context samples averaged in per-epoch validation')
    g.add_argument('--eval-contexts', type=int, default=3, help='context samples averaged in the final evaluation')
    g.add_argument('--eval-cache', choices=['on', 'off'], default='on', help='bounded CPU-only cache of fixed evaluation draws')
    g.add_argument('--eval-cache-mib', type=float, default=512., help='logical NumPy payload cache capacity per seed')
    g.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                   help='independent training seeds (paper default: 0 1 2 3 4)')
    g.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    g.add_argument('--threads', type=int, default=6)
    g.add_argument('--log-every', type=int, default=10)
    g.add_argument('--variant', default=None, help='label recorded in config.json (set by the ablation launcher)')
    return p


def load_preset(name):
    presets = json.loads(PRESETS.read_text(encoding='utf-8'))['datasets']
    if name not in presets:
        raise SystemExit(f'no preset for {name!r}; available: {", ".join(sorted(presets))}')
    return presets[name]


def parse_args(argv=None):
    p = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--preset', default=None)
    known, _ = pre.parse_known_args(argv)
    if known.preset:
        values = load_preset(known.preset)
        unknown = set(values) - {a.dest for a in p._actions}
        if unknown:
            raise SystemExit(f'presets.json: unknown keys {sorted(unknown)}')
        p.set_defaults(**values)
    args = p.parse_args(argv)
    args.dataset = args.dataset or args.preset or Path(args.data_root).name
    args.split_suffix = resolve_split_suffix(args.dataset, args.split_suffix, args.split_index)
    if args.split_index is None and args.split_suffix in {f'_split{i}' for i in range(10)}:
        args.split_index = int(args.split_suffix.removeprefix('_split'))
    if args.external_evidence != 'all' and (args.relations != 'on' or args.relation_source != 'graph'):
        p.error('--external-evidence requires relations=on and relation-source=graph')
    if args.hop_shares is not None and len(args.hop_shares) != args.max_hops:
        p.error('--hop-shares needs one value per hop (--max-hops values)')
    if min(args.context_budget, args.max_hops, args.epochs, args.batch_size, args.eval_batch_size,
           args.eval_every, args.eval_contexts, args.train_eval_contexts, args.threads, args.log_every,
           args.stat_fit_centers) < 1 or (args.walks is not None and args.walks < 1):
        p.error('budgets, epochs, batch sizes and evaluation settings must be positive')
    if args.background_size < 0:
        p.error('background proposal count must be nonnegative')
    if not 0 <= args.connection_record_budget <= 64:
        p.error('connection record budget must be in 0..64')
    if not (0 <= args.dropout < 1 and 0 <= args.input_dropout < 1):
        p.error('dropout rates must lie in [0,1)')
    if not np.isfinite(args.candidate_multiplier) or args.candidate_multiplier < 1:
        p.error('candidate multiplier must be finite and at least one')
    if not np.isfinite(args.eval_cache_mib) or args.eval_cache_mib <= 0:
        p.error('evaluation cache capacity must be finite and positive')
    if args.hop_shares is not None and (not np.isfinite(args.hop_shares).all() or min(args.hop_shares) < 0
                                       or sum(args.hop_shares) <= 0):
        p.error('hop shares must be finite, nonnegative and have positive sum')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('seeds must be distinct nonnegative integers')
    if args.query == 'context' and (args.execution != 'dense' or args.layers < 2):
        p.error('--query context needs --execution dense and at least two layers')
    return args


def variant_tag(args):
    tag = f'k{args.context_budget}_h{args.max_hops}'
    for flag, value, default in (('sampler', args.sampling, 'structured'),
                                 ('rel', args.relations, 'on'), ('src', args.relation_source, 'graph'),
                                 ('q', args.query, 'center'), ('exec', args.execution, 'packed')):
        if value != default:
            tag += f'_{flag}{value}'
    if args.relation_drop:
        tag += '_no' + '-'.join(args.relation_drop)
    return tag


def main(argv=None):
    args = parse_args(argv)
    dataset = args.dataset
    args.feature_normalize = resolve_feature_normalize(dataset, args.feature_normalize)
    if args.metric == 'auto':
        args.metric = 'auc' if dataset.lower().replace('-', '_') in AUC_DATASETS else 'acc'
    args.metric = 'accuracy' if args.metric == 'acc' else args.metric
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if args.out_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_dir = Path(__file__).resolve().parent / 'results' / f'{dataset}_{variant_tag(args)}_{stamp}'
    args.out_dir = claim_out_dir(args.out_dir)
    prep_start = time.perf_counter()
    graph = load_graph(args.data_root, args.split_suffix, args.feature_normalize)
    graph_index = GraphIndex(graph.adj)
    raw = load_raw_adjacency(args.data_root) if args.direction != 'off' else None
    direction = DirectionIndex(raw, graph_index, args.direction)
    del raw
    index_start = time.perf_counter()
    sampler = make_sampler(args, graph, graph_index, direction)
    index_seconds = time.perf_counter() - index_start
    fit_start = time.perf_counter()
    scales = fit_scales(args, sampler, graph) if args.relations == 'on' else None
    fit_seconds = time.perf_counter() - fit_start
    features = graph.features
    if device.type == 'cuda' and features.nbytes <= 2 * 1024 ** 3:     # keep features on the GPU when they fit
        features = torch.as_tensor(features, device=device)
    sync(device)
    preprocess_seconds = time.perf_counter() - prep_start
    here = Path(__file__).resolve().parent
    config = {'algorithm': 'SCOPE-GT', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'variant': args.variant,
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'preset': load_preset(args.preset) if args.preset else None,
              'command': ' '.join(sys.argv), 'torch': torch.__version__, 'device': str(device),
              'num_nodes': len(graph.labels), 'num_undirected_edges': graph_index.nnz // 2,
              'num_features': graph.features.shape[1], 'num_classes': int(graph.labels.max() + 1),
              'splits': {k: len(v) for k, v in graph.splits.items()}, 'dataset_sha256': graph.fingerprint,
              'split_protocol': graph.split_metadata,
              'graph_fingerprint': sampler.fingerprint,
              'source_sha256': {name: sha256(here / name) for name in SOURCE_FILES},
              'direction': {'directed': direction.directed, 'arcs': direction.arcs, 'note': direction.note},
              'hop_quota': sampler.base_quota.tolist() if sampler.sampling == 'structured' else None,
              'sampling': {'mode': 'uniform_rw_uniform_pool' if sampler.sampling == 'uniform_rw' else 'balanced_walk_uniform_strata',
                           'walks': sampler.walks, 'uses_hop_quotas': sampler.sampling == 'structured',
                           'index_bytes': sampler.index_bytes, 'index_seconds': index_seconds,
                           'candidate_limit': sampler.candidate_limit, 'max_walk_moves': sampler.candidate_access_budget,
                           'background_proposals_per_unique_context': sampler.background},
              'statistic_scales': scales, 'statistic_fit_seconds': fit_seconds, 'preprocess_seconds': preprocess_seconds}
    (args.out_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[SCOPE-GT] {dataset}: nodes={len(graph.labels)} edges={graph_index.nnz // 2} device={device}; {direction.note}', flush=True)
    print(f'[SCOPE-GT] K={args.context_budget} H={args.max_hops} '
          f'quota={sampler.base_quota.tolist() if sampler.sampling == "structured" else "none"} '
          f'W={sampler.walks} M={sampler.candidate_limit} moves<={sampler.candidate_access_budget} '
          f'sampling={sampler.sampling} selection=uniform relations={args.relations}/{args.relation_source} '
          f'query={args.query} execution={args.execution}', flush=True)
    results = []
    for seed in args.seeds:
        out = args.out_dir / f'seed{seed}'
        out.mkdir(parents=True)
        result = train_one_seed(args, graph, sampler, features, seed, device, out, scales, preprocess_seconds)
        results.append(result)
        write_csv(args.out_dir / 'training_summary.csv', results + aggregate_rows(results))
        print(f'seed={seed} best_epoch={result["best_epoch"]} test_{args.metric}={100 * result[f"test_{args.metric}"]:.2f}', flush=True)
    print_summary(args, dataset, results, sampler)
    print('Results: ' + str(args.out_dir.resolve()), flush=True)


def print_summary(args, dataset, results, sampler):
    mean, std = aggregate_rows(results)
    show = lambda key: f'{100 * mean[key]:.2f} ± {100 * (std[key] or 0.):.2f}'
    print(f'\n==== {dataset}: {len(results)} seed(s), selected by validation {args.metric} ====')
    print(f'Test {args.metric}: {show(f"test_{args.metric}")}')
    print(f'Val  {args.metric}: {show(f"val_{args.metric}")}')
    ttb = np.asarray([r['time_to_best_seconds'] for r in results])
    peaks = [r['train_peak_allocated_bytes'] for r in results if r['train_peak_allocated_bytes'] is not None]
    print(f'Params: {results[0]["parameters"]:,} | train time/epoch (s): {np.mean([r["epoch_seconds_all"] for r in results]):.3f} | '
          f'time to best (s): {ttb.mean():.1f} | preprocessing (s): {results[0]["preprocess_seconds"]:.2f} | '
          + (f'train peak GPU mem (MB): {max(peaks) / 1024 ** 2:.1f}' if peaks else 'peak GPU mem: n/a (CPU)'), flush=True)


if __name__ == '__main__':
    main()
