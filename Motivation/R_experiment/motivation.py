"""R information utility: Plain / Witness / Strength / Neighborhood / All.

Original-graph evidence is used regardless of sampled-sequence membership.
Every condition shares the same two-layer GT, node sequence and parameters.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .graph import GraphTables, load_graph
from .relation import sample_sequences
from .information import (CHANNELS, MODES, Coverage, InformationReader, InformationTables,
                          RelationOracle, active_channels)

PROTOCOL = "R_information_v3"
LABELS = dict(zip(MODES, ("Plain", "Witness", "Strength", "Neighborhood", "All")))
DEFAULT_K = (4, 8, 16, 32, 64)


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with Path(path).open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)


def fingerprint(root):
    result = {}
    for name in ('adj.npz', 'feat.npy', 'label.npy', 'train.npy', 'val.npy', 'test.npy'):
        digest = hashlib.sha256()
        with (Path(root) / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def sequence_cache(oracle, split, k, seed, args):
    cache, coverage = {}, {}
    channels = set().union(*(active_channels(mode) for mode in args.modes))
    for code, name in enumerate(('val', 'test')):
        rng = np.random.default_rng([seed, 23, code])
        cache[name], tally = [], Coverage()
        for start in range(0, len(split[name]), args.eval_batch_size):
            centers = split[name][start:start + args.eval_batch_size]
            contexts, _ = sample_sequences(oracle.graph, centers, k, rng)
            table = InformationTables(oracle, centers, contexts, channels)
            cache[name].append((centers, table))
            tally.add(table)
        coverage[name] = tally.summary()
    return cache, coverage


def calibrate(oracle, train_ids, args):
    """One label-free positive reference for count/mass, shared by every mode/K."""
    if not any('strength' in active_channels(mode) for mode in args.modes):
        return [1.0, 1.0]
    rng = np.random.default_rng([71])
    pick = np.sort(rng.choice(train_ids, min(len(train_ids), args.calib_centers), replace=False))
    positive = [[], []]
    for start in range(0, len(pick), args.eval_batch_size):
        centers = pick[start:start + args.eval_batch_size]
        contexts, _ = sample_sequences(oracle.graph, centers, max(args.k), rng)
        table = InformationTables(oracle, centers, contexts, {'strength'})
        for i in range(2):
            positive[i].append(table.strength_raw[..., i][table.strength_raw[..., i] > 0])
    tau = []
    for chunks in positive:
        values = np.concatenate(chunks)
        tau.append(max(float(np.median(values)), 1e-8) if len(values) else 1.0)
    return tau


def metric_scores(logits, labels, metric):
    logits = np.asarray(logits)
    result = {'accuracy': float(np.mean(logits.argmax(1) == labels))}
    if metric == 'auc':
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) != 2 or logits.shape[1] != 2:
            raise ValueError('ROC-AUC requires both classes in the binary split')
        prob = torch.softmax(torch.as_tensor(logits), 1).numpy()[:, 1]
        result['auc'] = float(roc_auc_score(labels, prob))
    return result


def run_one(mode, k, seed, oracle, features, labels, split, cache, tau, args, device):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = InformationReader(features.shape[1], int(labels.max()) + 1, args.hidden_dim, args.heads,
                              args.ff_dim, args.dropout, args.channel_init).to(device)
    x, y = torch.as_tensor(features, device=device), torch.as_tensor(labels, device=device)
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    primary = 'auc' if args.metric == 'auc' else 'accuracy'

    def inputs(table):
        ids = torch.as_tensor(np.maximum(table.nodes, 0), device=device)
        return (x[ids], torch.as_tensor(table.valid, device=device),
                *(torch.as_tensor(value, device=device) for value in table.inputs(mode, tau)))

    @torch.no_grad()
    def evaluate(name):
        model.eval()
        result = np.concatenate([model(*inputs(table)).cpu().numpy() for centers, table in cache[name]])
        return metric_scores(result, labels[split[name]], args.metric), result

    best, bad, graph_reads, attr_reads, contexts_total, centers_total = None, 0, 0, 0, 0, 0
    sequence_hash = hashlib.sha256()
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng([seed, 11, epoch])
        order = split['train'][rng.permutation(len(split['train']))]
        for start in range(0, len(order), args.batch_size):
            centers = order[start:start + args.batch_size]
            contexts, _ = sample_sequences(oracle.graph, centers, k, rng)
            table = InformationTables(oracle, centers, contexts, active_channels(mode))
            sequence_hash.update(table.nodes.tobytes())
            graph_reads += table.graph_reads
            attr_reads += table.attribute_reads
            centers_total += len(centers); contexts_total += int(table.valid[:, 1:].sum())
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(*inputs(table)), y[torch.as_tensor(centers, device=device)])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        scheduler.step()
        scores, _ = evaluate('val')
        history.append({'epoch': epoch + 1, **{'val_' + key: value for key, value in scores.items()}})
        if best is None or scores[primary] > best['score'] + 1e-6:
            best = {'score': scores[primary], 'epoch': epoch + 1, 'state': copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f'  K={k} seed={seed} {mode} epoch={epoch + 1} val_{primary}={scores[primary]:.4f}', flush=True)
        if args.patience and bad >= args.patience:
            break
    model.load_state_dict(best['state'])
    val, _ = evaluate('val'); test, logits = evaluate('test')
    run_dir = Path(args.out) / f'K{k}' / f'seed{seed}' / mode
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), run_dir / 'checkpoint.pt')
    np.savez_compressed(run_dir / 'test_predictions.npz', centers=split['test'], labels=labels[split['test']], logits=logits)
    write_csv(run_dir / 'history.csv', history)
    row = {'protocol': PROTOCOL, 'schema_version': 3, 'dataset': args.dataset, 'mode': mode, 'seed': seed, 'K': k,
           'metric': args.metric, 'hops': args.hops,
           'best_epoch': best['epoch'], 'epochs_run': len(history), 'seconds': time.perf_counter() - started,
           'parameters': sum(p.numel() for p in model.parameters()), 'initial_state_sha256': initial_hash,
           'train_sequence_sha256': sequence_hash.hexdigest(),
           'test_sequence_sha256': hashlib.sha256(b''.join(t.nodes.tobytes() for _, t in cache['test'])).hexdigest(),
           'train_evidence_attribute_reads_per_center': attr_reads / max(centers_total, 1),
           'train_graph_reads_per_center': graph_reads / max(centers_total, 1),
           'train_actual_contexts': contexts_total / max(centers_total, 1),
           **{'val_' + key: value for key, value in val.items()}, **{'test_' + key: value for key, value in test.items()}}
    (run_dir / 'metrics.json').write_text(json.dumps(row, indent=2), encoding='utf8')
    return row


def summarise(rows):
    result = []
    for dataset, k, mode in sorted({(r['dataset'], r['K'], r['mode']) for r in rows}):
        selected = [r for r in rows if (r['dataset'], r['K'], r['mode']) == (dataset, k, mode)]
        key = 'test_auc' if selected[0]['metric'] == 'auc' else 'test_accuracy'
        values = [r[key] for r in selected]
        result.append({'protocol': PROTOCOL, 'dataset': dataset, 'K': k, 'mode': mode,
                       'metric': selected[0]['metric'], 'seeds': len(values), 'test_mean': float(np.mean(values)),
                       'test_std': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return result


def paired(rows):
    result = []
    for dataset, k in sorted({(r['dataset'], r['K']) for r in rows}):
        conditions = {mode: {r['seed']: r for r in rows if r['dataset'] == dataset and r['K'] == k and r['mode'] == mode}
                      for mode in MODES}
        for left, right in ([(name, 'plain') for name in MODES[1:]] + [('all', name) for name in CHANNELS]):
            a, b = conditions[left], conditions[right]
            seeds = sorted(a.keys() & b.keys())
            if not seeds:
                continue
            key = 'test_auc' if a[seeds[0]]['metric'] == 'auc' else 'test_accuracy'
            delta = np.array([a[s][key] - b[s][key] for s in seeds])
            result.append({'protocol': PROTOCOL, 'dataset': dataset, 'K': k, 'comparison': left + '-' + right,
                           'primary': left == 'all' and right == 'plain', 'seeds': len(seeds),
                           'delta_mean': float(delta.mean()), 'delta_std': float(delta.std(ddof=1)) if len(seeds) > 1 else None,
                           'positive_seeds': int((delta > 0).sum()), 'per_seed_delta': json.dumps(delta.tolist())})
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--dataset')
    p.add_argument('--out', required=True)
    p.add_argument('--k', type=int, nargs='+', default=None,
                   help='Roman-empire: 2 4 8 16; others: 4 8 16 32 64')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--hops', type=int, default=2, help='uniform context pool radius; 2 limits dense-graph BFS expansion')
    p.add_argument('--calib-centers', type=int, default=256,
                   help='unlabeled training-center references for strength only')
    p.add_argument('--neighborhood-block-size', type=int, default=512)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ff-dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--channel-init', type=float, default=.1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=0, help='0: equal fixed epoch budget; validation still selects checkpoint')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--metric', choices=['auto', 'acc', 'auc'], default='auto')
    p.add_argument('--feature-normalize', choices=['l1', 'none'], default='l1')
    p.add_argument('--split-index', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--log-every', type=int, default=10)
    return p


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv)
    args.dataset = args.dataset or Path(args.data_root).name
    if args.k is None:
        args.k = [2, 4, 8, 16] if args.dataset.lower().replace('-', '_') == 'roman_empire' else list(DEFAULT_K)
    args.k = sorted(set(args.k))
    if min(args.k) < 1 or min(args.hops, args.epochs, args.calib_centers, args.neighborhood_block_size) < 1:
        parser.error('K, hops, epochs, calibration centers and neighborhood block size must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error('duplicate seeds/modes are not independent runs')
    if min(args.seeds) < 0 or args.patience < 0 or args.threads < 1 or args.ff_dim < 1 or args.log_every < 0:
        parser.error('seeds/patience/log-every must be nonnegative; threads/ff-dim must be positive')
    if args.heads < 1 or args.hidden_dim < 1 or args.hidden_dim % args.heads or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error('invalid model/batch dimensions')
    if not 0 <= args.dropout < 1 or not np.isfinite(args.channel_init) or args.channel_init < 0:
        parser.error('invalid dropout/channel initialization')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('lr must be positive and weight-decay nonnegative')
    if args.split_index < 0:
        parser.error('split-index must be nonnegative')
    if args.metric == 'auto':
        args.metric = 'auc' if args.dataset.lower().replace('-', '_') in ('minesweeper', 'tolokers', 'questions') else 'acc'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    # Never silently overwrite an old or partial experiment protocol.
    if out.exists():
        raise FileExistsError(f'Use a new result directory: {out}')
    adj, features, labels, split = load_graph(args.data_root, args.feature_normalize, args.split_index)
    if adj.shape != (len(labels), len(labels)) or features.ndim != 2 or features.shape[0] != len(labels):
        parser.error('graph/features/labels shape mismatch')
    if not np.isfinite(features).all():
        parser.error('nonfinite features')
    for name, ids in split.items():
        if not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(labels) or (labels[ids] < 0).any():
            parser.error(f'invalid or unlabeled {name} split')
    joined = np.concatenate(list(split.values()))
    if len(joined) != len(np.unique(joined)):
        parser.error('train/val/test splits overlap')
    classes = np.unique(labels[labels >= 0])
    if not len(classes) or not np.array_equal(classes, np.arange(len(classes))):
        parser.error('labels must be contiguous classes 0..C-1')
    if args.metric == 'auc' and (len(classes) != 2 or any(len(np.unique(labels[ids])) != 2 for ids in split.values())):
        parser.error('ROC-AUC requires two classes in each split')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    graph = GraphTables(adj)
    graph.build_balls(np.concatenate(list(split.values())), args.hops)
    channels = set().union(*(active_channels(mode) for mode in args.modes))
    oracle = RelationOracle(graph, features, args.neighborhood_block_size, need_neighborhood='neighborhood' in channels)
    tau = calibrate(oracle, split['train'], args)
    config = {'protocol': PROTOCOL, 'schema_version': 3, 'args': vars(args), 'dataset': args.dataset,
              'feature_normalize': args.feature_normalize, 'tau': tau, 'tau_calibration_K': max(args.k),
              'tau_uses_labels': False, 'dataset_sha256': fingerprint(args.data_root),
              'preprocess_seconds': time.perf_counter() - started, 'torch': torch.__version__,
              'preprocess_neighbor_graph_reads': oracle.preprocess_graph_reads,
              'neighbor_sum_bytes': 0 if oracle.neighbor_sums is None else oracle.neighbor_sums.nbytes,
              'channel_definitions': {'witness': 'uniform_mean_attributes_of_full_graph_common_neighbors',
                                      'strength': 'common_neighbor_count_and_degree_normalized_P2_mass',
                                      'neighborhood': 'mean_attributes_of_nonshared_neighbors_excluding_endpoints'},
              'relation_scope': 'original_graph_without_sequence_membership_filter',
              'interpretation': 'relation_information_utility_oracle_not_same_cost_SCOPE_GT',
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ('motivation.py', 'information.py', 'graph.py', 'relation.py')}}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    rows, coverage_rows = [], []
    for k in args.k:
        for seed in args.seeds:
            cache, coverage = sequence_cache(oracle, split, k, seed, args)
            coverage_rows += [{'protocol': PROTOCOL, 'dataset': args.dataset, 'K': k, 'seed': seed, 'split': name, **values}
                              for name, values in coverage.items()]
            write_csv(out / 'coverage.csv', coverage_rows)
            for mode in args.modes:
                row = run_one(mode, k, seed, oracle, features, labels, split, cache, tau, args, device)
                row.update(test_actual_contexts=coverage['test']['actual_contexts'])
                metrics_path = out / f'K{k}' / f'seed{seed}' / mode / 'metrics.json'
                metrics_path.write_text(json.dumps(row, indent=2), encoding='utf8')
                rows.append(row)
                write_csv(out / 'rows.csv', rows); write_csv(out / 'summary.csv', summarise(rows))
                write_csv(out / 'paired_results.csv', paired(rows))
                print(f"{args.dataset} K={k} seed={seed} {LABELS[mode]} test={row.get('test_auc', row['test_accuracy']):.4f}", flush=True)
            del cache
    return rows


if __name__ == '__main__':
    main()
