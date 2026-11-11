"""Train one query role at one depth using final SCOPE-GT presets and dataflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from SCOPE_GT import train as core
from SCOPE_GT.cache import EvaluationDrawCache
from SCOPE_GT.data import load_graph, resolve_feature_normalize
from SCOPE_GT.graph import DirectionIndex, GraphIndex, load_raw_adjacency
from SCOPE_GT.sampler import trim_padding

from . import PROTOCOL_ID
from .model import DepthSCOPEGT


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(serializable(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def source_hashes():
    """Only executed training/model dependencies; plotting changes do not affect a run."""
    here, base = Path(__file__).resolve().parent, Path(core.__file__).resolve().parent
    files = {f'SCOPE_GT/{name}': base / name for name in core.SOURCE_FILES}
    files.update({f'{here.name}/{name}': here / name for name in ('__init__.py', 'model.py', 'train.py')})
    return {name: core.sha256(path) for name, path in files.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--role', choices=['center', 'partial', 'full'], default='center')
    p.add_argument('--partial-ratio', type=float, default=.5)
    p.add_argument('--diagnostic-centers', type=int, default=4096,
                   help='fixed test centers for representation SVD; 0 disables diagnostics')
    p.add_argument('--profile-batches', type=int, default=3)
    p.add_argument('--profile-warmup', type=int, default=5)
    p.add_argument('--profile-repeats', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    extra, rest = p.parse_known_args(argv)
    # Native parser validates all production knobs. Query roles are controlled
    # here; use its center setting to allow the common single-layer anchor.
    if any(x == '--query' or x.startswith('--query=') or x == '--execution' or x.startswith('--execution=') for x in rest):
        p.error('query and execution are set by --role; all roles use dense')
    if '-h' in rest or '--help' in rest:
        core.build_parser().print_help()
        print('\nDepth-experiment options:')
        p.print_help()
        raise SystemExit(0)
    args = core.parse_args(rest + ['--query', 'center', '--execution', 'dense'])
    vars(args).update(vars(extra))
    if args.out_dir is None:
        p.error('--out-dir is required (normally supplied by run.py)')
    if args.layers < 1 or not 0 <= args.partial_ratio <= 1:
        p.error('layers must be positive; partial-ratio must be in [0,1]')
    if min(args.diagnostic_centers, args.profile_batches, args.profile_warmup) < 0 or args.profile_repeats < 1:
        p.error('diagnostic/profile counts must be nonnegative, and repeats must be positive')
    if args.relations != 'on' or args.relation_source != 'graph' or args.sampling != 'structured' or args.relation_drop:
        p.error('the query-depth study retains final structured sampling and complete graph T/C/B')
    args.feature_normalize = resolve_feature_normalize(args.dataset, args.feature_normalize)
    if args.metric == 'auto':
        args.metric = 'auc' if args.dataset.lower().replace('-', '_') in core.AUC_DATASETS else 'acc'
    args.metric = 'accuracy' if args.metric == 'acc' else args.metric
    return args


def build_model(args, graph, scales, seed):
    model = DepthSCOPEGT(graph.features.shape[1], int(graph.labels.max() + 1),
        dim=args.hidden_dim, heads=args.heads, layers=args.layers, ff_dim=args.ff_dim,
        dropout=args.dropout, input_dropout=args.input_dropout, enc_layers=args.enc_layers,
        rel_dim=args.rel_dim, hops=args.max_hops, relation_scale_init=args.relation_scale_init,
        background_scale_init=args.background_scale_init, topology_similarity=args.topology_similarity == 'on',
        connection_attributes=args.connection_attributes == 'on', connection_position=args.connection_position == 'on',
        learn_scales=args.relation_scales == 'learned', relations=True, role=args.role,
        partial_ratio=args.partial_ratio, partial_seed=seed)
    model.relation.set_scales(scales)
    return model


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def predict(model, sampler, centers, args, device, seed, features, views, cache=None):
    model.eval()
    total = None
    for view in range(views):
        model.set_query_context(view=view, training=False)
        streams = core.make_streams(seed, core.EVAL_TAGS, view)
        chunks = []
        for start in range(0, len(centers), args.eval_batch_size):
            batch = core.draw_batch(sampler, centers[start:start + args.eval_batch_size], streams,
                                    features, device, 'dense', cache=cache)
            chunks.append(model(batch).float().cpu())
        logits = torch.cat(chunks)
        total = logits if total is None else total + logits
    return (total / views).numpy()


def fixed_subset(centers, limit):
    centers = np.asarray(centers, dtype=np.int64)
    if len(centers) <= limit:
        return np.sort(centers)
    return np.sort(np.random.default_rng(20261029).choice(centers, limit, replace=False))


@torch.no_grad()
def representation_diagnostics(model, sampler, graph, features, args, seed, device, out):
    """Mean effective rank over views, using identical held-out center IDs."""
    if not args.diagnostic_centers:
        return {'center_effective_rank': None}
    centers = fixed_subset(graph.splits['test'], args.diagnostic_centers)
    model.eval()
    model.capture_embeddings(True)
    ranks, singulars = [], []
    digest = hashlib.sha256(centers.tobytes())
    try:
        for view in range(args.eval_contexts):
            model.set_query_context(view=view, training=False)
            streams = core.make_streams(seed, core.EVAL_TAGS, view)
            rows = []
            for start in range(0, len(centers), args.eval_batch_size):
                ids = centers[start:start + args.eval_batch_size]
                draw = trim_padding(sampler.draw(ids, streams))
                for name in sorted(draw):
                    value = draw[name]
                    if isinstance(value, np.ndarray):
                        digest.update(name.encode('utf-8'))
                        digest.update(str((value.shape, value.dtype.str)).encode('ascii'))
                        digest.update(np.ascontiguousarray(value).tobytes())
                batch = core.DenseBatch(ids, draw, features, device)
                model(batch)
                rows.append(model.last_center_embeddings.float().cpu())
            matrix = torch.cat(rows).double()
            values = torch.linalg.svdvals(matrix)
            prob = values / values.sum().clamp_min(1e-30)
            rank = float(torch.exp(-(prob * prob.clamp_min(1e-30).log()).sum())) if values.sum() > 0 else 0.
            ranks.append(rank)
            singulars.append(values.numpy())
    finally:
        model.capture_embeddings(False)
    np.savez_compressed(out / 'representation_diagnostics.npz', centers=centers,
                        singular_values=np.stack(singulars), effective_rank=np.asarray(ranks))
    return {'center_effective_rank': float(np.mean(ranks)), 'diagnostic_center_count': len(centers),
            'diagnostic_views': args.eval_contexts, 'evaluation_evidence_sha256': digest.hexdigest(),
            'representation_scope': 'uncentered post-final-LN center states; singular-value entropy; rank averaged over views'}


def profile_reader(model, sampler, graph, features, args, seed, device, out):
    """Reader only: cached encoded inputs, actual Q gathering, no optimizer step.

    Forward-only uses eval mode; forward/backward uses training dropout.
    Encoder, relation encoder, sampling, H2D and Partial selection are excluded.
    End-to-end epoch timing separately includes all of these recurring costs.
    """
    if not args.profile_batches:
        return {'reader_forward_seconds': None, 'reader_forward_backward_seconds': None}
    centers = fixed_subset(graph.splits['train'], args.batch_size * args.profile_batches)
    rows, peaks, increments, selection_times = [], [], [], []
    for number, start in enumerate(range(0, len(centers), args.batch_size)):
        ids = centers[start:start + args.batch_size]
        batch = core.draw_batch(sampler, ids, core.make_streams(seed, core.EVAL_TAGS, 0), features, device, 'dense')
        model.set_query_context(epoch=0, view=0, training=True)
        core.sync(device)
        started = time.perf_counter()
        model.prepare_query_selection(batch)
        core.sync(device)
        selection_times.append(time.perf_counter() - started)
        model.eval()
        with torch.no_grad():
            encoded = model.encoder(batch.features)
            pairs, selves = model._relation_rows(encoded, batch)
        encoded = encoded.detach().requires_grad_(True)
        pairs = tuple(x.detach().requires_grad_(True) for x in pairs)
        selves = tuple(x.detach().requires_grad_(True) for x in selves)
        leaves = (encoded,) + pairs + selves
        labels = torch.as_tensor(graph.labels[ids], device=device)
        for backward in (False, True):
            model.train(backward)
            measured = []
            for repeat in range(args.profile_warmup + args.profile_repeats):
                model.zero_grad(set_to_none=True)
                for tensor in leaves:
                    tensor.grad = None
                core.sync(device)
                baseline = None
                if device.type == 'cuda' and repeat >= args.profile_warmup:
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = torch.cuda.memory_allocated(device)
                started = time.perf_counter()
                with torch.set_grad_enabled(backward):
                    logits = model.read_encoded(batch, encoded, pairs, selves)
                    if backward:
                        F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing).backward()
                core.sync(device)
                elapsed = time.perf_counter() - started
                if repeat >= args.profile_warmup:
                    measured.append(elapsed)
                    if baseline is not None and backward:
                        peak = int(torch.cuda.max_memory_allocated(device))
                        peaks.append(peak)
                        increments.append(peak - baseline)
                del logits
            rows.append({'batch': number, 'centers': len(ids), 'mode': 'forward_backward' if backward else 'forward',
                         'median_seconds': float(np.median(measured)), 'repeats': args.profile_repeats,
                         'valid_contexts': batch.pairs, 'selected_context_queries': int(batch._depth_queries[1].numel())})
        del batch, encoded, pairs, selves, leaves
    model.zero_grad(set_to_none=True)
    model.eval()
    core.write_csv(out / 'reader_profile.csv', rows)
    return {'reader_forward_seconds': float(np.mean([r['median_seconds'] for r in rows if r['mode'] == 'forward'])),
            'reader_forward_backward_seconds': float(np.mean([r['median_seconds'] for r in rows if r['mode'] == 'forward_backward'])),
            'reader_peak_allocated_bytes': max(peaks) if peaks else None,
            'reader_incremental_peak_bytes': max(increments) if increments else None,
            'partial_selection_seconds_per_batch': float(np.mean(selection_times)),
            'reader_profile_scope': 'cached encodings and relation vectors; excludes sampling, encoders, H2D, query selection, optimizer; eval forward / train forward+backward',
            'profile_batches_measured': len(rows) // 2, 'profile_repeats': args.profile_repeats}


def train_seed(args, graph, sampler, features, seed, device, out, scales, preprocess_seconds):
    core.seed_all(seed)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = build_model(args, graph, scales, seed).to(device)
    initial_hash = state_hash(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    train_ids, val_ids, test_ids = (graph.splits[x] for x in ('train', 'val', 'test'))
    epoch_val = val_ids
    if args.val_max_centers and args.val_max_centers < len(val_ids):
        pick = np.random.default_rng(np.random.SeedSequence([seed, 31337])).choice(len(val_ids), args.val_max_centers, replace=False)
        epoch_val = val_ids[np.sort(pick)]
    cache = EvaluationDrawCache(int(args.eval_cache_mib * 2**20)) if args.eval_cache == 'on' else None
    history, best, bad, peak = [], {'score': -float('inf'), 'epoch': 0, 'state': None}, 0, None
    views = 2 if args.consistency > 0 else 1
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed, core.SHUFFLE_TAG, epoch]))
        pool = train_ids
        if args.centers_per_epoch and args.centers_per_epoch < len(pool):
            pool = rng.choice(pool, args.centers_per_epoch, replace=False)
        order = rng.permutation(len(pool))
        streams = [core.make_streams(seed, core.TRAIN_TAGS, epoch, view) for view in range(views)]
        tally = core.Tally(args.max_hops, sampler.graph.degree)
        loss_sum = torch.zeros((), device=device)
        core.sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for start in range(0, len(pool), args.batch_size):
            centers = pool[order[start:start + args.batch_size]]
            batches = [core.draw_batch(sampler, centers, stream, features, device, 'dense', tally) for stream in streams]
            labels = torch.as_tensor(graph.labels[centers], device=device)
            outputs = []
            for view, batch in enumerate(batches):
                model.set_query_context(epoch=epoch, view=view, training=True)
                outputs.append(model(batch))
            loss = sum(F.cross_entropy(x, labels, label_smoothing=args.label_smoothing) for x in outputs) / views
            if views > 1:
                p0, p1 = (torch.softmax(x, -1) for x in outputs)
                loss = loss + args.consistency * (p0 - p1).pow(2).sum(-1).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()
            loss_sum += loss.detach() * len(centers)
            del outputs, loss, batches, batch
        core.sync(device)
        seconds = time.perf_counter() - started
        if device.type == 'cuda':
            peak = max(peak or 0, int(torch.cuda.max_memory_allocated(device)))
        schedule.step()
        train_loss = float(loss_sum) / len(pool)
        if not np.isfinite(train_loss):
            raise RuntimeError(f'nonfinite train loss at epoch {epoch + 1}')
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'epoch_seconds': seconds, **tally.summary()}
        if epoch == args.epochs - 1 or (epoch + 1) % args.eval_every == 0:
            core.sync(device)
            started = time.perf_counter()
            val = core.scores(predict(model, sampler, epoch_val, args, device, seed, features, args.train_eval_contexts, cache),
                              graph.labels[epoch_val], args.metric)
            row.update(val_seconds=time.perf_counter() - started, val_loss=val['loss'], val_accuracy=val['accuracy'])
            row['val_' + args.metric] = val[args.metric]
            score = core.selection_score(val, args.metric)
            if score > best['score'] + 1e-6 or best['state'] is None:
                best = {'score': score, 'epoch': epoch + 1,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
        history.append(row)
        core.write_csv(out / 'history.csv', history)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            shown = row.get('val_' + args.metric)
            print(f'[{args.dataset} L={args.layers} {args.role} seed={seed}] epoch={epoch+1}/{args.epochs} '
                  f'loss={train_loss:.4f} val={100*shown if shown is not None else float("nan"):.2f} epoch={seconds:.3f}s', flush=True)
        if args.patience and bad >= args.patience:
            break
    model.load_state_dict(best['state'])
    optimiser.zero_grad(set_to_none=True)
    del optimiser, schedule
    core.sync(device)
    val = core.scores(predict(model, sampler, val_ids, args, device, seed, features, args.eval_contexts, cache),
                      graph.labels[val_ids], args.metric)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    core.sync(device)
    started = time.perf_counter()
    logits = predict(model, sampler, test_ids, args, device, seed, features, args.eval_contexts, cache)
    core.sync(device)
    inference_seconds = time.perf_counter() - started
    inference_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    test = core.scores(logits, graph.labels[test_ids], args.metric)
    torch.save({'state_dict': best['state'], 'args': serializable(vars(args)), 'seed': seed,
                'best_epoch': best['epoch'], 'statistic_scales': scales, 'protocol_id': PROTOCOL_ID,
                'dataset_sha256': graph.fingerprint}, out / 'checkpoint.pt')
    np.savez_compressed(out / 'test_predictions.npz', centers=test_ids, labels=graph.labels[test_ids], logits=logits)
    epoch_times = [r['epoch_seconds'] for r in history]
    result = {'status': 'completed', 'protocol_id': PROTOCOL_ID, 'dataset': args.dataset, 'role': args.role,
              'layers': args.layers, 'seed': seed, 'metric': args.metric, 'best_epoch': best['epoch'],
              'epochs_run': len(history), 'initial_state_sha256': initial_hash,
              'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'preprocess_seconds': preprocess_seconds, 'epoch_seconds': float(np.mean(epoch_times[1:] or epoch_times)),
              'epoch_seconds_all': float(np.mean(epoch_times)), 'train_seconds': float(sum(epoch_times)),
              'validation_seconds': float(sum(r.get('val_seconds', 0.) for r in history)),
              'time_to_best_seconds': float(sum(r['epoch_seconds'] + r.get('val_seconds', 0.) for r in history[:best['epoch']])),
              'inference_seconds': inference_seconds, 'train_peak_allocated_bytes': peak,
              'inference_peak_allocated_bytes': inference_peak,
              'train_sample_seconds': float(sum(r['sample_seconds'] for r in history)),
              'train_assemble_seconds': float(sum(r['assemble_seconds'] for r in history)),
              'training_peak_scope': 'training phases only; resident features, model, optimizer, batches; excludes validation and post-training diagnostics',
              'execution': 'dense', 'single_layer_shared_anchor': args.layers == 1}
    result.update({'val_' + key: value for key, value in val.items()})
    result.update({'test_' + key: value for key, value in test.items()})
    result.update(representation_diagnostics(model, sampler, graph, features, args, seed, device, out))
    result.update(profile_reader(model, sampler, graph, features, args, seed, device, out))
    write_json(out / 'metrics.json', result)
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Fail before graph preparation when an unrelated run already occupies this path.
    config_path = out / 'config.json'
    options = serializable(vars(args))
    contract = {k: v for k, v in options.items() if k not in ('resume', 'out_dir', 'seeds', 'log_every')}
    hashes = source_hashes()
    old = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    if old is not None:
        if not args.resume:
            raise SystemExit(f'Results already exist: {out}. Use --resume or a new --out-dir.')
        if old['contract'] != contract or old['source_sha256'] != hashes:
            raise SystemExit(f'Configuration/training source differs from this run: {out}. Use a fresh --out-dir; plotting edits do not affect this check.')
    started = time.perf_counter()
    graph = load_graph(args.data_root, args.split_suffix, args.feature_normalize)
    if old is not None and old['dataset_sha256'] != graph.fingerprint:
        raise SystemExit('Dataset files differ from the saved run; use a fresh output directory.')
    index = GraphIndex(graph.adj)
    raw = load_raw_adjacency(args.data_root) if args.direction != 'off' else None
    direction = DirectionIndex(raw, index, args.direction)
    sampler = core.make_sampler(args, graph, index, direction)
    scales = core.fit_scales(args, sampler, graph)
    features = graph.features
    if device.type == 'cuda' and features.nbytes <= 2 * 1024**3:
        features = torch.as_tensor(features, device=device)
    core.sync(device)
    preprocess_seconds = time.perf_counter() - started
    config = {'protocol_id': PROTOCOL_ID, 'args': options, 'contract': contract,
              'dataset': args.dataset, 'dataset_sha256': graph.fingerprint, 'source_sha256': hashes,
              'statistic_scales': scales, 'torch': torch.__version__, 'device': str(device),
              'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'num_nodes': len(graph.labels), 'sampling_fingerprint': sampler.fingerprint,
              'preprocess_seconds': preprocess_seconds, 'single_layer_shared_anchor': args.layers == 1}
    write_json(config_path, config)
    results = []
    for seed in args.seeds:
        dest = out / f'seed{seed}'
        dest.mkdir(exist_ok=True)
        metrics = dest / 'metrics.json'
        previous = json.loads(metrics.read_text(encoding='utf-8')) if metrics.exists() else None
        if args.resume and previous and previous.get('status') == 'completed':
            print(f'[skip] {args.dataset} L={args.layers} {args.role} seed={seed}', flush=True)
            result = previous
        else:
            try:
                result = train_seed(args, graph, sampler, features, seed, device, dest, scales, preprocess_seconds)
            except Exception as error:
                write_json(metrics, {'status': 'failed', 'dataset': args.dataset, 'role': args.role,
                                     'layers': args.layers, 'seed': seed, 'error': str(error), 'traceback': traceback.format_exc()})
                raise
        results.append(result)
        core.write_csv(out / 'training_summary.csv', results + core.aggregate_rows(results))
        print(f'[done] {args.dataset} L={args.layers} {args.role} seed={seed}: '
              f'{args.metric}={100*result["test_"+args.metric]:.2f}', flush=True)


if __name__ == '__main__':
    main()
