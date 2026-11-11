"""Center-query execution workers.

Default study: Sequential, Padded, Grouped and Dedup-Packed, complete training
 epochs with preset regularisation. Each unit is assembled/transferred lazily,
 its weighted multi-view loss is backpropagated immediately, and graphs are
 released before the next unit. Optimizer updates occur once per logical batch.
The optional fixed-input microbenchmarks have their own timing/memory scope.
CPU runs have missing GPU memory; CUDA OOM remains an explicit error status.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from SCOPE_GT import train as scope_gt
from SCOPE_GT.data import load_graph
from SCOPE_GT.graph import DirectionIndex, GraphIndex, load_raw_adjacency
from SCOPE_GT.sampler import ContextSampler, trim_padding

from .execution import LAYOUTS, make_units, unit_stats, execution_rows, make_unit_views

PROTOCOLS = ('motivation_fb', 'train_step', 'epoch')
MIB = 2. ** 20
SCHEMA = 'scope_execution_center_query_v5'


# ---------------------------------------------------------------- setup shared by the orchestrator and the workers
def load_setup(args):
    """Graph, features and sampler exactly as SCOPE_GT/train.py builds them."""
    start = time.perf_counter()
    graph = load_graph(args.data_root, args.split_suffix, args.feature_normalize)
    index = GraphIndex(graph.adj)
    direction = DirectionIndex(load_raw_adjacency(args.data_root) if args.direction != 'off' else None, index, args.direction)
    sampler = scope_gt.make_sampler(args, graph, index, direction)
    sampler.preprocess_seconds = time.perf_counter() - start
    return graph, sampler


def deterministic(args):
    """Same model with dropout disabled (equivalence checks, motivation_fb, train_step)."""
    return Namespace(**{**vars(args), 'dropout': 0., 'input_dropout': 0.})


def build_cache(args, graph, sampler, scales, n_batches, seed):
    """The first `n_batches` real training batches of epoch 0 (view 0) and the initial weights."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, scope_gt.SHUFFLE_TAG, 0]))
    pool = graph.splits['train']
    if args.centers_per_epoch and args.centers_per_epoch < len(pool):
        pool = rng.choice(pool, args.centers_per_epoch, replace=False)
    order = rng.permutation(len(pool))
    streams = scope_gt.make_streams(seed, scope_gt.TRAIN_TAGS, 0, 0)
    batches = []
    for start in range(0, min(len(pool), n_batches * args.batch_size), args.batch_size):
        centers = pool[order[start:start + args.batch_size]]
        batches.append({'centers': centers, 'labels': graph.labels[centers],
                        'draw': trim_padding(sampler.draw(centers, streams))})
    here = Path(scope_gt.__file__).resolve().parent
    source = {name: scope_gt.sha256(here / name) for name in scope_gt.SOURCE_FILES}
    for name in ('execution.py', 'benchmark_execution.py'):
        source['SCOPE_GT_ablation/' + name] = scope_gt.sha256(Path(__file__).with_name(name))
    scope_gt.seed_all(seed)
    state = scope_gt.build_model(args, graph, scales).state_dict()
    return {'schema': SCHEMA, 'source_sha256': source, 'args': vars(args), 'scales': scales, 'seed': seed, 'batches': batches,
            'state': {k: v.clone() for k, v in state.items()},
            'setup': {'preprocess_seconds': sampler.preprocess_seconds,
                      'index_cpu_bytes': sampler.index_bytes, 'candidate_limit': sampler.candidate_limit,
                      'candidate_access_budget': sampler.candidate_access_budget}}


def make_model(args, graph, cache, device, dropout_off=True):
    model = scope_gt.build_model(deterministic(args) if dropout_off else args, graph, cache['scales'])
    model.load_state_dict(cache['state'])
    return model.to(device)


def checksum(model):
    return float(sum(p.detach().double().sum() for p in model.state_dict().values() if p.is_floating_point()))


# ---------------------------------------------------------------- one logical step
def step(model, units, labels, n, backward=True):
    """Forward (and backward) of every unit; loss = sum CE / n per unit. Returns logits in batch order."""
    out = None
    for rows, batch in units:
        logits = model(batch)
        if backward:
            (F.cross_entropy(logits, labels[rows], reduction='sum') / n).backward()
        if out is None:
            out = logits.new_empty((n, logits.shape[1]))
        out[torch.as_tensor(rows, device=logits.device)] = logits.detach()
    return out


def move(units, device):
    return [(rows, batch.to(device)) for rows, batch in units]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------- equivalence
def check_equivalence(args, graph, cache, layouts, device, tol_logit=1e-4, tol_grad=1e-3):
    """Logits and every parameter gradient of each layout against dedup_packed (eval mode, dropout 0)."""
    model = make_model(args, graph, cache, device).eval()
    features = graph.features
    rows = []
    for i, item in enumerate(cache['batches']):
        n, labels = len(item['centers']), torch.as_tensor(item['labels'], device=device)
        reference = {}
        for layout in ['dedup_packed'] + [l for l in layouts if l != 'dedup_packed']:
            model.zero_grad(set_to_none=True)
            logits = step(model, move(make_units(layout, item['centers'], item['draw'], features), device), labels, n)
            grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
            if layout == 'dedup_packed':
                reference = {'logits': logits, 'grads': grads}
                continue
            scale = max(1., float(reference['logits'].abs().max()))
            logit_err = float((logits - reference['logits']).abs().max())
            grad_scale = max(float(g.abs().max()) for g in reference['grads'].values())
            same_set = set(grads) == set(reference['grads'])
            grad_err = max(float((grads[k] - reference['grads'][k]).abs().max()) for k in reference['grads']) if same_set else float('inf')
            grad_rel = grad_err / grad_scale if grad_scale > 0 else grad_err
            rows.append({'batch': i, 'layout': layout, 'max_logit_error': logit_err, 'max_grad_error': grad_err,
                         'max_grad_rel_error': grad_rel, 'passed': bool(same_set and logit_err <= tol_logit * scale
                                                                       and grad_rel <= tol_grad)})
    return rows


# ---------------------------------------------------------------- measurements (one layout per process)
def measure_fixed(protocol, layout, args, graph, cache, device, warmup, repeats):
    """motivation_fb or train_step over the cached batches; returns per-batch rows and the resident memory."""
    model = make_model(args, graph, cache, device)
    model.eval() if protocol == 'motivation_fb' else model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if protocol == 'train_step' else None
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cuda = device.type == 'cuda'
    items = []
    for item in cache['batches']:
        units = make_units(layout, item['centers'], item['draw'], graph.features)
        items.append((len(item['centers']), units, torch.as_tensor(item['labels']), unit_stats(units)))

    def one(n, units, labels):
        model.zero_grad(set_to_none=True)
        step(model, units, labels, n)
        if optimiser is not None:
            optimiser.step()

    def reset():
        if optimiser is not None:                     # same weights and zeroed AdamW state for every repetition
            model.load_state_dict(initial)
            for state in optimiser.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()

    for _ in range(warmup):
        for n, units, labels, _ in items:
            reset()
            one(n, move(units, device), labels.to(device))
    model.zero_grad(set_to_none=True)
    sync(device)
    if cuda:
        torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated(device) / MIB if cuda else None
    memory = []
    for n, units, labels, _ in items:                 # memory pass: inputs moved after the reset
        if not cuda:
            memory.append((None, None))
            continue
        reset()
        model.zero_grad(set_to_none=True)
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        one(n, move(units, device), labels.to(device))
        sync(device)
        memory.append((torch.cuda.max_memory_allocated(device) / MIB, torch.cuda.max_memory_reserved(device) / MIB))
    times = [[] for _ in items]
    for _ in range(repeats):
        for i, (n, units, labels, _) in enumerate(items):
            reset()
            dev, dev_labels = move(units, device), labels.to(device)
            sync(device)
            begin = time.perf_counter()
            one(n, dev, dev_labels)
            sync(device)
            times[i].append(1000. * (time.perf_counter() - begin))
            del dev
    rows = []
    for i, (n, _, _, stats) in enumerate(items):
        t = np.asarray(times[i])
        rows.append({'batch': i, **stats, 'time_ms_median': float(np.median(t)), 'time_ms_p10': float(np.percentile(t, 10)),
                     'time_ms_p90': float(np.percentile(t, 90)), 'peak_alloc_mib': memory[i][0],
                     'peak_reserved_mib': memory[i][1], 'resident_mib': resident,
                     'peak_increment_mib': memory[i][0] - resident if cuda else None})
    return rows


def training_loss(outputs, labels, args):
    """The original multi-view CE + consistency objective, per center."""
    loss = sum(F.cross_entropy(o, labels, label_smoothing=args.label_smoothing)
               for o in outputs) / len(outputs)
    if len(outputs) > 1:
        p0, p1 = (torch.softmax(o, -1) for o in outputs)
        loss = loss + args.consistency * (p0 - p1).pow(2).sum(-1).mean()
    return loss


def backward_logical_batch(model, layout, centers, draws, features, labels, args, device):
    """Stream units, accumulate sum(loss)/B, and release graphs after each backward.

    No optimizer step here: every layout has one update per ORIGINAL logical
    batch, including Sequential. Multi-view units use the same rows in all views.
    """
    stage = dict.fromkeys(('assemble_s', 'transfer_s', 'compute_s', 'encoded_rows', 'kv_projection_rows', 'valid_reads'), 0.)
    groups = execution_rows(layout, draws)
    for rows in groups:
        t0 = time.perf_counter()
        cpu_batches = make_unit_views(layout, centers, draws, features, rows)
        t1 = time.perf_counter()
        batches = [b.to(device) for b in cpu_batches]
        y = torch.as_tensor(labels[rows], device=device)
        t2 = time.perf_counter()
        stage['encoded_rows'] += sum(b.encoded_rows for b in batches)
        stage['valid_reads'] += sum(b.size + b.pairs for b in batches)
        stage['kv_projection_rows'] += len(model.blocks) * sum(
            b.size + b.pairs if getattr(b, 'kv_per_read', False) else b.kv_count for b in batches)
        outputs = [model(batch) for batch in batches]
        loss = training_loss(outputs, y, args) * (len(rows) / len(centers))
        loss.backward()
        t3 = time.perf_counter()
        stage['assemble_s'] += t1 - t0
        stage['transfer_s'] += t2 - t1
        stage['compute_s'] += t3 - t2
        del loss, outputs, y, batches, cpu_batches
    return stage, len(groups)


def measure_epoch(layout, args, graph, sampler, cache, device, epoch_centers,
                  warmup_epochs=0, measured_epochs=1):
    """Fresh training trajectory per layout; report every epoch and warmup flag."""
    if args.query != 'center':
        raise ValueError('execution study requires --query center')
    scope_gt.seed_all(cache['seed'])
    model = make_model(args, graph, cache, device, dropout_off=False).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    seed = cache['seed']
    views = 2 if args.consistency > 0 else 1
    rows_out = []
    for epoch in range(warmup_epochs + measured_epochs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, scope_gt.SHUFFLE_TAG, epoch]))
        pool = graph.splits['train']
        if epoch_centers and epoch_centers < len(pool):
            pool = rng.choice(pool, epoch_centers, replace=False)
        order = rng.permutation(len(pool))
        streams = [scope_gt.make_streams(seed, scope_gt.TRAIN_TAGS, epoch, view) for view in range(views)]
        accesses = contexts = path_records = background_records = groups = steps = 0
        stage = dict.fromkeys(('sample_s', 'assemble_s', 'transfer_s', 'compute_s', 'encoded_rows', 'kv_projection_rows', 'valid_reads'), 0.)
        optimiser.zero_grad(set_to_none=True)
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        for start in range(0, len(pool), args.batch_size):
            centers = pool[order[start:start + args.batch_size]]
            t0 = time.perf_counter()
            draws = [trim_padding(sampler.draw(centers, stream)) for stream in streams]
            for draw in draws:
                accesses += sum(draw[key] for key in scope_gt.ACCESS_KEYS)
                contexts += int(draw['ctx_valid'].sum())
                path_records += draw['witness_records']
                background_records += draw['background_records']
            stage['sample_s'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            optimiser.zero_grad(set_to_none=True)
            stage['compute_s'] += time.perf_counter() - t0
            part, unit_count = backward_logical_batch(model, layout, centers, draws,
                graph.features, graph.labels[centers], args, device)
            for key, value in part.items():
                stage[key] += value
            groups += unit_count
            t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()
            stage['compute_s'] += time.perf_counter() - t0
            steps += 1
            del draws
        schedule.step()
        sync(device)
        total = time.perf_counter() - begin
        cuda = device.type == 'cuda'
        rows_out.append({'batch': 'epoch', 'epoch': epoch + 1,
            'warmup_epoch': epoch < warmup_epochs, 'centers': len(pool), 'training_views': views,
            'logical_batch_size': args.batch_size, 'optimizer_steps': steps, 'execution_units': groups,
            'query': 'center', 'epoch_seconds': total, **stage,
            'stage_clock': 'host dispatch wall time; GPU synchronized only at epoch boundaries',
            'train_peak_alloc_mib': torch.cuda.max_memory_allocated(device) / MIB if cuda else None,
            'train_peak_reserved_mib': torch.cuda.max_memory_reserved(device) / MIB if cuda else None,
            'index_cpu_bytes': sampler.index_bytes, 'preprocess_seconds': sampler.preprocess_seconds,
            'graph_access_per_center': accesses / len(pool), 'contexts_per_center': contexts / len(pool),
            'path_records_per_center': path_records / len(pool),
            'background_records_per_center': background_records / len(pool)})
    return rows_out


def run_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    torch.set_num_threads(spec['threads'])
    with open(spec['cache'], 'rb') as handle:
        cache = pickle.load(handle)
    if cache.get('schema') != SCHEMA:
        raise ValueError('old execution cache schema; regenerate inputs with the current runner')
    args = Namespace(**cache['args'])
    device = torch.device(spec['device'])
    result = {'schema': SCHEMA, 'query': 'center', 'protocol': spec['protocol'],
              'layout': spec['layout'], 'status': 'ok', 'setup': cache['setup']}
    try:
        if spec['protocol'] == 'epoch':
            graph, sampler = load_setup(args)
            result['rows'] = measure_epoch(spec['layout'], args, graph, sampler, cache, device, spec['epoch_centers'],
                spec.get('epoch_warmup', 3), spec.get('epoch_repeats', 10))
        else:
            graph = load_graph(args.data_root, args.split_suffix, args.feature_normalize)
            model = make_model(args, graph, cache, torch.device('cpu'))
            if abs(checksum(model) - spec['checksum']) > 1e-6 * max(1., abs(spec['checksum'])):
                raise SystemExit('worker weights differ from the cached initial weights')
            del model
            result['rows'] = measure_fixed(spec['protocol'], spec['layout'], args, graph, cache, device,
                                           spec['warmup'], spec['repeats'])
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='oom', error=str(error).splitlines()[0], rows=[])
    except Exception as error:                          # keep the failure as evidence, never as a fake result
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), rows=[])
    Path(spec['result']).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='isolated benchmark worker')
    parser.add_argument('--worker', required=True)
    run_worker(parser.parse_args().worker)
