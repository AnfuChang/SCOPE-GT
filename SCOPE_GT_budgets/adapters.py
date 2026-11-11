"""Native models and training operations behind an explicit common measurement protocol.

Original training entrypoints are not patched or monkey-patched. These adapters use
their public model/sampler/optimizer operations and preserve method-specific losses.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .common import CODE, array_bytes
from SCOPE_GT.external_backbones import baseline_path


def native(method):
    if method == 'scope_gt':
        return importlib.import_module('SCOPE_GT.train')
    if method == 'sgformer':
        folder = str(baseline_path('SGFormer/sgformer_node').parent)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        return importlib.import_module('sgformer_node.train_sgformer')
    name, filename = {'nlgt': ('NLGT', 'nlgt_npz.py'),
                      'nagphormer': ('NAGphormer', 'baseline_suite.py')}[method]
    module_name = '_scope_experiment_' + method
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, baseline_path(Path(name) / filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def config_for(method, dataset, overrides=None):
    overrides = overrides or {}
    if method == 'scope_gt':
        c = native('scope_gt')
        parser = c.build_parser()
        parser.set_defaults(**c.load_preset(dataset))
        cfg = vars(parser.parse_args(['--data-root', '.', '--preset', dataset]))
        for key in ('out_dir', 'seeds', 'device', 'threads', 'data_root', 'dataset', 'preset'):
            cfg.pop(key, None)
        cfg['metric'] = 'auc' if dataset in c.AUC_DATASETS else 'accuracy'
        cfg['feature_normalize'] = c.resolve_feature_normalize(dataset, cfg['feature_normalize'])
    else:
        profiles = json.loads(Path(__file__).with_name('profiles.json').read_text(encoding='utf-8'))[method]
        if dataset not in profiles['datasets']:
            raise ValueError(f'No reviewed {method} profile for {dataset}; add a documented profile first')
        cfg = {**profiles['defaults'], **profiles['datasets'][dataset]}
        cfg['metric'] = 'auc' if dataset in ('tolokers', 'minesweeper', 'questions') else 'accuracy'
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f'Unknown {method} override keys: {sorted(unknown)}')
    cfg.update(overrides)
    if cfg['epochs'] < 1 or cfg['eval_every'] < 1 or cfg['patience'] < 0:
        raise ValueError('epochs/eval_every must be positive; patience must be nonnegative')
    if cfg['metric'] not in ('accuracy', 'auc'):
        raise ValueError('metric must be accuracy or auc')
    if method == 'nlgt' and (not cfg['strict_no_neighbor_labels'] or cfg['num_workers'] != 0):
        raise ValueError('This protocol requires NLGT label inputs off and num_workers=0 (RSS scope)')
    if method == 'nagphormer' and cfg['nr_aug'] != 'off':
        raise ValueError('This comparison uses NAGphormer, without NrAug')
    if method == 'scope_gt':
        # Run native validation, including hop shares, query/layout and finite budgets.
        checked = native('scope_gt').parse_args  # validations also read a temporary parser's defaults
        argv = ['--data-root', '.', '--preset', dataset]
        for key, value in cfg.items():
            if value is None or isinstance(value, bool):
                continue
            value = 'acc' if key == 'metric' and value == 'accuracy' else value
            argv += ['--' + key.replace('_', '-')]
            argv += [str(v) for v in value] if isinstance(value, list) else [str(value)]
        validated = checked(argv)
        # Keep one canonical selector: config_for may validate an effective
        # configuration twice, and argparse treats index/suffix as exclusive.
        cfg['split_suffix'] = validated.split_suffix
        cfg['split_index'] = None
    return cfg


class Adapter:
    def __init__(self, method, cfg, root, seed, device, artifact_dir):
        self.method, self.cfg, self.root, self.seed, self.device = method, cfg, Path(root), seed, device
        self.a = SimpleNamespace(**cfg, data_root=str(root), device=str(device))
        self.artifact_dir = Path(artifact_dir)
        self.n = native(method)
        self.state_extra = {}
        self.feature_location = 'cpu'

    def load(self):
        if self.method == 'scope_gt':
            self.graph = self.n.load_graph(self.root, self.a.split_suffix, self.a.feature_normalize)
            self.labels, self.splits = self.graph.labels, self.graph.splits
        elif self.method == 'nlgt':
            self.adj, self.features, self.labels, _ = self.n.load_graph_npz(self.root)
        elif self.method == 'nagphormer':
            # Native row normalization remains an explicit method-specific input transform.
            self.a.resplit_622 = False
            self.exp = self.n.Experiment(self.a)
            self.data = self.exp.load_data(self.seed)
            self.labels = self.data['labels']
        else:
            self.graph = self.n.load_bilgt_graph(str(self.root), to_undirected=bool(self.a.to_undirected),
                                                feature_norm=self.a.feature_norm)
            self.labels = self.graph.labels
        # Use exactly the same canonical splits, with bool masks handled consistently.
        splits = {}
        for name in ('train', 'val', 'test'):
            ids = np.load(self.root / f'{name}.npy', allow_pickle=False)
            ids = np.flatnonzero(ids) if ids.dtype == bool else ids
            if not np.issubdtype(ids.dtype, np.integer):
                raise ValueError('Non-integer split IDs')
            ids = ids.astype(np.int64).reshape(-1)
            if not len(ids) or ids.min() < 0 or ids.max() >= len(self.labels) or np.any(self.labels[ids] < 0):
                raise ValueError('Invalid/empty/unlabeled split')
            splits[name] = ids
        joined = np.concatenate(list(splits.values()))
        if len(np.unique(joined)) != len(joined):
            raise ValueError('Duplicate/overlapping split IDs')
        classes = np.unique(self.labels[self.labels >= 0])
        if self.labels.ndim != 1 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError('Single-label contiguous classes required by this protocol')
        self.splits, self.classes = splits, len(classes)
        if self.method == 'nagphormer':
            for name, ids in splits.items():
                self.data[name + '_idx'] = ids
        elif self.method == 'sgformer':
            for name, ids in splits.items():
                setattr(self.graph, name + '_idx', ids)

    def prepare(self, checkpoint=None):
        a, n, device = self.a, self.n, self.device
        # Native seeds are set before data preparation; SCOPE-GT resets after calibration,
        # matching train_one_seed's independent backbone initialization.
        if self.method == 'scope_gt':
            self.index = n.GraphIndex(self.graph.adj)
            self.direction = n.DirectionIndex(n.load_raw_adjacency(self.root) if a.direction != 'off' else None,
                                                self.index, a.direction)
            self.sampler = n.make_sampler(a, self.graph, self.index, self.direction)
            scales = checkpoint['extra']['scales'] if checkpoint else (
                n.fit_scales(a, self.sampler, self.graph) if a.relations == 'on' else None)
            self.state_extra['scales'] = scales
            self.features = self.graph.features
            if device.type == 'cuda' and self.features.nbytes <= 2 * 1024 ** 3:
                self.features = torch.as_tensor(self.features, device=device)
                self.feature_location = str(device)
            n.seed_all(self.seed)
            self.model = n.build_model(a, self.graph, scales).to(device)
            self.eval_cache = n.EvaluationDrawCache(int(a.eval_cache_mib * 2 ** 20)) if a.eval_cache == 'on' else None
        elif self.method == 'nlgt':
            self.sampler = n.GraphSampler(self.adj, a.fanouts)
            self.model = n.NLGTModel(in_dim=self.features.shape[1], num_classes=self.classes, hid_dim=a.hid_dim,
                num_layers=a.num_layers, num_heads=a.num_heads, dim_feedforward=a.dim_feedforward,
                dropout=a.dropout).to(device)
            self.train_mask = np.zeros(len(self.labels), dtype=bool)
            self.train_mask[self.splits['train']] = True
        elif self.method == 'nagphormer':
            token_path = self.artifact_dir / 'tokens.pt'
            if checkpoint:
                # Persist exact LapPE+Hop2Token: no eigenvector resampling at deployment.
                token = torch.load(token_path, map_location='cpu', weights_only=False)
                if a.nag_cache_on_device:
                    token = token.to(device)
                self.cache = {'token_seq': token}
                self.model = n.NAGphormer(in_dim=token.shape[-1], hid_dim=a.hid_dim, out_dim=self.classes,
                    num_layers=a.num_layers, num_heads=a.num_heads, dim_feedforward=a.dim_feedforward,
                    dropout=a.dropout, seq_len=token.shape[1]).to(device)
            else:
                self.model, self.cache, _ = self.exp.build_model_and_cache(self.data)
                torch.save(self.cache['token_seq'].detach().cpu(), token_path)
            self.feature_location = str(self.cache['token_seq'].device) + ' (precomputed tokens)'
            if checkpoint is None:
                self.labels_t = torch.as_tensor(self.labels, dtype=torch.long, device=device)
        else:
            if a.arch == 'official_large':
                self.model = n.SGFormerOfficialLarge(self.graph.num_features, a.hidden_dim, self.classes,
                    graph_weight=a.alpha, trans_layers=a.trans_layers, trans_heads=a.trans_heads,
                    trans_dropout=a.dropout, trans_use_bn=bool(a.trans_use_bn),
                    trans_use_residual=bool(a.trans_use_residual), trans_use_weight=bool(a.trans_use_weight),
                    trans_use_act=bool(a.trans_use_act), gcn_layers=a.gcn_layers, gnn_dropout=a.gnn_dropout,
                    gnn_use_bn=bool(a.gnn_use_bn), gnn_use_residual=bool(a.gnn_use_residual),
                    gnn_use_weight=bool(a.gnn_use_weight), gnn_use_init=bool(a.gnn_use_init),
                    gnn_use_act=bool(a.gnn_use_act)).to(device)
            else:
                self.model = n.SGFormerNode(self.graph.num_features, a.hidden_dim, self.classes,
                    alpha=a.alpha, beta=a.beta, gcn_layers=a.gcn_layers, trans_layers=a.trans_layers,
                    dropout=a.dropout, gnn_dropout=a.gnn_dropout).to(device)
            self.cpu_cache = {}
            self.x_full = self.adj_t = None
            # Inference doesn't retain unused training-side full graph GPU tensors.
            full_gpu = a.eval_mode == 'full' or (checkpoint is None and a.train_mode == 'full')
            if full_gpu:
                self.x_full = torch.from_numpy(self.graph.features).to(device)
                self.adj_t = n.scipy_to_torch_sparse(self.graph.adj_norm, device)
                self.feature_location = str(device)
            if a.eval_mode == 'full_cpu':
                self.cpu_cache = {'x': torch.from_numpy(self.graph.features),
                                  'adj': n.scipy_to_torch_sparse(self.graph.adj_norm, torch.device('cpu'))}
            self.train_mask = np.zeros(len(self.labels), dtype=bool)
            self.train_mask[self.splits['train']] = True
            if checkpoint is None:
                self.labels_t = n.tensor_labels(self.labels, device, self.graph.problem_type)
        if checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def optimizer(self):
        a, model = self.a, self.model
        self.schedule = None
        if self.method == 'scope_gt':
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
            self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        elif self.method == 'nlgt':
            opt = self.n.build_optimizer(a, model)
            self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                                      eta_min=max(a.lr * a.lrf, 1e-8))
        elif self.method == 'nagphormer':
            cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[a.optimizer]
            opt = cls(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        elif a.arch == 'official_large':
            trans, gnn = model.param_groups()
            opt = torch.optim.Adam([{'params': trans, 'weight_decay': a.weight_decay},
                {'params': gnn, 'weight_decay': a.gnn_weight_decay}], lr=a.lr)
        else:
            gnn = list(model.gcn.parameters())
            ids = {id(p) for p in gnn}
            opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if id(p) not in ids],
                'weight_decay': a.weight_decay}, {'params': gnn, 'weight_decay': a.gnn_weight_decay}], lr=a.lr)
        return opt

    def epoch(self, epoch, opt):
        a, n, device, model = self.a, self.n, self.device, self.model
        model.train()
        if self.method == 'scope_gt':
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, n.SHUFFLE_TAG, epoch - 1]))
            pool = self.splits['train']
            if a.centers_per_epoch and a.centers_per_epoch < len(pool):
                pool = rng.choice(pool, a.centers_per_epoch, replace=False)
            order = rng.permutation(len(pool))
            views = 2 if a.consistency > 0 else 1
            streams = [n.make_streams(self.seed, n.TRAIN_TAGS, epoch - 1, v) for v in range(views)]
            class Tally(n.Tally):
                def add(self, draw, batch, *args, **kwargs):
                    self.before = getattr(self, 'before', 0) + draw.get('candidates_before_cap', 0)
                    super().add(draw, batch, *args, **kwargs)
            tally = Tally(a.max_hops, self.sampler.graph.degree)
            loss_sum = torch.zeros((), device=device)
            for start in range(0, len(pool), a.batch_size):
                centers = pool[order[start:start + a.batch_size]]
                batches = [n.draw_batch(self.sampler, centers, stream, self.features, device, a.execution, tally)
                           for stream in streams]
                outputs = [model(b) for b in batches]
                y = torch.as_tensor(self.labels[centers], device=device)
                loss = sum(F.cross_entropy(o, y, label_smoothing=a.label_smoothing) for o in outputs) / views
                if views > 1:
                    p0, p1 = (torch.softmax(o, -1) for o in outputs)
                    loss += a.consistency * (p0 - p1).pow(2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                loss_sum += loss.detach() * len(centers)
            extra = tally.summary()
            extra['candidates_before_cap_per_center'] = getattr(tally, 'before', 0) / max(tally.total['centers'], 1)
            extra.update(per_center_unit='center_view', training_views=views)
            loss_value, seen, steps = float(loss_sum) / len(pool), len(pool), math.ceil(len(pool) / a.batch_size)
        elif self.method == 'nlgt':
            ds = n.SampledNodeDataset(centers=self.splits['train'], feat=self.features, labels=self.labels,
                sampler=self.sampler, max_nodes=a.max_nodes, num_classes=self.classes,
                seed=self.seed + epoch * 7919, strict_no_neighbor_labels=True, train_mask=self.train_mask)
            loader = n.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=device.type == 'cuda')
            loss_sum, seen, steps, skipped = 0., 0, 0, 0
            for x, labels, valid, y, _ in loader:
                x, labels, valid, y = [v.to(device, non_blocking=True) for v in (x, labels, valid, y)]
                opt.zero_grad(set_to_none=True)
                logits = model(x, labels, valid)
                if not torch.isfinite(logits).all():
                    skipped += 1
                    continue
                loss = F.cross_entropy(logits, y, label_smoothing=a.label_smoothing)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                loss_sum += float(loss) * len(y)
                seen += len(y)
                steps += 1
            if not steps:
                raise RuntimeError('NLGT: all training batches were nonfinite')
            loss_value, extra = loss_sum / seen, {'skipped_batches': skipped, 'training_views': 1}
        elif self.method == 'nagphormer':
            train = torch.as_tensor(self.splits['train'])
            perm = train[torch.randperm(len(train))]
            total, steps = 0., 0
            opt.zero_grad(set_to_none=True)
            for start in range(0, len(perm), a.batch_size):
                ids = perm[start:start + a.batch_size]
                tokens = self.exp._nag_batch_to_device(self.cache['token_seq'], ids)
                y = self.labels_t[ids.to(device, non_blocking=device.type == 'cuda')]
                loss = F.cross_entropy(model(tokens), y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                total += float(loss.detach()) * len(ids)
                steps += 1
            seen, loss_value, extra = len(train), total / len(train), {'training_views': 1}
        else:
            labels = self.labels_t
            if a.train_mode == 'full':
                loss_value, _ = n.train_one_epoch_full(model, opt, self.x_full, self.adj_t, labels,
                    self.splits['train'], self.graph.problem_type, a.grad_clip)
                seen, steps = len(self.splits['train']), 1
            else:
                loss_value, seen = n.train_one_epoch_batch(model, opt, self.graph, labels, self.train_mask,
                    device, a.batch_size, self.graph.problem_type, a.grad_clip)
                steps = None  # partition batches without train labels are skipped by the native helper
            extra = {'training_views': 1}
        if self.schedule:
            self.schedule.step()
        return {'loss': loss_value, 'supervised_centers': int(seen), 'optimizer_steps': steps, **extra}

    def eval_due(self, epoch):
        return epoch == self.a.epochs or epoch % self.a.eval_every == 0 or (
            self.method == 'nagphormer' and epoch == 1)

    @torch.no_grad()
    def predict(self, ids, best_epoch=0, validation=False):
        a, n, device = self.a, self.n, self.device
        self.model.eval()
        extra = {}
        if self.method == 'scope_gt':
            tally = n.Tally(a.max_hops, self.sampler.graph.degree)
            views = a.train_eval_contexts if validation else a.eval_contexts
            logits = n.predict(self.model, self.sampler, ids, a, device, self.seed, views,
                               self.features, tally, self.eval_cache)
            extra = {**tally.summary(), 'per_center_unit': 'center_view', 'inference_views': views}
            if self.eval_cache:
                extra['structure_cache'] = self.eval_cache.statistics()
        elif self.method == 'nlgt':
            stream = 104729 if validation else 15485863
            logits = n.predict_with_votes(model=self.model, centers=ids, feat=self.features, labels=self.labels,
                sampler=self.sampler, max_nodes=a.max_nodes, num_classes=self.classes,
                batch_size=a.eval_batch_size, device=device, votes=a.eval_votes,
                base_seed=self.seed + best_epoch * stream, strict_no_neighbor_labels=True,
                train_mask=self.train_mask)
            extra['inference_views'] = a.eval_votes
        elif self.method == 'nagphormer':
            chunks = []
            for start in range(0, len(ids), a.eval_batch_size):
                tokens = self.exp._nag_batch_to_device(self.cache['token_seq'], torch.as_tensor(ids[start:start + a.eval_batch_size]))
                chunks.append(self.model(tokens).float().cpu())
            logits = torch.cat(chunks).numpy()
        else:
            if a.eval_mode == 'full_cpu':
                # No GPU training model/optimizer is copied at inference: CPU checkpoint-only model.
                if next(self.model.parameters()).device.type == 'cpu':
                    logits_t = self.model(self.cpu_cache['x'], self.cpu_cache['adj'])
                else:
                    logits_t = n.predict_full_cpu(self.model, self.graph, self.cpu_cache)
            elif a.eval_mode == 'full':
                logits_t = self.model(self.x_full, self.adj_t)
            else:
                logits_t = n.predict_in_batches(self.model, self.graph, device, a.eval_batch_size)
            logits = logits_t[torch.as_tensor(ids, device=logits_t.device)].float().cpu().numpy()
        return logits, extra

    def storage(self):
        if self.method == 'scope_gt':
            return {'graph_index_array_bytes': array_bytes(self.index, self.direction, self.sampler.log_degree),
                    'feature_storage_bytes': array_bytes(self.features),
                    'structure_cache': self.eval_cache.statistics() if self.eval_cache else None}
        if self.method == 'nlgt':
            return {'adjacency_array_bytes': array_bytes(self.adj), 'feature_storage_bytes': array_bytes(self.features),
                    'note': 'Python adjacency-list object overhead is included in RSS, not ndarray byte count'}
        if self.method == 'nagphormer':
            return {'token_storage_bytes': array_bytes(self.cache), 'token_artifact_bytes': (self.artifact_dir / 'tokens.pt').stat().st_size}
        return {'graph_feature_array_bytes': array_bytes(self.graph, self.cpu_cache, self.x_full, self.adj_t)}


def seed_native(method, seed, config=None):
    n = native(method)
    if method == 'scope_gt':
        n.seed_all(seed)
    elif method == 'nagphormer':
        n.set_seed(seed, deterministic=(config or {}).get('deterministic', True))
    else:
        n.set_seed(seed)
