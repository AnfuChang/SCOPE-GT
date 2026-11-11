"""Use the existing seven implementations, preserving their training losses."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from SCOPE_GT_budgets.adapters import Adapter as ExistingAdapter, seed_native
from .config import CODE
from SCOPE_GT.external_backbones import baseline_path


def load_native(method):
    paths = {'graphgps':'GraphGPS/npz_adapter/graphgps_node_baseline.py',
             'exphormer':'Exphormer/exphormer_npz.py',
             'gophormer':'Gophormer/gophormer_npz.py', 'swapgt':'SwapGT/swapgt_npz.py'}
    name = '_scope_efficiency_' + method
    if name not in sys.modules:
        path = baseline_path(paths[method])
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_adapter(method, cfg, root, seed, device, artifacts):
    cfg = dict(cfg)
    if method in ('scope_gt','nlgt','sgformer'):
        seed_native(method, seed, cfg)
        return ExistingAdapter(method, cfg, root, seed, device, artifacts)
    return AdditionalAdapter(method, cfg, root, seed, device, artifacts)


class AdditionalAdapter:
    def __init__(self, method, cfg, root, seed, device, artifacts):
        self.method, self.cfg, self.root, self.seed, self.device = method, cfg, Path(root), seed, device
        self.a = SimpleNamespace(**cfg, data_root=str(root), data_dir=str(root), dataset=root.name,
                                 split_suffixes=[''], cache_dir=str(artifacts / 'pe_cache'), allow_random_split=False)
        self.n = load_native(method)
        self.n.set_seed(seed)
        self.feature_location = 'cpu'

    def load(self):
        n, a = self.n, self.a
        if self.method == 'graphgps':
            self.data = n.load_bilgt_graph(a)
            self.labels = self.data.y.numpy().reshape(-1)
        elif self.method == 'exphormer':
            self.data = n.load_bilgt_graph(self.root, a.feature_norm)
            self.labels = self.data.y.numpy().reshape(-1)
        else:
            self.data = n.ProtocolData(a)
            self.labels = self.data.labels
        self.splits = {}
        for split in ('train','val','test'):
            ids = np.load(self.root / f'{split}.npy', allow_pickle=False)
            self.splits[split] = np.flatnonzero(ids) if ids.dtype == bool else ids.astype(np.int64).reshape(-1)
        self.classes = len(np.unique(self.labels[self.labels >= 0]))

    def prepare(self):
        n, a, d, dev = self.n, self.a, self.data, self.device
        self.schedule = self.plateau = None
        if self.method == 'graphgps':
            self.x = n.normalize_features(n.attach_pe(d, a), a.feature_norm).to(dev)
            edge = n.add_reverse_and_self_loops(d.edge_index, len(d.x), undirected=a.undirected)
            self.adj = n.build_norm_adj(edge, len(d.x), dev)
            self.model = n.GraphGPSNodeClassifier(self.x.shape[1], self.classes, a.hidden, a.layers, a.heads,
                a.dropout, a.attn_dropout, a.global_model, a.norm, a.dense_attention_limit).to(dev)
        elif self.method == 'exphormer':
            self.x = d.x.to(dev)
            att, typ = n.build_attention_edges(d.edge_index, len(d.x), self.seed, expander_degree=a.expander_degree)
            gcn = n.add_self_loops(n.add_reverse_edges(n.remove_self_loops(d.edge_index.cpu())), len(d.x))
            gcn, _ = n.coalesce_edges(gcn, None, len(d.x))
            self.edges = [att.to(dev), typ.to(dev), gcn.to(dev), n.gcn_norm(gcn,len(d.x)).to(dev)]
            self.model = n.BILGTExphormer(self.x.shape[1], self.classes, a.hidden_dim, a.num_layers, a.num_heads,
                a.dropout, a.attn_dropout, a.ffn_dim, a.norm, True, a.num_virtual_nodes).to(dev)
        elif self.method == 'gophormer':
            self.graph = n.GraphData(d)
            self.sampler = n.EgoGraphSampler(self.graph, a.fanouts, a.global_nodes, a.prox_order, a.prox_norm)
            self.model = n.GophormerModel(d.num_features, self.classes, a.hid_dim, a.num_layers, a.num_heads,
                a.dim_feedforward, a.dropout, a.global_nodes, a.prox_order, pre_ln=a.pre_ln).to(dev)
            self.steps = 0
        else:
            self.shared = n.build_shared(d, a, dev)
            self.tokens = n.build_tokens(d, self.shared, a)
            self.model = n.SwapGT(n_layers=a.n_layers, input_dim=d.num_features, hidden_dim=a.hidden,
                n_class=self.classes, num_heads=a.n_heads, ffn_dim=a.ffn_dim, dropout_rate=a.dropout,
                attention_dropout_rate=a.attention_dropout, args=a).to(dev)
            self.generator = torch.Generator().manual_seed(self.seed)
        self.labels_t = torch.as_tensor(self.labels, dtype=torch.long, device=dev)
        if self.method in ('graphgps','exphormer'):
            self.feature_location = str(dev)

    def optimizer(self):
        a, n = self.a, self.n
        if self.method == 'swapgt':
            opt = torch.optim.AdamW(self.model.parameters(), lr=a.peak_lr, weight_decay=a.weight_decay)
            self.schedule = n.PolynomialDecayLR(opt, warmup_updates=a.warmup_updates, tot_updates=a.tot_updates,
                                               lr=a.peak_lr, end_lr=a.end_lr, power=1.)
        elif self.method == 'gophormer':
            opt = torch.optim.Adam(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
            if a.lr_scheduler == 'plateau':
                self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=a.lr_decay,
                    patience=a.lr_patience, min_lr=a.end_lr, eps=0.)
        else:
            cls = {'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[a.optimizer.lower()]
            opt = cls(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
            if self.method == 'exphormer':
                self.schedule = n.make_scheduler(opt, a.epochs, a.warmup_epochs, a.scheduler)
            elif a.scheduler == 'cosine':
                self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.min_lr)
        return opt

    def full_logits(self):
        return self.model(self.x, self.adj) if self.method == 'graphgps' else self.model(self.x, *self.edges)

    def epoch(self, epoch, opt):
        a, n, dev = self.a, self.n, self.device
        self.model.train()
        train = self.splits['train']
        if self.method in ('graphgps','exphormer'):
            opt.zero_grad(set_to_none=True)
            ids = torch.as_tensor(train, device=dev)
            loss = F.cross_entropy(self.full_logits()[ids], self.labels_t[ids])
            loss.backward()
            if a.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.clip_grad_norm)
            opt.step()
            if self.schedule:
                self.schedule.step()
            return dict(loss=float(loss.detach()), supervised_centers=len(train), optimizer_steps=1, training_views=1)
        if self.method == 'gophormer':
            rng = np.random.RandomState((self.seed * 1000003 + epoch) % 2**32)
            # Match the native trainer's in-place order update between epochs.
            if not hasattr(self, 'train_order'):
                self.train_order = train.copy()
            rng.shuffle(self.train_order)
            order = self.train_order
        else:
            order = train[torch.randperm(len(train), generator=self.generator).numpy()]
        total, steps = 0., 0
        for start in range(0,len(order),a.batch_size):
            ids = order[start:start+a.batch_size]
            if self.method == 'gophormer':
                warm = a.warmup_epochs * math.ceil(len(train)/a.batch_size) if self.plateau else 0
                if self.steps < warm:
                    for group in opt.param_groups:
                        group['lr'] = a.lr * (self.steps + 1) / warm
                batch = self.sampler.collate(ids.tolist(), a.train_samples, rng)
                logits = self.model(batch['feat'].to(dev), batch['mask'].to(dev), batch['phi'].to(dev))
                loss = F.cross_entropy(logits, batch['labels'].to(dev))
                if a.train_samples > 1 and a.cr_weight > 0:
                    probs = torch.softmax(logits,-1).view(len(ids),a.train_samples,-1)
                    target = n.sharpen_distribution(probs.mean(1),a.cr_temperature).detach()
                    loss += a.cr_weight * (probs-target[:,None,:]).pow(2).sum(-1).mean()
            else:
                nodes = torch.as_tensor(ids)
                _, loss, _, _ = self.model(self.tokens.gather(nodes,dev),self.labels_t[ids],True,None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.method == 'gophormer' and a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),a.grad_clip)
            opt.step()
            if self.method == 'swapgt':
                self.schedule.step()
            else:
                self.steps += 1
            total += float(loss.detach()) * len(ids)
            steps += 1
        return dict(loss=total/len(train), supervised_centers=len(train), optimizer_steps=steps,
                    training_views=a.train_samples if self.method=='gophormer' else 2)

    @torch.no_grad()
    def predict(self, ids, best_epoch=0, validation=False):
        a, n, dev = self.a, self.n, self.device
        self.model.eval()
        if self.method in ('graphgps','exphormer'):
            return self.full_logits()[torch.as_tensor(ids,device=dev)].float().cpu().numpy(), {}
        outputs = []
        size = a.eval_batch_size if self.method == 'gophormer' else a.batch_size
        if self.method == 'gophormer':
            rng = np.random.RandomState((self.seed*7919+(17 if validation else 31)) % 2**32)
        for start in range(0,len(ids),size):
            nodes = ids[start:start+size]
            if self.method == 'gophormer':
                batch = self.sampler.collate(nodes.tolist(),a.eval_samples,rng)
                logits = self.model(batch['feat'].to(dev),batch['mask'].to(dev),batch['phi'].to(dev))
                out = torch.softmax(logits,-1).view(len(nodes),a.eval_samples,-1).mean(1)
            else:
                out, _, _, _ = self.model(self.tokens.gather(torch.as_tensor(nodes),dev), self.labels_t[nodes],False,None)
            outputs.append(out.float().cpu())
        # Both native implementations return class probabilities at inference.
        return torch.cat(outputs).numpy(), {'output_kind':'probabilities'}

    def after_validation(self, score):
        if self.plateau and self.steps >= self.a.warmup_epochs*math.ceil(len(self.splits['train'])/self.a.batch_size):
            self.plateau.step(score)
