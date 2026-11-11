"""Frozen recipes taken from the existing paper launchers; no new tuning."""
from pathlib import Path
import json

CODE = Path(__file__).resolve().parents[1]
METHODS = ('scope_gt', 'graphgps', 'exphormer', 'sgformer', 'gophormer', 'nlgt', 'swapgt')
NAMES = dict(zip(METHODS, ('SCOPE-GT', 'GraphGPS', 'Exphormer', 'SGFormer', 'Gophormer', 'NLGT', 'SwapGT')))
FOLDERS = {'scope_gt':'SCOPE_GT', 'graphgps':'GraphGPS', 'exphormer':'Exphormer',
           'sgformer':'SGFormer', 'gophormer':'Gophormer',
           'nlgt':'NLGT', 'swapgt':'SwapGT'}


def canonical(dataset):
    key=dataset.lower().replace('-','_')
    if key in ('computers', 'computer', 'amazon_computer'):
        return 'amazon_computers'
    return 'ogbn_products' if key in ('products', 'ogbnproducts') else key


def config_for(method, dataset, overrides=None, scaling=False):
    dataset = canonical(dataset)
    if dataset not in ('pubmed', 'arxiv_year', 'pokec', 'amazon_computers', 'ogbn_products') or method not in METHODS:
        raise ValueError('Unsupported efficiency dataset or method')
    large = dataset in ('pokec','arxiv_year','ogbn_products')
    if method == 'scope_gt':
        from SCOPE_GT_budgets.adapters import config_for as old_config
        cfg = old_config(method, dataset)
    elif method in ('nlgt', 'sgformer'):
        profiles = json.loads((CODE / 'SCOPE_GT_budgets/profiles.json').read_text(encoding='utf-8'))[method]
        cfg = {**profiles['defaults'], **profiles['datasets'].get(dataset,{})}
        if method=='sgformer' and dataset=='arxiv_year':
            # Existing run_sgformer.sh: sgf_arxiv(), full-graph official_large.
            cfg.update(hidden_dim=256,gcn_layers=3,alpha=.5,lr=.001,dropout=.5,gnn_dropout=.5,
                       weight_decay=0.,gnn_weight_decay=0.,arch='official_large',gnn_use_init=0,
                       train_mode='full',eval_mode='full',grad_clip=0.)
        cfg['metric'] = 'accuracy'
    elif method == 'graphgps':
        cfg = dict(hidden=64, layers=2, heads=4, dropout=.2, attn_dropout=0., norm='none',
                   global_model='performer', dense_attention_limit=12000, optimizer='adamw', lr=.0005,
                   weight_decay=1e-5, scheduler='cosine', min_lr=1e-5, clip_grad_norm=1.,
                   feature_norm='none', undirected=True, use_degree_pe=large, lap_pe_k=0 if large else 4,
                   epochs=1000, patience=100, eval_every=5, metric='accuracy')
    elif method == 'exphormer':
        width=96 if large else (72 if dataset=='pubmed' else 80)
        cfg = dict(hidden_dim=width, num_layers=3 if large else 4, num_heads=2,
                   ffn_dim=2*width, dropout=.3 if large else .4,
                   attn_dropout=.5 if large else .8, lr=.01 if large else .001,
                   weight_decay=.001, optimizer='adamw', scheduler='cosine', warmup_epochs=5,
                   expander_degree=6, num_virtual_nodes=0, norm='batch', feature_norm='none',
                   clip_grad_norm=1., epochs=1000, patience=100, eval_every=5, metric='accuracy')
    elif method == 'gophormer':
        cfg = dict(hid_dim=64, num_heads=8, num_layers=4 if large else 3, dim_feedforward=256,
                   dropout=.5, weight_decay=1e-5, end_lr=1e-9, fanouts=[7,7], lr=.0002,
                   batch_size=128 if large else 32, global_nodes=2, prox_order=1, cr_weight=.5,
                   train_samples=4, eval_samples=5, cr_temperature=.5,
                   eval_batch_size=256 if large else 64, grad_clip=1., prox_norm='row', pre_ln=False,
                   lr_scheduler='plateau', warmup_epochs=10, lr_decay=.5, lr_patience=5,
                   feature_normalize='none', epochs=1000, patience=30, eval_every=5, metric='accuracy')
    else:
        cfg = dict(hidden=512, ffn_dim=64, n_layers=1, n_heads=8, depth=2, alpha=.1, weight_decay=1e-5,
                   end_lr=.0001, warmup_updates=400, tot_updates=1000, batch_size=2048,
                   dropout=.1, attention_dropout=.5, peak_lr=.001 if large else .0005,
                   pp_k=2 if large else 4, sample_num=8 if large else 6, num_aug=2 if large else 3,
                   lambda1=.6 if large else 20., lambda2=.2 if large else .1,
                   feature_normalize='none', epochs=1000, patience=30, eval_every=5, metric='accuracy')
        if dataset in ('pubmed','ogbn_products'):
            # Existing run_swapgt.sh: fallback photo (dense) recipe on both datasets.
            cfg.update(attention_dropout=.1,peak_lr=.001,pp_k=3,sample_num=4,num_aug=2,lambda1=10.,lambda2=.1)
    # Original early-stopping units: SCOPE-GT counts checks (native interval=1),
    # GraphGPS/NLGT count epochs; SGFormer and the common trainers count checks.
    cfg['patience_epochs'] = cfg['patience'] * (cfg['eval_every'] if method in ('sgformer','exphormer','gophormer','swapgt') else 1)
    if method == 'gophormer':
        cfg['lr_patience'] *= cfg['eval_every']  # keep approximate epoch patience under per-epoch validation
    cfg['eval_every'] = 1
    updates = overrides or {}
    if set(updates) - set(cfg):
        raise ValueError(f'Unknown {method} overrides: {sorted(set(updates)-set(cfg))}')
    cfg.update(updates)
    if method == 'scope_gt' and cfg['centers_per_epoch'] != 0:
        raise ValueError('Efficiency epoch time requires centers_per_epoch=0 (complete training split)')
    if method == 'scope_gt' and cfg['query'] != 'center':
        raise ValueError('SCOPE-GT main method requires Center-query')
    if method == 'nlgt' and (not cfg['strict_no_neighbor_labels'] or cfg['num_workers'] != 0):
        raise ValueError('NLGT protocol requires no neighbor label inputs and DataLoader workers=0')
    if cfg['epochs'] < 1 or cfg['patience_epochs'] < 0:
        raise ValueError('Invalid epochs/patience')
    return cfg


def identity_note(method):
    if method == 'gophormer':
        return 'Bundled paper-guided dense Gophormer NPZ adapter; separate from the upstream DGL implementation'
    if method in ('graphgps','exphormer'):
        return 'Bundled dependency-light project implementation and NPZ launcher recipe'
    return 'Existing paper model and native training operations'
