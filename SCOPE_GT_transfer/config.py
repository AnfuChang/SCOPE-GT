"""Backbone recipes; native and transferred arms share this entire configuration."""
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ('nlgt', 'gophormer')
METHODS = DEFAULT_METHODS + ('hubgt', 'swapgt')
VARIANTS = ('native', 'scmb', 'center_padded')
DATASETS = ('lastfm_asia', 'minesweeper', 'wikics', 'corafull', 'amazon_computers', 'dblp')


def canonical(name):
    key = name.lower().replace('-', '_')
    return {'lastfm':'lastfm_asia', 'last_fm':'lastfm_asia', 'cora_full':'corafull', 'computers':'amazon_computers'}.get(key,key)


def config_for(method, dataset, updates=None):
    if method not in METHODS or canonical(dataset) not in DATASETS:
        raise ValueError(f'Unsupported backbone/dataset: {method}/{dataset}')
    if method in ('hubgt', 'swapgt'):
        from .extra_backbones import extra_config
        return extra_config(method, canonical(dataset), updates)
    if method == 'nlgt':
        profile = json.loads((CODE/'SCOPE_GT_budgets/profiles.json').read_text(encoding='utf-8'))['nlgt']
        cfg = {**profile['defaults'], **profile['datasets'].get(canonical(dataset),{})}
        # The reviewed paper profile uses the no-label-input protocol; embedding remains present.
    else:
        cfg = dict(hid_dim=64,num_layers=3,num_heads=8,dim_feedforward=256,dropout=.5,
            fanouts=[7,7],global_nodes=2,prox_order=1,prox_norm='row',pre_ln=False,
            batch_size=32,eval_batch_size=64,train_samples=4,eval_samples=5,
            cr_weight=.5,cr_temperature=.5,lr=.0002,weight_decay=1e-5,end_lr=1e-9,
            lr_scheduler='plateau',warmup_epochs=10,lr_decay=.5,lr_patience=5,grad_clip=1.,
            feature_normalize='none',epochs=1000,patience=30,eval_every=5)
    if set(updates or {})-set(cfg):
        raise ValueError(f'Unknown {method} options: {sorted(set(updates)-set(cfg))}')
    cfg.update(updates or {})
    positive = ['hid_dim','num_layers','num_heads','dim_feedforward','batch_size','eval_batch_size','epochs','eval_every']
    positive += ['max_nodes','eval_votes'] if method=='nlgt' else ['train_samples','eval_samples']
    if any(cfg[k]<1 for k in positive) or cfg['hid_dim']%cfg['num_heads'] or cfg['patience']<0:
        raise ValueError('Invalid widths, heads, epochs, batches, views or patience')
    if not 0 <= cfg['dropout'] < 1 or cfg['lr']<=0 or cfg['weight_decay']<0:
        raise ValueError('Invalid dropout or optimizer settings')
    if method=='nlgt' and cfg['num_workers']!=0:
        raise ValueError('The paired sampler uses a single CPU process (num_workers=0)')
    if method=='gophormer' and (cfg['global_nodes']<0 or cfg['prox_order']<0 or
            cfg['prox_norm'] not in ('row','sym') or cfg['lr_scheduler'] not in ('plateau','none')):
        raise ValueError('Invalid global tokens, proximity or scheduler')
    return cfg


def implementation(method):
    if method in ('hubgt', 'swapgt'):
        return f'External {"HubGT/hubgt_npz.py" if method == "hubgt" else "SwapGT/swapgt_npz.py"} NPZ adapter; original model and preprocessing'
    return ('External NLGT/nlgt_npz.py model and sampler' if method=='nlgt' else
            'External paper-guided Gophormer/gophormer_npz.py; not the official DGL adapter')


def quality(predictions, labels, dataset):
    if canonical(dataset) != 'minesweeper':
        return float((predictions.argmax(1)==labels).mean())
    from sklearn.metrics import roc_auc_score
    import numpy as np
    # Accept either vote-averaged logits or averaged probabilities.
    if not (np.all(predictions>=0) and np.allclose(predictions.sum(1),1.,atol=1e-5)):
        z=predictions-predictions.max(1,keepdims=True)
        predictions=np.exp(z)/np.exp(z).sum(1,keepdims=True)
    return float(roc_auc_score(labels,predictions[:,1]))
