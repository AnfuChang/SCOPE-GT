"""LastFM HubGT/SwapGT recipes, original preprocessing, and paired training streams."""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

from SCOPE_GT_budgets.common import sha
from .config import CODE
from .adapters import state_digest
from .batches import to_device
from .extra_batches import assemble_extra, input_digest
from .extra_models import ExtraTransferModel
from SCOPE_GT.external_backbones import baseline_path, source_key


def extra_config(method, dataset, updates=None):
    if dataset != 'lastfm_asia':
        raise ValueError('New HubGT/SwapGT transfer recipes currently target LastFM only')
    common = dict(epochs=1000,patience=30,eval_every=5,feature_normalize='none',grad_clip=0.)
    if method == 'hubgt':
        cfg = dict(n_layers=4,num_heads=8,hidden_dim=128,ffn_dim=128,dp_input=.1,dp_bias=.1,
                   dp_ffn=.5,dp_attn=.5,aggr_output=0,var_vfeat=1,weight_decay=.01,peak_lr=.0002,
                   end_lr=1e-9,sched_epochs=300,batch_size=1024,eval_batch_size=1024,
                   ss=48,s0=24,s0g=8,s1=12,feature_norm='auto',perturb_std=0.,num_workers=0)
    else:
        # Historical external SwapGT launcher LastFM fallback: official Photo dense recipe.
        cfg = dict(hidden=512,ffn_dim=64,n_layers=1,n_heads=8,dropout=.1,attention_dropout=.1,
                   alpha=.1,pp_k=3,num_aug=2,depth=2,sample_num=4,lambda1=10.,lambda2=.1,
                   batch_size=2048,eval_batch_size=2048,warmup_updates=400,tot_updates=1000,
                   peak_lr=.001,end_lr=.0001,weight_decay=1e-5)
    cfg.update(common)
    if set(updates or {}) - set(cfg):
        raise ValueError(f'Unknown {method} options: {sorted(set(updates)-set(cfg))}')
    cfg.update(updates or {})
    positive = ['epochs','eval_every','batch_size','eval_batch_size','n_layers','ffn_dim']
    positive += ['hidden_dim','num_heads','ss','sched_epochs'] if method == 'hubgt' else [
        'hidden','n_heads','num_aug','sample_num','depth','tot_updates']
    width, heads = (cfg['hidden_dim'],cfg['num_heads']) if method == 'hubgt' else (cfg['hidden'],cfg['n_heads'])
    drops = ('dp_input','dp_bias','dp_ffn','dp_attn') if method == 'hubgt' else ('dropout','attention_dropout')
    if any(cfg[k] < 1 for k in positive) or width % heads or any(not 0 <= cfg[k] < 1 for k in drops):
        raise ValueError('Invalid dimensions, counts or dropout')
    if cfg['patience'] < 0 or cfg['weight_decay'] < 0 or cfg['peak_lr'] <= 0 or cfg['grad_clip'] < 0:
        raise ValueError('Invalid optimizer/stopping settings')
    if method == 'hubgt' and (cfg['aggr_output'] != 0 or cfg['perturb_std'] != 0 or cfg['num_workers'] != 0):
        raise ValueError('LastFM HubGT transfer retains center readout, no feature perturbation, and synchronous native fetching')
    if method == 'hubgt' and (cfg['feature_norm'] not in ('auto','none','pyg') or min(cfg['s0'],cfg['s0g'],cfg['s1']) < 0):
        raise ValueError('Invalid HubGT index recipe')
    return cfg


def load_native(method):
    """Official copies import generic names such as model/lr: isolate those aliases."""
    folder = baseline_path(('HubGT' if method == 'hubgt' else 'SwapGT') + f'/{method}_npz.py').parent
    name = '_scope_transfer_native_' + method
    if name in sys.modules: return sys.modules[name]
    aliases = ('model','lr','hubgt_collate')
    saved = {key:sys.modules.pop(key) for key in aliases if key in sys.modules}
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(folder),str(folder.parent),str(folder/'Precompute')]
        spec = importlib.util.spec_from_file_location(name,folder/f'{method}_npz.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except BaseException:
        sys.modules.pop(name,None)
        raise
    finally:
        for key in aliases: sys.modules.pop(key,None)
        sys.modules.update(saved)
        sys.path[:] = previous


def extra_sources(method):
    folder = baseline_path('HubGT' if method == 'hubgt' else 'SwapGT')
    files = []
    for root in (folder,baseline_path('common')):
        for p in sorted(root.rglob('*')):
            if any(x in ('results','logs','figures','__pycache__','.git','cache','build') for x in p.relative_to(root).parts):
                continue
            if p.is_file() and p.suffix in ('.py','.json','.sh','.pyx','.pxd','.h','.so','.pyd'):
                files.append(p)
    for name in ('extra_backbones.py','extra_batches.py','extra_models.py'):
        files.append(Path(__file__).parent/name)
    files.append(CODE/'SCOPE_GT_budgets/common.py')
    files.append(CODE/'SCOPE_GT/external_backbones.py')
    return {source_key(p):sha(p) for p in files}


def prepare_hub_index(native, data, cfg, seed, out):
    """Production always uses the original, version-checked compiled PyPLL index."""
    folder = baseline_path('HubGT/Precompute').parent
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(folder),str(folder/'Precompute')]
        native.check_extension()
        PyPLL = importlib.import_module('Precompute').PyPLL
    except ImportError as error:
        raise RuntimeError('Build HubGT/Precompute with this Python: python setup.py build_ext --inplace --force') from error
    finally:
        sys.path[:] = previous
    edge = data.edge_index().astype(np.uint32)
    degree = np.bincount(edge[0],minlength=data.num_nodes)
    order = np.ascontiguousarray(np.argsort(degree,kind='stable')[::-1],dtype=np.uint32)
    index = PyPLL()
    index.set_args(True,seed,cfg['ss'],cfg['s0g'],cfg['s0'],cfg['s1'])
    path = Path(out)/'hub_index'
    if '/log' in path.as_posix():
        raise ValueError("HubGT's original index rewrites '/log' paths; use an output directory without '/log'")
    # Include the same cold index construction in every arm, including resumed failed jobs.
    index.get_index(edge,order,str(path),True,False)
    return index


class ExtraTransferAdapter:
    def __init__(self,method,variant,cfg,root,seed,device,out):
        self.method,self.variant,self.cfg,self.root,self.seed,self.device,self.out = method,variant,cfg,Path(root),seed,device,Path(out)
        self.n = load_native(method)
        self.feature_location = 'cpu'

    def load(self):
        args = SimpleNamespace(data_root=str(self.root),dataset='lastfm_asia',dataset_key='lastfm_asia',
                               split_suffixes=[''],feature_normalize=self.cfg['feature_normalize'])
        self.data = self.n.ProtocolData(args)
        self.labels,self.classes = self.data.labels,self.data.num_classes
        self.splits = self.data.splits('')
        if set(self.splits) != {'train','val','test'}:
            raise ValueError('Expected train/val/test splits')

    def prepare(self):
        n,d,cfg,dev = self.n,self.data,self.cfg,self.device
        n.set_seed(self.seed)
        self.a = SimpleNamespace(**cfg)
        if self.method == 'hubgt':
            norm = n.resolve_feature_norm(SimpleNamespace(feature_norm=cfg['feature_norm'],dataset_key='lastfm_asia'),d.features)
            features = n.pyg_normalize_features(d.features) if norm == 'pyg' else d.features
            self.features = np.concatenate([features,np.zeros((n.N_BPROOT,d.num_features),np.float32)])
            self.index = prepare_hub_index(n,d,cfg,self.seed,self.out)
            model = n.GT(n_layers=cfg['n_layers'],num_heads=cfg['num_heads'],input_dim=d.num_features,
                         hidden_dim=cfg['hidden_dim'],output_dim=1 if self.classes == 2 else self.classes,
                         attn_bias_dim=1,dp_input=cfg['dp_input'],dp_bias=cfg['dp_bias'],dp_ffn=cfg['dp_ffn'],
                         dp_attn=cfg['dp_attn'],ffn_dim=cfg['ffn_dim'],num_global_node=n.N_BPROOT,
                         num_nodes=d.num_nodes,var_vfeat=bool(cfg['var_vfeat']),aggr_output=False)
        else:
            shared = n.build_shared(d,self.a,dev)
            self.tokens = n.build_tokens(d,shared,self.a)
            model = n.SwapGT(n_layers=cfg['n_layers'],input_dim=d.num_features,hidden_dim=cfg['hidden'],
                             n_class=self.classes,num_heads=cfg['n_heads'],ffn_dim=cfg['ffn_dim'],
                             dropout_rate=cfg['dropout'],attention_dropout_rate=cfg['attention_dropout'],args=self.a)
        self.model = ExtraTransferModel(model.to(dev),self.method,self.variant)
        self.initial_state_sha256 = state_digest(self.model)

    def optimizer(self):
        cfg,n = self.cfg,self.n
        opt = torch.optim.AdamW(self.model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'])
        if self.method == 'hubgt':
            self.schedule = n.PolynomialDecayLR(opt,warmup=max(cfg['sched_epochs']//10,50),tot=cfg['sched_epochs'],
                                                lr=cfg['peak_lr'],end_lr=cfg['end_lr'],power=1.)
        else:
            self.schedule = n.PolynomialDecayLR(opt,warmup_updates=cfg['warmup_updates'],tot_updates=cfg['tot_updates'],
                                                lr=cfg['peak_lr'],end_lr=cfg['end_lr'],power=1.)
        return opt

    def graphs(self,ids):
        ids = np.asarray(ids,np.int64)
        if self.method == 'hubgt':
            nodes,bias = self.index.fetch_parallel(np.ascontiguousarray(ids,dtype=np.int32),self.cfg['ss'])
            nodes = np.asarray(nodes,np.int64).reshape(len(ids),self.cfg['ss'])
            bias = np.asarray(bias,np.int64).reshape(len(ids),self.cfg['ss'],self.cfg['ss'],1)
            if not np.array_equal(nodes[:,0],ids): raise ValueError('HubGT index changed target positions')
            return [dict(center=int(c),nodes=u.copy(),feat=self.features[u],bias=v.copy()) for c,u,v in zip(ids,nodes,bias)]
        nodes = torch.as_tensor(ids)
        attr,topo = self.tokens.attr_ids[nodes].numpy(),self.tokens.topo_ids[nodes].numpy()
        all_ids = np.concatenate([attr,topo],axis=1).astype(np.int64)
        source = np.zeros_like(all_ids)
        source[:,attr.shape[1]:] = self.tokens.topo_from_prop.numpy()[None]
        feat = self.tokens.gather(nodes,torch.device('cpu')).numpy()
        return [dict(center=int(c),nodes=u.copy(),source=s.copy(),feat=x) for c,u,s,x in zip(ids,all_ids,source,feat)]

    def batch(self,graphs):
        labels = self.labels[[g['center'] for g in graphs]]
        return assemble_extra(graphs,self.method,self.variant,self.cfg,labels,self.classes,self.data.num_nodes)

    def epoch(self,epoch,opt):
        cfg = self.cfg
        self.model.train()
        train = self.splits['train']
        generator = torch.Generator().manual_seed(self.seed+epoch*65537)
        order = train[torch.randperm(len(train),generator=generator).numpy()]
        import hashlib
        digest = hashlib.sha256(); counts = {}; total = 0.; steps = 0
        for start in range(0,len(order),cfg['batch_size']):
            ids = order[start:start+cfg['batch_size']]
            graphs = self.graphs(ids); digest.update(input_digest(graphs).encode())
            batch,stats = self.batch(graphs)
            for key,value in stats.items(): counts[key] = counts.get(key,0)+value
            batch = to_device(batch,self.device)
            opt.zero_grad(set_to_none=True)
            output = self.model(batch)
            if self.method == 'swapgt':
                loss = output[1]
            elif self.classes == 2:
                loss = F.binary_cross_entropy_with_logits(output[:,0],batch['labels'].to(output.dtype))
            else:
                loss = F.cross_entropy(output,batch['labels'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss; no batches are skipped')
            loss.backward()
            if cfg['grad_clip'] > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg['grad_clip'])
            opt.step()
            if self.method == 'swapgt': self.schedule.step()
            total += float(loss.detach())*len(ids); steps += 1
            del output,loss,batch,graphs
        if self.method == 'hubgt': self.schedule.step()
        return dict(loss=total/len(train),supervised_centers=len(train),optimizer_steps=steps,
                    training_views=2*(cfg['num_aug']+1) if self.method == 'swapgt' else 1,
                    input_sha256=digest.hexdigest(),batch_totals=counts)

    def eval_due(self,epoch):
        return epoch == 1 or epoch % self.cfg['eval_every'] == 0 or epoch == self.cfg['epochs']

    def after_validation(self,score): pass

    @torch.no_grad()
    def predict(self,ids,best_epoch=0,validation=False):
        import hashlib
        self.model.eval(); result = []; counts = {}; digest = hashlib.sha256()
        for start in range(0,len(ids),self.cfg['eval_batch_size']):
            graphs = self.graphs(ids[start:start+self.cfg['eval_batch_size']])
            digest.update(input_digest(graphs).encode())
            batch,stats = self.batch(graphs)
            for key,value in stats.items(): counts[key] = counts.get(key,0)+value
            output = self.model(to_device(batch,self.device))
            if self.method == 'swapgt': output = output[0]
            elif self.classes == 2:
                p = output.sigmoid(); output = torch.cat([1-p,p],dim=1)
            else: output = output.softmax(-1)
            result.append(output.cpu().numpy())
        return np.concatenate(result),dict(input_sha256=digest.hexdigest(),batch_totals=counts)
