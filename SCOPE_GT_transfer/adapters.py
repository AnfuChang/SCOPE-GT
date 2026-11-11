"""Paired training streams; keep each backbone's native losses and optimizer."""
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F

from SCOPE_GT_efficiency.adapters import make_adapter
from .batches import nlgt_graphs,gophormer_graphs,assemble,to_device,draw_digest
from .model import TransferModel


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LegacyTransferAdapter:
    def __init__(self,method,variant,cfg,root,seed,device,out):
        self.method,self.variant,self.cfg,self.seed,self.device=method,variant,cfg,seed,device
        self.base=make_adapter(method,cfg,root,seed,device,out)
        self.feature_location='cpu'
        self.steps=0

    def load(self):
        self.base.load()
        self.labels,self.splits,self.classes=self.base.labels,self.base.splits,self.base.classes
        # Gophormer's base loader is permissive: apply the same split checks to both backbones.
        all_ids=[]
        for split,ids in self.splits.items():
            if not len(ids) or ids.min()<0 or ids.max()>=len(self.labels) or np.any(self.labels[ids]<0):
                raise ValueError(f'Invalid {split} split')
            all_ids.extend(ids.tolist())
        if len(set(all_ids))!=len(all_ids): raise ValueError('Overlapping/duplicate split IDs')

    def prepare(self):
        self.base.prepare()
        if self.method=='gophormer':
            # The generic efficiency adapter also supports full-graph methods;
            # its unused full-label GPU tensor is not part of native Gophormer.
            del self.base.labels_t
        self.model=TransferModel(self.base.model,self.method,self.variant)
        self.initial_state_sha256=state_digest(self.model)
        self.base.model=self.model
        self.train_order=self.splits['train'].copy()

    def optimizer(self):
        return self.base.optimizer()

    def graphs(self,ids,indices,sample_seed,rng=None,views=1):
        if self.method=='nlgt':
            return nlgt_graphs(self.base.sampler,self.base.features,self.labels,self.base.train_mask,
                               self.classes,self.cfg,ids,indices,sample_seed)
        return gophormer_graphs(self.base.sampler,ids,views,rng)

    def batch(self,graphs):
        return assemble(graphs,self.method,self.variant,self.cfg,self.classes)

    def epoch(self,epoch,opt):
        cfg,dev=self.cfg,self.device
        self.model.train()
        train=self.splits['train']
        if self.method=='nlgt':
            generator=torch.Generator().manual_seed(self.seed+epoch*65537)
            indices=torch.randperm(len(train),generator=generator).numpy()
            order=train[indices]; rng=None; views=1
        else:
            rng=np.random.RandomState((self.seed*1000003+epoch)%2**32)
            rng.shuffle(self.train_order)
            order=self.train_order; indices=np.arange(len(order)); views=cfg['train_samples']
        total=0.; steps=0; counts={}; digest=hashlib.sha256()
        warmup=cfg.get('warmup_epochs',0)*math.ceil(len(train)/cfg['batch_size']) if self.method=='gophormer' and self.base.plateau else 0
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]
            graphs=self.graphs(ids,indices[start:start+len(ids)],self.seed+epoch*7919,rng,views)
            digest.update(draw_digest(graphs).encode())
            batch,stats=self.batch(graphs)
            for key,value in stats.items(): counts[key]=counts.get(key,0)+value
            batch=to_device(batch,dev)
            if self.steps<warmup:
                for group in opt.param_groups: group['lr']=cfg['lr']*(self.steps+1)/warmup
            opt.zero_grad(set_to_none=True)
            logits=self.model(batch)
            target=torch.as_tensor(np.repeat(self.labels[ids],views),device=dev,dtype=torch.long)
            loss=F.cross_entropy(logits,target,label_smoothing=cfg.get('label_smoothing',0.))
            if self.method=='gophormer' and views>1 and cfg['cr_weight']>0:
                probs=torch.softmax(logits,-1).view(len(ids),views,-1)
                sharpened=self.base.n.sharpen_distribution(probs.mean(1),cfg['cr_temperature']).detach()
                loss=loss+cfg['cr_weight']*(probs-sharpened[:,None]).pow(2).sum(-1).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss; no batches are skipped')
            loss.backward()
            if cfg['grad_clip']>0: torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg['grad_clip'])
            opt.step(); self.steps+=1; steps+=1
            total+=float(loss.detach())*len(ids)
            # Release previous logical-batch activations before assembling the next batch.
            del logits,loss,batch,graphs,target
            if self.method=='gophormer' and views>1 and cfg['cr_weight']>0:
                del probs,sharpened
        if self.method=='nlgt': self.base.schedule.step()
        else: self.base.steps=self.steps
        return dict(loss=total/len(train),supervised_centers=len(train),optimizer_steps=steps,
                    training_views=views,input_sha256=digest.hexdigest(),batch_totals=counts)

    def eval_due(self,epoch):
        return epoch%self.cfg['eval_every']==0 or epoch==self.cfg['epochs'] or (self.method=='gophormer' and epoch==1)

    def after_validation(self,score):
        if self.method=='gophormer': self.base.after_validation(score)

    @torch.no_grad()
    def predict(self,ids,best_epoch=0,validation=False):
        self.model.eval()
        cfg=self.cfg; outputs=[]; digest=hashlib.sha256(); counts={}
        if self.method=='nlgt':
            # The native evaluator averages logits over votes, not class probabilities.
            values=np.zeros((len(ids),self.classes),np.float64)
            stream=104729 if validation else 15485863
            for vote in range(cfg['eval_votes']):
                sample_seed=self.seed+best_epoch*stream+vote*1000003
                for start in range(0,len(ids),cfg['eval_batch_size']):
                    part=ids[start:start+cfg['eval_batch_size']]
                    graphs=self.graphs(part,np.arange(start,start+len(part)),sample_seed)
                    digest.update(draw_digest(graphs).encode())
                    batch,stats=self.batch(graphs)
                    for key,value in stats.items(): counts[key]=counts.get(key,0)+value
                    values[start:start+len(part)]+=self.model(to_device(batch,self.device)).cpu().numpy()
            values/=cfg['eval_votes']
        else:
            # Native Gophormer uses fixed per-split samples and averages probabilities.
            rng=np.random.RandomState((self.seed*7919+(17 if validation else 31))%2**32)
            for start in range(0,len(ids),cfg['eval_batch_size']):
                part=ids[start:start+cfg['eval_batch_size']]
                graphs=self.graphs(part,None,None,rng,cfg['eval_samples'])
                digest.update(draw_digest(graphs).encode())
                batch,stats=self.batch(graphs)
                for key,value in stats.items(): counts[key]=counts.get(key,0)+value
                logits=self.model(to_device(batch,self.device))
                outputs.append(torch.softmax(logits,-1).view(len(part),cfg['eval_samples'],-1).mean(1).cpu().numpy())
            values=np.concatenate(outputs)
        return values,dict(input_sha256=digest.hexdigest(),batch_totals=counts)


def TransferAdapter(method, variant, cfg, root, seed, device, out):
    if method in ('hubgt', 'swapgt'):
        from .extra_backbones import ExtraTransferAdapter
        return ExtraTransferAdapter(method, variant, cfg, root, seed, device, out)
    return LegacyTransferAdapter(method, variant, cfg, root, seed, device, out)
