"""Native samplers, identical sampled reads, and either native padding or packed memory."""
import hashlib
import numpy as np
import torch


def nlgt_graphs(sampler, features, labels, train_mask, classes, cfg, centers, indices, sample_seed):
    result=[]
    for center, index in zip(centers,indices):
        center=int(center)
        rng=np.random.default_rng(sample_seed+int(index)*10007+center*97)
        nodes=np.asarray(sampler.sample_subgraph(center,rng)[:cfg['max_nodes']],dtype=np.int64)
        tokens=np.full(len(nodes),classes+1,dtype=np.int64)
        if not cfg['strict_no_neighbor_labels']:
            y=labels[nodes]
            keep=train_mask[nodes] & (y>=0) & (y<classes)
            tokens[keep]=y[keep]
        tokens[0]=classes+1
        result.append(dict(nodes=nodes,feat=features[nodes],tokens=tokens,real_n=len(nodes),center=center))
    return result


def gophormer_graphs(sampler,centers,views,rng):
    return [sampler.build_single(int(c),rng) for c in centers for _ in range(views)]


def draw_digest(graphs):
    h=hashlib.sha256()
    for g in graphs:
        h.update(np.asarray([int(g['center']),len(g['nodes'])],dtype='<i8').tobytes())
        h.update(np.asarray(g['nodes'],dtype='<i8').tobytes())
        if 'tokens' in g: h.update(np.asarray(g['tokens'],dtype='<i8').tobytes())
        if 'phi' in g: h.update(g['phi'][0].astype('<f4',copy=False).tobytes())
    return h.hexdigest()


def assemble(graphs,method,variant,cfg,classes):
    """CPU assembly stays inside epoch timing. No padded arrays are built for SCMB."""
    b=len(graphs)
    real_lengths=[len(g['nodes']) for g in graphs]
    ng=cfg.get('global_nodes',0)
    width=cfg['max_nodes'] if method=='nlgt' else max(real_lengths)+ng
    reads=sum(real_lengths)+b*ng
    unique_nodes=len(set(int(u) for g in graphs for u in g['nodes']))
    base=dict(sequences=b,real_occurrences=sum(real_lengths),unique_real_nodes=unique_nodes,
              virtual_occurrences=b*ng,valid_reads=reads,native_padded_slots=b*width,
              native_padding_slots=b*width-reads,native_attention_allocated_pairs=b*width*width)
    if variant=='scmb':
        rowmap={}; feats=[]; labels=[]; virtual=[]; rows=[]; owner=[]; centers=[]; phi=[]
        for i,g in enumerate(graphs):
            first=len(rows)
            for j,u in enumerate(g['nodes']):
                token=int(g['tokens'][j]) if method=='nlgt' else classes+1
                # Role/label-conditioned inputs are never merged merely by node ID.
                key=(int(u),token)
                if key not in rowmap:
                    rowmap[key]=len(feats); feats.append(g['feat'][j]); labels.append(token); virtual.append(-1)
                rows.append(rowmap[key]); owner.append(i)
            centers.append(rows[first])
            for j in range(ng):
                # Keep every sequence's virtual-token row private, even with shared embedding parameters.
                rows.append(len(feats)); owner.append(i)
                feats.append(np.zeros(g['feat'].shape[1],dtype=np.float32)); labels.append(classes+1); virtual.append(j)
            if method=='gophormer': phi.append(g['phi'][0])
        data=dict(feat=torch.from_numpy(np.asarray(feats,dtype=np.float32)),
                  tokens=torch.tensor(labels,dtype=torch.long),virtual=torch.tensor(virtual,dtype=torch.long),
                  read_rows=torch.tensor(rows,dtype=torch.long),owner=torch.tensor(owner,dtype=torch.long),
                  center_rows=torch.tensor(centers,dtype=torch.long))
        if phi: data['phi']=torch.from_numpy(np.concatenate(phi,axis=0))
        base.update(memory_rows=len(feats),projection_rows_per_layer=len(feats),
                    allocated_padding_slots=0,attention_pairs_per_layer=reads)
    else:
        feat=np.zeros((b,width,graphs[0]['feat'].shape[-1]),np.float32)
        mask=np.zeros((b,width),bool)
        tokens=np.full((b,width),classes+1,np.int64)
        if method=='gophormer':
            shape=(b,width,width,cfg['prox_order']+2) if variant=='native' else (b,width,cfg['prox_order']+2)
            phi=np.zeros(shape,np.float32)
        for i,g in enumerate(graphs):
            n=real_lengths[i]; feat[i,:n]=g['feat'][:n]; mask[i,:n]=True
            if method=='nlgt': tokens[i,:n]=g['tokens']
            else:
                if ng: mask[i,-ng:]=True
                if variant=='native':
                    phi[i,:n,:n]=g['phi'][:n,:n]
                    if ng:
                        phi[i,-ng:,:n,-1]=1.; phi[i,:n,-ng:,-1]=1.; phi[i,-ng:,-ng:,-1]=1.
                else:
                    phi[i,:n]=g['phi'][0,:n]
                    if ng: phi[i,-ng:]=g['phi'][0,n:]
        data=dict(feat=torch.from_numpy(feat),mask=torch.from_numpy(mask),tokens=torch.from_numpy(tokens))
        if method=='gophormer': data['phi']=torch.from_numpy(phi)
        base.update(memory_rows=b*width,projection_rows_per_layer=b*width,
                    allocated_padding_slots=b*width-reads,
                    attention_pairs_per_layer=b*width*width if variant=='native' else b*width)
    data['size']=b
    return data,base


def to_device(batch,device):
    return {k:v.to(device,non_blocking=device.type=='cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
