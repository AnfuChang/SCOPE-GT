"""Center-query architectural adaptation using the backbone's own parameters.

Context memory is the original input encoding, fixed across attention layers within
this forward. It is differentiable and recomputed each batch, not an inference cache.
Gophormer's global embeddings remain private sequence rows and fixed K/V in this arm.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from SCOPE_GT.packed_ops import segment_softmax,segment_sum


class TransferModel(nn.Module):
    def __init__(self,backbone,method,variant):
        super().__init__()
        self.backbone,self.method,self.variant=backbone,method,variant

    def forward(self,batch):
        m=self.backbone
        if self.variant=='native':
            return m(batch['feat'],batch['tokens'],batch['mask']) if self.method=='nlgt' else m(batch['feat'],batch['mask'],batch['phi'])
        packed=self.variant=='scmb'
        pool=m.input_proj(batch['feat'])
        if self.method=='nlgt':
            pool=m.in_drop(pool+m.label_embed(batch['tokens']))
        else:
            if m.global_nodes:
                pool=pool.clone()
                if packed:
                    indices=batch['virtual']; keep=indices>=0
                    pool[keep]=m.global_embed[indices[keep]]
                else:
                    pool[:,-m.global_nodes:]=m.global_embed[None]
            pool=m.input_dropout(pool)
        if not packed: pool=pool*batch['mask'][...,None].to(pool.dtype)
        g=pool[batch['center_rows']] if packed else pool[:,0]
        layers=m.blocks if self.method=='nlgt' else m.layers
        for block in layers:
            att=block.attn
            if self.method=='nlgt':
                ctx=block.norm1(pool); query=block.norm1(g); dim=g.shape[-1]
                q=F.linear(query,att.qkv.weight[:dim])
                k,v=F.linear(ctx,att.qkv.weight[dim:]).chunk(2,dim=-1)
                output=self.attend(q,k,v,batch,att.num_heads,att.attn_drop,packed)
                g=g+att.proj_drop(att.proj(output))
                g=g+block.ffn(block.norm2(g))
            else:
                ctx=block.ln1(pool) if block.pre_ln else pool
                query=block.ln1(g) if block.pre_ln else g
                q,k,v=att.q_proj(query),att.k_proj(ctx),att.v_proj(ctx)
                output=att.o_proj(self.attend(q,k,v,batch,att.num_heads,att.attn_dropout,packed,att.prox_bias))
                if block.pre_ln:
                    g=g+block.dropout(output)
                    g=g+block.dropout(block.ff(block.ln2(g)))
                else:
                    g=block.ln1(g+block.dropout(output))
                    g=block.ln2(g+block.dropout(block.ff(g)))
        if self.method=='nlgt': return m.head(m.norm(g))
        return m.classifier(m.norm(g))

    @staticmethod
    def attend(q,k,v,batch,heads,drop,packed,prox_bias=None):
        b,d=q.shape; dh=d//heads
        q=q.reshape(b,heads,dh)
        if packed:
            owner,rows=batch['owner'],batch['read_rows']
            keys=k.reshape(-1,heads,dh)[rows]
            values=v.reshape(-1,heads,dh)[rows]
            scores=(q[owner]*keys).sum(-1)/math.sqrt(dh)
            if prox_bias is not None: scores=scores+(batch['phi']@prox_bias)[:,None]
            weights=drop(segment_softmax(scores,owner,b))
            return segment_sum(weights[...,None]*values,owner,b).reshape(b,d)
        keys=k.reshape(b,-1,heads,dh); values=v.reshape(b,-1,heads,dh)
        scores=(q[:,None]*keys).sum(-1)/math.sqrt(dh)
        if prox_bias is not None: scores=scores+(batch['phi']@prox_bias)[...,None]
        scores=scores.masked_fill(~batch['mask'][...,None],-float('inf'))
        weights=drop(torch.softmax(scores,dim=1))
        return (weights[...,None]*values).sum(1).reshape(b,d)
