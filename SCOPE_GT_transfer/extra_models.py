"""SCMB readers using unmodified backbone parameters and all native readout queries."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from SCOPE_GT.packed_ops import segment_softmax, segment_sum


def attention(q, k, v, batch, heads, drop, packed, bias=None):
    queries, dim = q.shape
    dh = dim // heads
    q = q.reshape(queries, heads, dh)
    if packed and batch.get('queries_per_sequence',1) > 1:
        # All SwapGT anchors share the original sequence's domain. Gather K/V once
        # per key position, not once per query-key pair (which would multiply memory).
        b = batch['size']; nq = batch['queries_per_sequence']
        t = len(batch['read_rows']) // b
        keys = k.reshape(-1,heads,dh)[batch['read_rows']].reshape(b,t,heads,dh)
        values = v.reshape(-1,heads,dh)[batch['read_rows']].reshape(b,t,heads,dh)
        score = torch.einsum('bqhd,bthd->bqht',q.reshape(b,nq,heads,dh),keys) / math.sqrt(dh)
        weights = drop(score.softmax(-1))
        return torch.einsum('bqht,bthd->bqhd',weights,values).reshape(queries,dim)
    if packed:
        owner, rows = batch['owner'], batch['read_rows']
        keys, values = k.reshape(-1, heads, dh)[rows], v.reshape(-1, heads, dh)[rows]
        score = (q[owner] * keys).sum(-1) / math.sqrt(dh)
        if bias is not None: score = score + bias
        weights = drop(segment_softmax(score, owner, queries))
        return segment_sum(weights[..., None] * values, owner, queries).reshape(queries, dim)
    b, t = batch['feat'].shape[:2]
    nq = queries // b
    keys, values = k.reshape(b, t, heads, dh), v.reshape(b, t, heads, dh)
    score = torch.einsum('bqhd,bthd->bqht', q.reshape(b,nq,heads,dh), keys) / math.sqrt(dh)
    if bias is not None: score = score + bias.permute(0,2,1)[:,None]
    weights = drop(score.softmax(-1))
    return torch.einsum('bqht,bthd->bqhd', weights, values).reshape(queries, dim)


class ExtraTransferModel(nn.Module):
    def __init__(self, backbone, method, variant):
        super().__init__()
        self.backbone, self.method, self.variant = backbone, method, variant

    def forward(self, batch):
        if self.method == 'hubgt': return self.hubgt(batch)
        return self.swapgt(batch)

    def hubgt(self, batch):
        m = self.backbone
        if self.variant == 'native':
            return m(SimpleNamespace(x=batch['feat'], ids=batch['ids'], attn_bias=batch['attn_bias']))
        packed = self.variant == 'scmb'
        pool = m.node_encoder(batch['feat'])
        if m.var_vfeat:
            virtual = batch['node_ids'] >= m.num_nodes
            pool = pool.clone()
            pool[virtual] += m.virtual_feat[batch['node_ids'][virtual] - m.num_nodes]
        pool = m.input_dropout(pool)
        g = pool.reshape(-1, m.hidden_dim)[batch['query_rows']]
        for layer in m.layers:
            att = layer.self_attention
            base, query = layer.self_attention_norm(pool), layer.self_attention_norm(g)
            q, k, v = att.linear_q(query), att.linear_k(base), att.linear_v(base)
            code = batch['bias']
            bias = att.bias_enc.linear_bias(code)
            bias = bias.masked_fill((code == 255)[...,None], -torch.inf)
            if att.dp_bias > 0:
                # Native bias dropout exempts the query's diagonal position, not all equal IDs.
                mask = torch.rand_like(bias) < att.dp_bias
                if packed:
                    t = len(code) // batch['size']
                    mask[torch.arange(len(code), device=code.device) % t == 0] = False
                else:
                    mask[:,0] = False
                bias = bias.masked_fill(mask, -torch.inf)
            o = attention(q,k,v,batch,att.num_heads,att.att_dropout,packed,bias)
            g = g + layer.self_attention_dropout(att.output_layer(o))
            g = g + layer.ffn_dropout(layer.ffn(layer.ffn_norm(g)))
        return m.downstream_out_proj(m.final_ln(g))

    def swapgt(self, batch):
        m = self.backbone
        if self.variant == 'native':
            return m(batch['feat'], batch['labels'], self.training, None)
        packed = self.variant == 'scmb'
        pool = m.node_emb(batch['feat'])
        g = pool.reshape(-1,m.hidden_dim)[batch['query_rows']]
        initial = g
        for layer in m.layers:
            att = layer.self_attention
            base, query = layer.self_attention_norm(pool), layer.self_attention_norm(g)
            q, k, v = att.linear_q(query), att.linear_k(base), att.linear_v(base)
            o = attention(q,k,v,batch,att.num_heads,att.att_dropout,packed)
            g = g + layer.self_attention_dropout(att.output_layer(o))
            g = g + layer.ffn_dropout(layer.ffn(layer.ffn_norm(g)))
            g = g + initial  # SwapGT's outer x0 residual, in addition to block residuals.
        centers = m.final_ln(g).reshape(batch['size'], 2*(m.args.num_aug+1), m.hidden_dim)
        attr, topo = centers.split(m.args.num_aug+1, dim=1)
        loss_sim = (1-torch.nn.functional.cosine_similarity(attr,attr.mean(1,keepdim=True),dim=2)).mean()
        loss_sim = loss_sim + (1-torch.nn.functional.cosine_similarity(topo,topo.mean(1,keepdim=True),dim=2)).mean()
        out_attr = torch.cat([attr[:,0],attr[:,1:].mean(1)],dim=1)
        out_topo = torch.cat([topo[:,0],topo[:,1:].mean(1)],dim=1)
        fused = m.args.lambda2*out_attr + (1-m.args.lambda2)*out_topo
        logits = m.Linear1(torch.relu(m.out_proj(fused)))
        ce = m.CEloss(logits,batch['labels'])
        return logits.softmax(-1), ce+m.args.lambda1*loss_sim, ce, loss_sim
