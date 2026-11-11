"""Decomposition arms that complement the Table IX Native / +SCMB runs.

V0 native         existing arm: full self-attention, native padding.
V1 native_packed  full self-attention and context updates as in Native, but no padding is
                  allocated: the sequences of a batch are grouped by exact length and every group
                  runs the unchanged native forward with an all-valid mask.
V2 cq_noshare     center-query attention over fixed input memory (the +SCMB architecture) with
                  packed reads, but no cross-sequence reuse: every sequence encodes its own tokens
                  and projects its own K/V. This is the unchanged +SCMB code on per-sequence rows.
V3 cq_sharedenc   as V2, but the input encoder runs once per unique batch input; occurrences
                  gather the shared encoding and still project their own K/V.
V4 scmb           existing arm: as V3, plus K/V projected once per unique input.

Steps V0->V1 (padding), V1->V2 (center-query architecture), V2->V3 (encoder sharing) and
V3->V4 (K/V sharing) add up to the Native -> +SCMB change reported in Table IX.

HubGT (48 tokens) and SwapGT (30 tokens) inputs have fixed lengths and no padding
(assemble_extra rejects unequal lengths and records zero padding slots), so V1 equals Native
for them and is not run.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .adapters import LegacyTransferAdapter, state_digest
from .batches import assemble
from .extra_backbones import ExtraTransferAdapter
from .extra_batches import assemble_extra
from .extra_models import ExtraTransferModel, attention
from .model import TransferModel

DECOMPOSED_VARIANTS = ('native_packed', 'cq_noshare', 'cq_sharedenc')
APPLICABLE = {'nlgt': DECOMPOSED_VARIANTS, 'gophormer': DECOMPOSED_VARIANTS,
              'hubgt': ('cq_noshare', 'cq_sharedenc'), 'swapgt': ('cq_noshare', 'cq_sharedenc')}
SKIP_REASON = {'hubgt': 'fixed 48-token inputs contain no padding, so V1 native_packed is identical to Native',
               'swapgt': 'fixed 30-token inputs contain no padding, so V1 native_packed is identical to Native'}
LABELS = {'native': 'FQ', 'native_packed': 'FQ-LG', 'cq_noshare': 'CQ-U',
          'cq_sharedenc': 'CQ-E', 'scmb': 'SCMB'}
ORDER = ('native', 'native_packed', 'cq_noshare', 'cq_sharedenc', 'scmb')
DEFINITIONS = {
    'native_packed': 'Native full self-attention; sequences grouped by exact length, unchanged native forward, no padding',
    'cq_noshare': 'Center-query reader over fixed memory; packed reads; per-sequence encodings and K/V',
    'cq_sharedenc': 'Center-query reader; encoder once per unique batch input; per-occurrence K/V',
}


def check_applicable(method, variant):
    if variant not in APPLICABLE.get(method, ()):
        raise ValueError(f'{method}/{variant} is not run: {SKIP_REASON.get(method, "unsupported arm")}')


# --------------------------------------------------------------------------- batches
def _common_stats(graphs, ng, native_width):
    b = len(graphs)
    real = sum(len(g['nodes']) for g in graphs)
    reads = real + b * ng
    return dict(sequences=b, real_occurrences=real,
                unique_real_nodes=len({int(u) for g in graphs for u in g['nodes']}),
                virtual_occurrences=b * ng, valid_reads=reads, native_padded_slots=b * native_width,
                native_padding_slots=b * native_width - reads,
                native_attention_allocated_pairs=b * native_width * native_width)


def assemble_native_packed(graphs, method, cfg, classes, pad_multiple=1):
    """V1. Each length group reuses the native assembly, so the token layout (Gophormer virtual
    tokens last, phi channels) is exactly the native one without the batch-wide padding."""
    if pad_multiple < 1:
        raise ValueError('pad_multiple must be >= 1')
    b = len(graphs)
    ng = cfg.get('global_nodes', 0) if method == 'gophormer' else 0
    lengths = [len(g['nodes']) + ng for g in graphs]
    native_width = cfg['max_nodes'] if method == 'nlgt' else max(lengths)
    groups = {}
    for i, n in enumerate(lengths):
        groups.setdefault(-(-n // pad_multiple) * pad_multiple, []).append(i)
    data = dict(size=b, groups=len(groups))
    slots = pairs = 0
    for gi, key in enumerate(sorted(groups)):
        index = groups[key]
        width = max(lengths[i] for i in index)
        # assemble() pads NLGT to cfg['max_nodes'] and Gophormer to the group maximum.
        group_cfg = dict(cfg, max_nodes=width) if method == 'nlgt' else cfg
        part, _ = assemble([graphs[i] for i in index], method, 'native', group_cfg, classes)
        for name in ('feat', 'mask', 'tokens', 'phi'):
            if name in part:
                data[f'g{gi}_{name}'] = part[name]
        data[f'g{gi}_index'] = torch.as_tensor(index, dtype=torch.long)
        slots += len(index) * width
        pairs += len(index) * width * width
    stats = _common_stats(graphs, ng, native_width)
    stats.update(memory_rows=slots, encoder_rows=slots, projection_rows_per_layer=slots,
                 allocated_padding_slots=slots - stats['valid_reads'], attention_pairs_per_layer=pairs,
                 forward_groups=len(groups))
    return data, stats


def assemble_center_query(graphs, method, variant, cfg, classes):
    """V2/V3 for NLGT and Gophormer. Reads keep the +SCMB order: real tokens, then the
    sequence-private virtual tokens; Gophormer phi is the center row of each sequence."""
    b = len(graphs)
    ng = cfg.get('global_nodes', 0) if method == 'gophormer' else 0
    shared = variant == 'cq_sharedenc'
    feats, tokens, virtual, occ, owner, firsts, phi = [], [], [], [], [], [], []
    unique = {}
    for i, g in enumerate(graphs):
        local = unique if shared else {}  # V2: reuse only within the sequence (Table VI "Packed")
        firsts.append(len(occ))
        for j, u in enumerate(g['nodes']):
            token = int(g['tokens'][j]) if method == 'nlgt' else classes + 1
            key = (int(u), token)  # role/label-conditioned inputs are never merged by node ID alone
            if key not in local:
                local[key] = len(feats)
                feats.append(g['feat'][j]); tokens.append(token); virtual.append(-1)
            occ.append(local[key]); owner.append(i)
        for j in range(ng):
            occ.append(len(feats)); owner.append(i)
            feats.append(np.zeros(g['feat'].shape[1], dtype=np.float32)); tokens.append(classes + 1); virtual.append(j)
        if method == 'gophormer':
            phi.append(g['phi'][0])
    reads = len(occ)
    data = dict(size=b, feat=torch.from_numpy(np.asarray(feats, dtype=np.float32)),
                tokens=torch.tensor(tokens, dtype=torch.long), virtual=torch.tensor(virtual, dtype=torch.long),
                owner=torch.tensor(owner, dtype=torch.long))
    if shared:
        data.update(occ_rows=torch.tensor(occ, dtype=torch.long), read_rows=torch.arange(reads),
                    center_rows=torch.tensor(firsts, dtype=torch.long))
    else:
        data.update(read_rows=torch.tensor(occ, dtype=torch.long),
                    center_rows=torch.tensor([occ[f] for f in firsts], dtype=torch.long))
    if phi:
        data['phi'] = torch.from_numpy(np.concatenate(phi, axis=0))
    native_width = cfg['max_nodes'] if method == 'nlgt' else max(len(g['nodes']) for g in graphs) + ng
    stats = _common_stats(graphs, ng, native_width)
    stats.update(memory_rows=len(feats), encoder_rows=len(feats),
                 projection_rows_per_layer=reads if shared else len(feats),
                 allocated_padding_slots=0, attention_pairs_per_layer=reads, forward_groups=1)
    return data, stats


def assemble_extra_center_query(graphs, method, variant, cfg, labels, classes, num_nodes):
    """V2/V3 for HubGT and SwapGT, built on the existing center_padded inputs (fixed lengths,
    so no padding). V3 encodes each unique (node, feature source) once, as +SCMB does."""
    data, stats = assemble_extra(graphs, method, 'center_padded', cfg, labels, classes, num_nodes)
    b, t = data['node_ids'].shape
    stats.update(memory_rows=b * t, encoder_rows=b * t, projection_rows_per_layer=b * t, forward_groups=1)
    if variant == 'cq_noshare':
        return data, stats
    ids = data['node_ids'].numpy().reshape(-1)
    source = np.stack([g.get('source', np.zeros(t, np.int64)) for g in graphs]).reshape(-1)
    _, first, inverse = np.unique(np.stack([ids, source], axis=1), axis=0, return_index=True, return_inverse=True)
    data['feat'] = data['feat'].reshape(b * t, -1)[torch.from_numpy(first)].contiguous()
    data['unique_node_ids'] = torch.from_numpy(ids[first].copy())
    data['occ_rows'] = torch.from_numpy(np.asarray(inverse, dtype=np.int64).reshape(-1))
    stats.update(memory_rows=len(first), encoder_rows=len(first))
    return data, stats


# --------------------------------------------------------------------------- models
class DecomposedTransferModel(TransferModel):
    """NLGT/Gophormer arms; no new parameters, so the initial state equals Native/+SCMB."""

    def __init__(self, backbone, method, variant):
        check_applicable(method, variant)
        # V2 runs the unchanged +SCMB packed reader; only its memory rows are per sequence.
        super().__init__(backbone, method, 'scmb' if variant == 'cq_noshare' else variant)
        self.arm = variant

    def forward(self, batch):
        if self.arm == 'native_packed':
            return self.native_packed(batch)
        if self.arm == 'cq_noshare':
            return super().forward(batch)
        return self.shared_encoder(batch)

    def native_packed(self, batch):
        m = self.backbone
        outputs, order = [], []
        for gi in range(batch['groups']):
            feat, mask = batch[f'g{gi}_feat'], batch[f'g{gi}_mask']
            if self.method == 'nlgt':
                outputs.append(m(feat, batch[f'g{gi}_tokens'], mask))
            else:
                outputs.append(m(feat, mask, batch[f'g{gi}_phi']))
            order.append(batch[f'g{gi}_index'])
        logits, order = torch.cat(outputs), torch.cat(order)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        return logits[inverse]  # original sequence order (Gophormer views stay center-major)

    def shared_encoder(self, batch):
        """The +SCMB packed reader with one change: encodings are gathered to occurrences
        before the layers, so K/V are projected per occurrence instead of per unique input."""
        m = self.backbone
        pool = m.input_proj(batch['feat'])
        if self.method == 'nlgt':
            pool = m.in_drop(pool + m.label_embed(batch['tokens']))
        else:
            if m.global_nodes:
                pool = pool.clone()
                indices = batch['virtual']; keep = indices >= 0
                pool[keep] = m.global_embed[indices[keep]]
            pool = m.input_dropout(pool)
        pool = pool[batch['occ_rows']]
        g = pool[batch['center_rows']]
        layers = m.blocks if self.method == 'nlgt' else m.layers
        for block in layers:
            att = block.attn
            if self.method == 'nlgt':
                ctx = block.norm1(pool); query = block.norm1(g); dim = g.shape[-1]
                q = F.linear(query, att.qkv.weight[:dim])
                k, v = F.linear(ctx, att.qkv.weight[dim:]).chunk(2, dim=-1)
                output = self.attend(q, k, v, batch, att.num_heads, att.attn_drop, True)
                g = g + att.proj_drop(att.proj(output))
                g = g + block.ffn(block.norm2(g))
            else:
                ctx = block.ln1(pool) if block.pre_ln else pool
                query = block.ln1(g) if block.pre_ln else g
                q, k, v = att.q_proj(query), att.k_proj(ctx), att.v_proj(ctx)
                output = att.o_proj(self.attend(q, k, v, batch, att.num_heads, att.attn_dropout, True, att.prox_bias))
                if block.pre_ln:
                    g = g + block.dropout(output)
                    g = g + block.dropout(block.ff(block.ln2(g)))
                else:
                    g = block.ln1(g + block.dropout(output))
                    g = block.ln2(g + block.dropout(block.ff(g)))
        if self.method == 'nlgt':
            return m.head(m.norm(g))
        return m.classifier(m.norm(g))


class DecomposedExtraModel(ExtraTransferModel):
    """HubGT/SwapGT arms; V2 is the existing center_padded reader (no padding exists)."""

    def __init__(self, backbone, method, variant):
        check_applicable(method, variant)
        super().__init__(backbone, method, 'center_padded' if variant == 'cq_noshare' else variant)
        self.arm = variant

    def forward(self, batch):
        if self.arm == 'cq_noshare':
            return super().forward(batch)
        return self.hubgt_shared(batch) if self.method == 'hubgt' else self.swapgt_shared(batch)

    def hubgt_shared(self, batch):
        m = self.backbone
        pool = m.node_encoder(batch['feat'])
        if m.var_vfeat:
            ids = batch['unique_node_ids']
            virtual = ids >= m.num_nodes
            pool = pool.clone()
            pool[virtual] += m.virtual_feat[ids[virtual] - m.num_nodes]
        pool = m.input_dropout(pool)
        pool = pool[batch['occ_rows']].reshape(batch['size'], -1, m.hidden_dim)
        view = dict(batch, feat=pool)  # attention() reads (sequences, tokens) from 'feat'
        g = pool.reshape(-1, m.hidden_dim)[batch['query_rows']]
        for layer in m.layers:
            att = layer.self_attention
            base, query = layer.self_attention_norm(pool), layer.self_attention_norm(g)
            q, k, v = att.linear_q(query), att.linear_k(base), att.linear_v(base)
            code = batch['bias']
            bias = att.bias_enc.linear_bias(code)
            bias = bias.masked_fill((code == 255)[..., None], -torch.inf)
            if att.dp_bias > 0:
                # Same rule as the native model and the existing arms: the query's own position is exempt.
                mask = torch.rand_like(bias) < att.dp_bias
                mask[:, 0] = False
                bias = bias.masked_fill(mask, -torch.inf)
            o = attention(q, k, v, view, att.num_heads, att.att_dropout, False, bias)
            g = g + layer.self_attention_dropout(att.output_layer(o))
            g = g + layer.ffn_dropout(layer.ffn(layer.ffn_norm(g)))
        return m.downstream_out_proj(m.final_ln(g))

    def swapgt_shared(self, batch):
        m = self.backbone
        pool = m.node_emb(batch['feat'])[batch['occ_rows']].reshape(batch['size'], -1, m.hidden_dim)
        view = dict(batch, feat=pool)
        g = pool.reshape(-1, m.hidden_dim)[batch['query_rows']]
        initial = g
        for layer in m.layers:
            att = layer.self_attention
            base, query = layer.self_attention_norm(pool), layer.self_attention_norm(g)
            q, k, v = att.linear_q(query), att.linear_k(base), att.linear_v(base)
            o = attention(q, k, v, view, att.num_heads, att.att_dropout, False)
            g = g + layer.self_attention_dropout(att.output_layer(o))
            g = g + layer.ffn_dropout(layer.ffn(layer.ffn_norm(g)))
            g = g + initial  # SwapGT's outer x0 residual, in addition to block residuals.
        centers = m.final_ln(g).reshape(batch['size'], 2 * (m.args.num_aug + 1), m.hidden_dim)
        attr, topo = centers.split(m.args.num_aug + 1, dim=1)
        loss_sim = (1 - F.cosine_similarity(attr, attr.mean(1, keepdim=True), dim=2)).mean()
        loss_sim = loss_sim + (1 - F.cosine_similarity(topo, topo.mean(1, keepdim=True), dim=2)).mean()
        out_attr = torch.cat([attr[:, 0], attr[:, 1:].mean(1)], dim=1)
        out_topo = torch.cat([topo[:, 0], topo[:, 1:].mean(1)], dim=1)
        fused = m.args.lambda2 * out_attr + (1 - m.args.lambda2) * out_topo
        logits = m.Linear1(torch.relu(m.out_proj(fused)))
        ce = m.CEloss(logits, batch['labels'])
        return logits.softmax(-1), ce + m.args.lambda1 * loss_sim, ce, loss_sim


# --------------------------------------------------------------------------- adapters
class DecomposedLegacyAdapter(LegacyTransferAdapter):
    def __init__(self, method, variant, cfg, root, seed, device, out, pad_multiple=1):
        super().__init__(method, variant, cfg, root, seed, device, out)
        self.pad_multiple = pad_multiple

    def prepare(self):
        super().prepare()
        self.model = DecomposedTransferModel(self.model.backbone, self.method, self.variant)
        self.base.model = self.model
        self.initial_state_sha256 = state_digest(self.model)

    def batch(self, graphs):
        if self.variant == 'native_packed':
            return assemble_native_packed(graphs, self.method, self.cfg, self.classes, self.pad_multiple)
        return assemble_center_query(graphs, self.method, self.variant, self.cfg, self.classes)


class DecomposedExtraAdapter(ExtraTransferAdapter):
    def prepare(self):
        super().prepare()
        self.model = DecomposedExtraModel(self.model.backbone, self.method, self.variant)
        self.initial_state_sha256 = state_digest(self.model)

    def batch(self, graphs):
        labels = self.labels[[g['center'] for g in graphs]]
        return assemble_extra_center_query(graphs, self.method, self.variant, self.cfg, labels,
                                           self.classes, self.data.num_nodes)


def make_decomposed_adapter(pad_multiple=1):
    """Factory with the TransferAdapter signature used by SCOPE_GT_transfer.worker."""
    def factory(method, variant, cfg, root, seed, device, out):
        check_applicable(method, variant)
        if method in ('hubgt', 'swapgt'):
            return DecomposedExtraAdapter(method, variant, cfg, root, seed, device, out)
        return DecomposedLegacyAdapter(method, variant, cfg, root, seed, device, out, pad_multiple)
    return factory
