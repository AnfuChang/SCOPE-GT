"""Keep native HubGT/SwapGT inputs and domains; deduplicate complete token inputs."""
import hashlib
import numpy as np
import torch


def input_digest(graphs):
    h = hashlib.sha256()
    for graph in graphs:
        for key in ('nodes', 'source', 'feat', 'bias'):
            if key in graph:
                x = np.ascontiguousarray(graph[key])
                h.update(key.encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def assemble_extra(graphs, method, variant, cfg, labels, classes, num_nodes):
    b, t = len(graphs), len(graphs[0]['nodes'])
    if any(len(g['nodes']) != t for g in graphs):
        raise ValueError('Native HubGT/SwapGT inputs must retain their fixed token lengths')
    anchors = np.arange(0, t, cfg['sample_num'] + 1) if method == 'swapgt' else np.array([0])
    expected = 2 * (cfg['num_aug'] + 1) if method == 'swapgt' else 1
    if len(anchors) != expected:
        raise ValueError('Missing SwapGT readout centers')
    ids = np.stack([g['nodes'] for g in graphs])
    source = np.stack([g.get('source', np.zeros(t, np.int64)) for g in graphs])
    feat = np.stack([g['feat'] for g in graphs]).astype(np.float32, copy=False)
    data = dict(size=b, labels=torch.as_tensor(labels, dtype=torch.long))
    if method == 'hubgt':
        bias = np.stack([g['bias'] for g in graphs]).astype(np.int64, copy=False)
        data['bias'] = torch.from_numpy(bias[:, 0, :, 0].copy())
    if variant == 'native':
        data['feat'] = torch.from_numpy(feat)
        if method == 'hubgt':
            data.update(ids=torch.from_numpy(ids.copy()), attn_bias=torch.from_numpy(bias))
    elif variant == 'center_padded':
        data.update(feat=torch.from_numpy(feat), node_ids=torch.from_numpy(ids.copy()),
                    query_rows=torch.as_tensor((np.arange(b)[:, None] * t + anchors).reshape(-1)))
    else:
        # A raw feature and a propagated feature of the same node are different inputs.
        keys = np.stack([ids.reshape(-1), source.reshape(-1)], axis=1)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        rows = inverse.reshape(b, t)
        data.update(feat=torch.from_numpy(feat.reshape(-1, feat.shape[-1])[first].copy()),
                    node_ids=torch.from_numpy(ids.reshape(-1)[first].copy()),
                    query_rows=torch.from_numpy(rows[:, anchors].reshape(-1).copy()),
                    read_rows=torch.from_numpy(rows.reshape(-1).copy()),
                    owner=torch.arange(b).repeat_interleave(t), queries_per_sequence=expected)
        if method == 'hubgt':
            data['bias'] = data['bias'].reshape(-1)
    real = ids[ids < num_nodes]
    memory_rows = len(data['feat']) if variant == 'scmb' else b * t
    reads = b * expected * t
    stats = dict(sequences=b, readout_queries=b * expected, real_occurrences=len(real),
                 unique_real_nodes=len(np.unique(real)), virtual_occurrences=int((ids >= num_nodes).sum()),
                 valid_reads=reads, valid_key_positions=b*t, native_padded_slots=b*t, native_padding_slots=0,
                 native_attention_allocated_pairs=b*t*t, memory_rows=memory_rows,
                 projection_rows_per_layer=memory_rows, allocated_padding_slots=0,
                 attention_pairs_per_layer=b*t*t if variant == 'native' else reads)
    return data, stats
