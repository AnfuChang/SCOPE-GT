"""Batches: one node table, flat pair-owned evidence records and either a packed or a dense layout.

  PackedBatch  only valid (center, context) pairs; attention is a segment softmax per center (final)
  DenseBatch   [B, 1 + T] padded layout with masks (reference path, context-update control, execution study)

scope='batch' deduplicates nodes over the logical batch, so a node used by several centers (or as
evidence) is encoded once and its base K/V row is shared. scope='center' keeps one node pool per
center (deduplicated only inside that center's sequence and evidence) and physically repeats nodes
shared by different centers. Pair-level fields and records are identical in all layouts.
"""
import numpy as np
import torch
import threading

PAIR_FIELDS = ('hop', 'edge', 'topofeat', 'connection_known', 'moment_count', 'moment_mass', 'context_bg_count')
OPTIONAL_PAIR_FIELDS = ('connection_count_used', 'connection_mass_used', 'connection_positions_used')

# Per-thread, per-device staging slabs. Events prevent overwriting a source
# while an asynchronous copy is still in flight. Only small metadata is staged;
# feature matrices retain the original placement and transfer policy.
_STAGING = threading.local()
_MAX_STAGING_BYTES = 16 * 2 ** 20  # per dtype/device, at most three retained slabs


def _copy_metadata(buffer, device):
    device = torch.device(device)
    if buffer.device == device:
        return buffer
    if device.type != 'cuda' or buffer.device.type != 'cpu' or not buffer.numel():
        return buffer.to(device)
    if buffer.numel() * buffer.element_size() > _MAX_STAGING_BYTES:
        return buffer.to(device)  # bounded staging memory, synchronous fallback
    entries = getattr(_STAGING, 'entries', None)
    if entries is None:
        entries = _STAGING.entries = {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = index, buffer.dtype
    old = entries.get(key)
    if old is not None and not old[1].query():
        old[1].synchronize()
    capacity = max(256, 1 << (buffer.numel() - 1).bit_length())
    if old is None or old[0].numel() < buffer.numel():
        old = torch.empty(capacity, dtype=buffer.dtype, pin_memory=True), torch.cuda.Event()
        entries[key] = old
    host, event = old
    source = host[:buffer.numel()]
    source.copy_(buffer)
    out = source.to(device, non_blocking=True)
    event.record(torch.cuda.current_stream(device))
    return out


def _metadata_slabs(fields):
    buffers, views = [], {}
    for dtype in (np.int64, np.float32, np.bool_):
        selected = [(name, np.asarray(value, dtype=dtype)) for name, value in fields.items()
                    if value.dtype == np.dtype(dtype)]
        if not selected:
            continue
        data = np.empty(sum(value.size for _, value in selected), dtype=dtype)
        offset = 0
        for name, value in selected:
            data[offset:offset + value.size] = value.reshape(-1)
            views[name] = len(buffers), offset, value.shape, value.size
            offset += value.size
        buffers.append(torch.from_numpy(data))
    return tuple(buffers), views


def _bind_metadata(batch, buffers, views):
    batch._metadata_buffers, batch._metadata_views = buffers, views
    for name, (group, offset, shape, size) in views.items():
        setattr(batch, name, buffers[group].narrow(0, offset, size).view(shape))


class _Batch:
    def __init__(self, centers, draw, features, device, scope='batch'):
        centers = np.asarray(centers, np.int64)
        if not len(centers):
            raise ValueError('a batch needs at least one center')
        if scope not in ('batch', 'center'):
            raise ValueError("scope must be 'batch' or 'center'")
        valid = draw['ctx_valid']
        rows, cols = np.nonzero(valid)
        n = 1 + max(int(x.max()) for x in (centers, draw['ctx'], draw['moment_nodes'], draw['context_bg_nodes']) if x.size)
        stride = n if scope == 'center' else 0
        # node-table keys: node id, or (center row, node id) when nodes are not shared across centers
        center_keys = np.arange(len(centers)) * stride + centers
        ctx_keys = rows * stride + draw['ctx'][rows, cols]
        moment_keys = draw['moment_rows'] * stride + draw['moment_nodes']
        bg_keys = draw['context_bg_rows'] * stride + draw['context_bg_nodes']
        keys = np.unique(np.concatenate([center_keys, ctx_keys, moment_keys, bg_keys]))
        node_ids = keys % n if stride else keys
        self.features = (features[torch.as_tensor(node_ids, dtype=torch.long, device=device)]
                         if torch.is_tensor(features) else torch.as_tensor(features[node_ids], device=device))
        cp, ep = np.searchsorted(keys, center_keys), np.searchsorted(keys, ctx_keys)
        kv = np.unique(np.r_[cp, ep])
        long = lambda x: torch.as_tensor(np.asarray(x, np.int64), device=device)
        real = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
        pair_ids = np.full(valid.shape, -1, np.int64)
        pair_ids[rows, cols] = np.arange(len(rows))
        self.size, self.pool_ids, self.scope = len(centers), node_ids, scope
        self.center_pos, self.pair_ctx_pos = long(cp), long(ep)
        self.pair_rows, self.pair_cols = long(rows), long(cols)
        self.kv_rows, self.center_kv = long(kv), long(np.searchsorted(kv, cp))
        self.pair_ctx_kv = long(np.searchsorted(kv, ep))
        for name in PAIR_FIELDS:
            value = draw[name][rows, cols]
            tensor = (torch.as_tensor(value, dtype=torch.bool, device=device) if name == 'connection_known'
                      else long(value) if name in ('hop', 'edge') else real(value))
            setattr(self, name, tensor)
        self.moment_owner = long(pair_ids[draw['moment_rows'], draw['moment_cols']])
        self.moment_node_pos = long(np.searchsorted(keys, moment_keys))
        self.moment_weight = real(draw['moment_weight'])
        self.moment_position = long(draw['moment_position'])
        self.moment_external = torch.as_tensor(draw.get('moment_external', np.zeros(len(draw['moment_nodes']), bool)),
                                              dtype=torch.bool, device=device)
        for name in OPTIONAL_PAIR_FIELDS:
            if name in draw:
                setattr(self, name, real(draw[name][rows, cols]))
        self.context_bg_owner = long(pair_ids[draw['context_bg_rows'], draw['context_bg_cols']])
        self.context_bg_node_pos = long(np.searchsorted(keys, bg_keys))
        self.encoded_rows, self.kv_count = len(keys), len(kv)
        self.unique_nodes = len(np.unique(node_ids)) if stride else len(keys)
        self.pairs = len(rows)
        self._dense_shape = valid.shape

    def to(self, device):
        """Copy with every tensor moved to `device` (keeps host-to-device transfer out of timings)."""
        out = object.__new__(type(self))
        out.__dict__.update({k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in self.__dict__.items()})
        return out


class PackedBatch(_Batch):
    """Only valid pairs; attention runs as a segment softmax over each center's [self, contexts]."""
    packed = True

    def __init__(self, centers, draw, features, device, scope='batch'):
        # Build graph-dependent indices on CPU once. Keep the legacy dense
        # constructor intact for the independent-sequence controls.
        centers = np.asarray(centers, np.int64)
        if not len(centers) or scope not in ('batch', 'center'):
            raise ValueError('nonempty centers and batch/center scope required')
        valid = draw['ctx_valid']
        rows, cols = np.nonzero(valid)
        n = 1 + max(int(x.max()) for x in (centers, draw['ctx'], draw['moment_nodes'], draw['context_bg_nodes']) if x.size)
        stride = n if scope == 'center' else 0
        center_keys = np.arange(len(centers)) * stride + centers
        ctx_keys = rows * stride + draw['ctx'][rows, cols]
        moment_keys = draw['moment_rows'] * stride + draw['moment_nodes']
        bg_keys = draw['context_bg_rows'] * stride + draw['context_bg_nodes']
        keys = np.unique(np.concatenate([center_keys, ctx_keys, moment_keys, bg_keys]))
        node_ids = keys % n if stride else keys
        cp, ep = np.searchsorted(keys, center_keys), np.searchsorted(keys, ctx_keys)
        kv = np.unique(np.r_[cp, ep])
        pair_ids = np.full(valid.shape, -1, np.int64)
        pair_ids[rows, cols] = np.arange(len(rows))
        fields = dict(node_index=node_ids, center_pos=cp, pair_ctx_pos=ep, pair_rows=rows,
                      pair_cols=cols, kv_rows=kv, center_kv=np.searchsorted(kv, cp),
                      pair_ctx_kv=np.searchsorted(kv, ep),
                      moment_owner=pair_ids[draw['moment_rows'], draw['moment_cols']],
                      moment_node_pos=np.searchsorted(keys, moment_keys),
                      moment_weight=np.asarray(draw['moment_weight'], np.float32),
                      moment_position=np.asarray(draw['moment_position'], np.int64),
                      moment_external=np.asarray(draw.get('moment_external', np.zeros(len(draw['moment_nodes']), bool)), np.bool_),
                      context_bg_owner=pair_ids[draw['context_bg_rows'], draw['context_bg_cols']],
                      context_bg_node_pos=np.searchsorted(keys, bg_keys))
        for name in PAIR_FIELDS:
            dtype = np.bool_ if name == 'connection_known' else np.int64 if name in ('hop', 'edge') else np.float32
            fields[name] = np.asarray(draw[name][rows, cols], dtype=dtype)
        for name in OPTIONAL_PAIR_FIELDS:
            if name in draw:
                fields[name] = np.asarray(draw[name][rows, cols], np.float32)
        owner = np.r_[rows, np.arange(len(centers))]
        order = np.argsort(owner, kind='stable')  # contexts then self, as before
        fields.update(read_owner=owner[order], read_order=order,
                      read_kv=np.r_[fields['pair_ctx_kv'], fields['center_kv']][order],
                      read_has=np.ones(len(centers), np.bool_))
        buffers, views = _metadata_slabs(fields)
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        _bind_metadata(self, tuple(_copy_metadata(b, device) for b in buffers), views)
        if torch.is_tensor(features):
            index = self.node_index if features.device == device else torch.as_tensor(node_ids, device=features.device)
            self.features = features[index].to(device)
        else:
            self.features = torch.as_tensor(features[node_ids], device=device)
        self.size, self.pool_ids, self.scope = len(centers), node_ids, scope
        self.encoded_rows, self.kv_count = len(keys), len(kv)
        self.unique_nodes = len(np.unique(node_ids)) if stride else len(keys)
        self.pairs, self._dense_shape = len(rows), valid.shape
        self._aliases()

    def _aliases(self):
        self.edge_center = self.pair_rows                       # owning center of each pair
        self.ctx_pos, self.ctx_kv = self.pair_ctx_pos, self.pair_ctx_kv

    def to(self, device):
        """Move three metadata slabs once, then recreate zero-copy field views."""
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        out = object.__new__(type(self))
        out.__dict__.update(self.__dict__)
        buffers = tuple(_copy_metadata(b, device) for b in self._metadata_buffers)
        _bind_metadata(out, buffers, self._metadata_views)
        out.features = self.features.to(device, non_blocking=device.type == 'cuda')
        out._aliases()
        return out


class DenseBatch(_Batch):
    """[B, T] padded layout; invalid slots are masked. Used as the reference path, for the
    context-update control (per-sequence private context states) and the execution study."""
    packed = False

    def __init__(self, centers, draw, features, device, scope='batch'):
        super().__init__(centers, draw, features, device, scope)
        b, t = self._dense_shape
        valid = torch.as_tensor(draw['ctx_valid'], device=device)
        self.ctx_valid = valid
        ctx_pos = torch.zeros((b, t), dtype=torch.long, device=device)
        ctx_kv = torch.zeros((b, t), dtype=torch.long, device=device)
        ctx_pos[self.pair_rows, self.pair_cols] = self.pair_ctx_pos
        ctx_kv[self.pair_rows, self.pair_cols] = self.pair_ctx_kv
        self.ctx_pos, self.ctx_kv = ctx_pos, ctx_kv
        self.slots = b * t
