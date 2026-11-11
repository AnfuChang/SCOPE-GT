"""Dense, parameter-matched Center / Partial-query / Full-query readers.

Sampling and typed center-read relations are supplied by SCOPE_GT unchanged.
Only the scope of attention-block updates differs. No context outputs are
computed in the final layer because the classifier consumes only the center.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from SCOPE_GT.model import SCOPEGT


class DepthSCOPEGT(SCOPEGT):
    def __init__(self, *args, role='center', partial_ratio=.5, partial_seed=0, **kwargs):
        if role not in ('center', 'partial', 'full'):
            raise ValueError('role must be center, partial, or full')
        if not 0 <= partial_ratio <= 1 or partial_seed < 0:
            raise ValueError('partial_ratio must be in [0,1]; partial_seed must be nonnegative')
        kwargs['query'] = 'center' if role == 'center' else 'context'
        super().__init__(*args, **kwargs)
        self.role, self.partial_ratio, self.partial_seed = role, float(partial_ratio), int(partial_seed)
        self._query_context = (False, 0, 0)
        self._capture = False
        self.last_center_embeddings = None

    def set_query_context(self, epoch=0, view=0, training=False):
        """Independent selection stream; evaluation does not depend on epoch."""
        self._query_context = (bool(training), int(epoch) if training else 0, int(view))

    def capture_embeddings(self, enabled=True):
        self._capture = bool(enabled)
        self.last_center_embeddings = None

    def prepare_query_selection(self, batch):
        """Cache valid query rows on this batch, outside isolated reader timings.

        Each node receives a deterministic priority keyed by seed, phase,
        epoch, view, center and context identity. Selection is independent of
        global RNG state and batch order. The selected set is reused across
        layers. Returned indices contain only actual valid context queries.
        """
        if batch.packed:
            raise ValueError('depth comparisons use the common dense layout')
        key = (self.role, self.partial_ratio, self.partial_seed, self._query_context)
        saved = getattr(batch, '_depth_queries', None)
        if saved is not None and saved[0] == key:
            return saved[1:]
        if self.role == 'center' or len(self.blocks) == 1 or (self.role == 'partial' and self.partial_ratio == 0):
            empty = torch.empty(0, dtype=torch.long, device=batch.features.device)
            batch._depth_query_columns = empty.reshape(1, 0).expand(batch.size, 0)
            batch._depth_query_valid = batch._depth_query_columns.bool()
            batch._depth_queries = (key, empty, empty)
            return empty, empty
        if self.role == 'full' or self.partial_ratio == 1:
            # DenseBatch already carries the valid pair indices. Reuse them
            # without a device-to-host copy or redundant query-set discovery.
            owner, col = batch.pair_rows, batch.pair_cols
            batch._depth_query_columns = torch.arange(batch.ctx_valid.shape[1],
                device=batch.features.device)[None].expand(batch.size, -1)
            batch._depth_query_valid = batch.ctx_valid
            batch._depth_queries = (key, owner, col)
            return owner, col
        valid = batch.ctx_valid.detach().cpu().numpy()
        cp = batch.center_pos.detach().cpu().numpy()
        xp = batch.ctx_pos.detach().cpu().numpy()
        centers, contexts = batch.pool_ids[cp], batch.pool_ids[xp]
        owner, col = [], []
        if self.role != 'center' and len(self.blocks) > 1:
            train, epoch, view = self._query_context
            for row in range(len(valid)):
                cols = np.flatnonzero(valid[row])
                count = len(cols) if self.role == 'full' else math.ceil(len(cols) * self.partial_ratio)
                if count and count < len(cols):
                    prefix = f'{self.partial_seed}:{int(train)}:{epoch}:{view}:{int(centers[row])}:'
                    priorities = [int.from_bytes(hashlib.blake2b(
                        (prefix + str(int(contexts[row, c]))).encode('ascii'), digest_size=8).digest(), 'little')
                        for c in cols]
                    order = np.lexsort((contexts[row, cols], np.asarray(priorities, dtype=np.uint64)))
                    cols = cols[order[:count]]
                else:
                    cols = cols[:count]
                owner.extend([row] * len(cols))
                col.extend(cols.tolist())
        device = batch.features.device
        counts = np.bincount(np.asarray(owner, dtype=np.int64), minlength=batch.size)
        width = int(counts.max(initial=0))
        query_cols = np.zeros((batch.size, width), dtype=np.int64)
        query_valid = np.zeros((batch.size, width), dtype=bool)
        col_cpu = np.asarray(col, dtype=np.int64)
        offset = 0
        for row, count in enumerate(counts):
            query_cols[row, :count] = col_cpu[offset:offset + count]
            query_valid[row, :count] = True
            offset += count
        batch._depth_query_columns = torch.as_tensor(query_cols, dtype=torch.long, device=device)
        batch._depth_query_valid = torch.as_tensor(query_valid, dtype=torch.bool, device=device)
        owner = torch.as_tensor(owner, dtype=torch.long, device=device)
        col = torch.as_tensor(col, dtype=torch.long, device=device)
        batch._depth_queries = (key, owner, col)
        return owner, col

    def forward(self, batch):
        if batch.packed:
            raise ValueError('all query roles must use the common dense layout')
        pool = self.encoder(batch.features)
        pairs, selves = self._relation_rows(pool, batch)
        return self.read_encoded(batch, pool, pairs, selves)

    def read_encoded(self, batch, pool, pairs, selves):
        """Reader on encoded inputs, also used for isolated reader profiling."""
        g = pool[batch.center_pos]
        owner, col = self.prepare_query_selection(batch)
        update = owner.numel() > 0
        states = pool[batch.ctx_pos] if update else None
        mask = torch.cat([batch.ctx_valid.new_ones((batch.size, 1)), batch.ctx_valid], 1)
        rows = pool[batch.kv_rows]
        for layer, block in enumerate(self.blocks):
            if update:
                # The original reference also projected a whole unused base table.
                # Project only its consumed self rows, preserving the same self-memory.
                sk, sv = block.table(pool[batch.center_pos])
                ck, cv = block.table(states)
            else:
                bk, bv = block.table(rows)
                sk, sv = bk[batch.center_kv], bv[batch.center_kv]
                ck, cv = bk[batch.ctx_kv], bv[batch.ctx_kv]
            keys, values = torch.cat([sk[:, None], ck], 1), torch.cat([sv[:, None], cv], 1)
            center_k, center_v = keys, values
            if pairs is not None:
                dk, dv = block.deltas(pairs)
                sdk, sdv = block.deltas(selves)
                delta_k, delta_v = torch.zeros_like(keys), torch.zeros_like(values)
                delta_k[batch.pair_rows, batch.pair_cols + 1] = dk
                delta_v[batch.pair_rows, batch.pair_cols + 1] = dv
                delta_k[:, 0], delta_v[:, 0] = sdk, sdv
                center_k, center_v = keys + delta_k, values + delta_v
            output = block.attend_dense(g[:, None], center_k, center_v, mask)[:, 0]
            new_g = block.finish(g, output, torch.ones(batch.size, dtype=torch.bool, device=g.device))
            if update and layer + 1 < len(self.blocks):
                # Gather Q before attention. Query padding is limited to the
                # largest selected set in this batch (roughly K/2 for Partial).
                qc, qv = batch._depth_query_columns, batch._depth_query_valid
                queries = states.gather(1, qc[..., None].expand(-1, -1, states.shape[-1]))
                out = block.attend_dense(queries, keys, values, mask)
                updated = queries + block.drop(block.out(out))
                updated = updated + block.drop(block.ff(block.ff_ln(updated)))
                next_states = states.clone()
                next_states[owner, col] = updated[qv]
                states = next_states
            g = new_g
        z = self.final_ln(g)
        self.last_center_embeddings = z.detach() if self._capture else None
        return self.classifier(z)
