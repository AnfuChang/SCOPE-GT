"""node2vec positional encodings of GOAT (official/pos_enc/*_node2vec.py), computed once per graph and cached.

The official scripts train torch_geometric.nn.Node2Vec (p = q = 1, one negative walk per positive walk) with
SparseAdam on sparse embeddings and save embedding.weight; arxiv_ERM_ns.py then loads
pos_enc/<dataset>_embedding_<global_dim>.pt, so the embedding width equals --global_dim (64).
Settings of the scripts:
  arxiv_node2vec.py                         walk 80, context 20, 10 walks/node, batch 64,  lr 0.01, 5 epochs
  products_node2vec.py, pokec_node2vec.py   walk 40, context 20, 10 walks/node, batch 256, lr 0.01, 1 epoch
Node2Vec needs torch_cluster or pyg_lib for its random walks; neither is required here. This module
re-implements the parts of Node2Vec that the scripts use with the same definitions: uniform random walks
(p = q = 1; a node without neighbours repeats itself), the sliding context windows of pos_sample, uniformly
random negative walks of neg_sample, the skip-gram loss of Node2Vec.loss (EPS = 1e-15) and N(0, 1)
initialisation. Walks are drawn on the training device instead of in CPU loader workers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

PE_SETTINGS = {
    "arxiv": dict(walk_length=80, context_size=20, walks_per_node=10, batch_size=64, lr=0.01, epochs=5),
    "products": dict(walk_length=40, context_size=20, walks_per_node=10, batch_size=256, lr=0.01, epochs=1),
}
EPS = 1e-15


def _windows(rw: torch.Tensor, walk_length: int, context_size: int) -> torch.Tensor:
    num = 1 + walk_length + 1 - context_size
    return torch.cat([rw[:, j:j + context_size] for j in range(num)], dim=0)


def _random_walk(rowptr, col, start, walk_length):
    walk = [start]
    cur = start
    for _ in range(walk_length):
        lo = rowptr[cur]
        deg = rowptr[cur + 1] - lo
        r = (torch.rand(cur.shape[0], device=cur.device, dtype=torch.float64) * deg).long()
        r = torch.minimum(r, (deg - 1).clamp_min(0))
        nxt = col[(lo + r).clamp_max(col.shape[0] - 1)] if col.numel() else cur
        cur = torch.where(deg > 0, nxt, cur)
        walk.append(cur)
    return torch.stack(walk, dim=1)


def train_node2vec(adj: sp.csr_matrix, dim: int, walk_length: int, context_size: int, walks_per_node: int,
                   batch_size: int, lr: float, epochs: int, device: torch.device, seed: int, log_every: int = 0):
    assert walk_length >= context_size
    n = adj.shape[0]
    adj = adj.tocsr()
    rowptr = torch.from_numpy(adj.indptr.astype(np.int64)).to(device)
    col = torch.from_numpy(adj.indices.astype(np.int64)).to(device)
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(n, dim, sparse=True).to(device)          # reset_parameters: N(0, 1)
    opt = torch.optim.SparseAdam(list(emb.parameters()), lr=lr)
    steps = 0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = perm[i:i + batch_size]
            pos = _windows(_random_walk(rowptr, col, batch.repeat(walks_per_node), walk_length),
                           walk_length, context_size)
            nb = batch.repeat(walks_per_node)
            neg = torch.cat([nb.view(-1, 1), torch.randint(n, (nb.shape[0], walk_length), device=device)], dim=-1)
            neg = _windows(neg, walk_length, context_size)
            opt.zero_grad()
            h0 = emb(pos[:, 0]).view(pos.shape[0], 1, dim)
            hr = emb(pos[:, 1:].reshape(-1)).view(pos.shape[0], -1, dim)
            pos_loss = -torch.log(torch.sigmoid((h0 * hr).sum(dim=-1).view(-1)) + EPS).mean()
            h0 = emb(neg[:, 0]).view(neg.shape[0], 1, dim)
            hr = emb(neg[:, 1:].reshape(-1)).view(neg.shape[0], -1, dim)
            neg_loss = -torch.log(1 - torch.sigmoid((h0 * hr).sum(dim=-1).view(-1)) + EPS).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            opt.step()
            steps += 1
            if log_every and steps % log_every == 0:
                print(f"[node2vec] epoch {epoch} step {steps} loss {float(loss):.4f}", flush=True)
    torch.random.set_rng_state(gen_state)
    return emb.weight.detach().cpu()


def _graph_key(adj: sp.csr_matrix) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(adj.shape, dtype=np.int64).tobytes())
    h.update(adj.indptr.astype(np.int64).tobytes())
    h.update(adj.indices.astype(np.int64).tobytes())
    return h.hexdigest()[:16]


def load_or_compute(adj: sp.csr_matrix, dataset: str, setting: str, dim: int, cache_dir: Path,
                    device: torch.device, seed: int):
    """Returns (pe [N, dim] float32 CPU, info dict with the time and peak GPU memory of the computation)."""
    cfg = dict(PE_SETTINGS[setting])
    key = _graph_key(adj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_node2vec{dim}_{setting}_seed{seed}_{key}"
    pt, meta = cache_dir / f"{stem}.pt", cache_dir / f"{stem}.json"
    if pt.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        info["cached"] = True
        print(f"[node2vec] cached {pt.name} (computed in {info['seconds']:.1f}s, "
              f"peak {info['peak_mem_mb']:.1f}MB)", flush=True)
        return torch.load(pt, map_location="cpu"), info
    print(f"[node2vec] computing {stem}: {cfg}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    pe = train_node2vec(adj, dim, device=device, seed=seed, log_every=1000, **cfg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak = 0.0
    seconds = time.perf_counter() - t0
    info = dict(cfg, dim=dim, seed=seed, graph_key=key, seconds=seconds, peak_mem_mb=peak, device=str(device))
    torch.save(pe, pt)
    meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
    info["cached"] = False
    print(f"[node2vec] done in {seconds:.1f}s, peak {peak:.1f}MB -> {pt}", flush=True)
    return pe, info
