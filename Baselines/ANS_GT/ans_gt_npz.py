import argparse
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # [ACE-baseline] imported at start-up, otherwise the one-off import cost is timed in the first validation
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# The classification OOT budget covers training only, not preprocessing or validation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.epoch_deadline import EpochDeadline


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_sparse_npz(path: str | Path) -> sp.csr_matrix:
    mat = sp.load_npz(path).tocsr().astype(np.float32)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adj.npz must be square, got shape={mat.shape}")
    return mat


def make_undirected_binary(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.maximum(adj.T)
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(adj.sum(1)).reshape(-1)
    inv_deg = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]
    return sp.diags(inv_deg).dot(adj).tocsr()



def validate_split_indices(labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                           test_idx: np.ndarray) -> None:
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        idx = np.asarray(idx)
        if idx.ndim != 1:
            raise ValueError(f"{name}.npy must be 1D, got shape={idx.shape}")
        if idx.size == 0:
            raise ValueError(f"{name}.npy is empty")
        if idx.min() < 0 or idx.max() >= len(labels):
            raise ValueError(f"{name}.npy contains out-of-range indices")
        if np.any(labels[idx] < 0):
            raise ValueError(f"{name}.npy contains unlabeled nodes (label < 0)")

    train_set = set(np.asarray(train_idx, dtype=np.int64).tolist())
    val_set = set(np.asarray(val_idx, dtype=np.int64).tolist())
    test_set = set(np.asarray(test_idx, dtype=np.int64).tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train/val/test splits overlap")


def load_fixed_split(data_root: str | Path, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_root = Path(data_root)
    train_p = data_root / "train.npy"
    val_p = data_root / "val.npy"
    test_p = data_root / "test.npy"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raise FileNotFoundError(
            "This script now requires a pre-generated fixed split: train.npy / val.npy / test.npy"
        )
    train_idx = np.load(train_p).astype(np.int64)
    val_idx = np.load(val_p).astype(np.int64)
    test_idx = np.load(test_p).astype(np.int64)
    validate_split_indices(labels, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


# Binary, class-imbalanced datasets reported with ROC-AUC (same list as ACE_GTv6/run_v3.sh and
# Baselines/common/protocol.py): model selection, per-run output and mean/std all use ROC-AUC there.
AUC_DATASETS = ("questions", "minesweeper", "tolokers")


def resolve_metric(metric: str, dataset: str) -> str:
    metric = metric.lower()
    if metric == "auto":
        return "rocauc" if dataset.lower().replace("-", "_") in AUC_DATASETS else "acc"
    return "rocauc" if metric in ("auc", "roc_auc", "rocauc") else "acc"


def metric_name(metric: str) -> str:
    return "ROC_AUC" if metric == "rocauc" else "Acc"


def print_efficiency(params, train_epoch_s, time_to_best_s, peak_mem_mb, pre_s=None, best_epoch=None) -> None:
    """Efficiency block with the same labels/definitions in every baseline (see common/protocol.py):
    Params = trainable parameters; Train time/epoch = training only (no evaluation), mean over epochs
    and runs; Time to best = training + validation time from epoch 1 to the end of the best-validation
    epoch (test evaluation and one-off preprocessing excluded), mean ± std over runs; Peak GPU mem =
    torch.cuda.max_memory_allocated, max over runs."""
    ttb = np.asarray(time_to_best_s, dtype=np.float64)
    print("==== Efficiency ====")
    print(f"Params: {int(params):,} ({int(params) / 1e6:.3f}M)")
    print(f"Train time/epoch (s): {float(np.mean(train_epoch_s)):.4f}")
    at = f" (best epoch {float(np.mean(best_epoch)):.1f})" if best_epoch is not None else ""
    print(f"Time to best (s): {ttb.mean():.2f} ± {(ttb.std(ddof=1) if ttb.size > 1 else 0.0):.2f}{at}")
    if pre_s is not None:
        pre = np.asarray(pre_s, dtype=np.float64)
        print(f"Preprocessing (s): {pre.mean():.2f}")
        print(f"Time to best incl. preprocessing (s): {(ttb + pre).mean():.2f}")
    print(f"Peak GPU mem (MB): {peak_mem_mb:.2f}")


def compute_score(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> float:
    if metric == "acc":
        return accuracy(logits, labels)
    from sklearn.metrics import roc_auc_score
    y = labels.detach().cpu().numpy()
    prob = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
    if prob.shape[1] == 2:
        return float(roc_auc_score(y, prob[:, 1]))  # score of the positive class
    return float(roc_auc_score(y, prob, multi_class="ovr"))


# -----------------------------
# Graph data container
# -----------------------------

@dataclass
class GraphData:
    adj: sp.csr_matrix
    row_adj: sp.csr_matrix
    feat: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict
    label_mapping: Optional[Dict]

    @property
    def num_nodes(self) -> int:
        return self.feat.shape[0]

    @property
    def num_features(self) -> int:
        return self.feat.shape[1]

    @property
    def num_classes(self) -> int:
        valid = self.labels[self.labels >= 0]
        return int(valid.max()) + 1 if valid.size else 0


# -----------------------------
# Dataset loading
# -----------------------------

def load_graph_dataset(data_root: str | Path, resplit_622: bool, save_splits: bool, seed: int) -> GraphData:
    data_root = Path(data_root)
    adj = make_undirected_binary(load_sparse_npz(data_root / "adj.npz"))
    feat = np.load(data_root / "feat.npy")
    labels = np.load(data_root / "label.npy")
    feat = np.asarray(feat, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if feat.shape[0] != adj.shape[0] or labels.shape[0] != adj.shape[0]:
        raise ValueError("Mismatch among adj/feat/label sizes")

    metadata = {}
    meta_path = data_root / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    label_mapping = None
    lm_path = data_root / "label_mapping.json"
    if lm_path.exists():
        with open(lm_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

    train_idx, val_idx, test_idx = load_fixed_split(data_root, labels)

    row_adj = row_normalize(adj)
    return GraphData(adj=adj, row_adj=row_adj, feat=feat, labels=labels,
                     train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                     metadata=metadata, label_mapping=label_mapping)


# -----------------------------
# Feature KNN cache
# -----------------------------

class KNNSearch:
    def __init__(self, feat: np.ndarray, topk: int = 64, exact_max_nodes: int = 50000,
                 candidate_pool: int = 4096, seed: int = 0):
        self.topk = topk
        self.exact_max_nodes = exact_max_nodes
        self.candidate_pool = candidate_pool
        self.rng = np.random.default_rng(seed)
        x = torch.from_numpy(feat.astype(np.float32))
        self.x = F.normalize(x, dim=1)
        self.n = self.x.shape[0]
        self._indices: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None
        if self.n <= self.exact_max_nodes:
            self._precompute_exact()

    def _precompute_exact(self) -> None:
        x = self.x
        n = x.shape[0]
        chunk = 1024 if n <= 20000 else 512
        idx_list = []
        score_list = []
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            sim = x[s:e] @ x.T
            row_ids = torch.arange(s, e)
            sim[torch.arange(e - s), row_ids] = -1e9
            vals, inds = torch.topk(sim, k=min(self.topk, n - 1), dim=1)
            idx_list.append(inds.cpu().numpy())
            score_list.append(vals.cpu().numpy())
        self._indices = np.concatenate(idx_list, axis=0)
        self._scores = np.concatenate(score_list, axis=0)

    def query(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._indices is not None and self._scores is not None:
            idx = self._indices[node]
            score = self._scores[node]
        else:
            n = self.n
            pool = min(self.candidate_pool, max(64, n - 1))
            cand = self.rng.choice(n, size=pool, replace=False)
            cand = cand[cand != node]
            if cand.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            sims = (self.x[node:node + 1] @ self.x[cand].T).squeeze(0)
            k = min(self.topk, cand.size)
            vals, pos = torch.topk(sims, k=k)
            idx = cand[pos.cpu().numpy()]
            score = vals.cpu().numpy()
        score = np.maximum(score, 0.0)
        denom = float(score.sum()) + 1e-12
        return idx.astype(np.int64), (score / denom).astype(np.float32)


# -----------------------------
# Graph coarsening (graph-aware BFS clustering)
# -----------------------------

class GraphCoarsener:
    def __init__(self, adj: sp.csr_matrix, feat: np.ndarray, rate: float = 0.1, seed: int = 0):
        self.adj = adj
        self.feat = feat
        self.rate = rate
        self.seed = seed
        self.num_nodes = adj.shape[0]
        self.num_clusters = max(1, int(round(self.num_nodes * rate)))
        self.node2cluster, self.cluster_members = self._multi_source_bfs_partition()
        self.cluster_feat = self._build_cluster_feat()
        self.cluster_adj = self._build_cluster_adj()
        self.cluster_row_adj = row_normalize(self.cluster_adj)

    def _multi_source_bfs_partition(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        n = self.num_nodes
        deg = np.asarray(self.adj.sum(1)).reshape(-1)
        order = np.argsort(-deg)
        rng = np.random.default_rng(self.seed)
        seeds = []
        taken = np.zeros(n, dtype=bool)
        for u in order:
            if len(seeds) >= self.num_clusters:
                break
            if not taken[u]:
                seeds.append(int(u))
                taken[u] = True
                nbrs = self.adj.indices[self.adj.indptr[u]:self.adj.indptr[u + 1]]
                taken[nbrs] = True
        if len(seeds) < self.num_clusters:
            remaining = np.setdiff1d(np.arange(n), np.asarray(seeds, dtype=np.int64), assume_unique=False)
            rng.shuffle(remaining)
            need = self.num_clusters - len(seeds)
            seeds.extend(remaining[:need].tolist())
        seeds = np.asarray(seeds[:self.num_clusters], dtype=np.int64)

        node2cluster = np.full(n, -1, dtype=np.int64)
        q = deque()
        for cid, s in enumerate(seeds):
            node2cluster[s] = cid
            q.append(int(s))
        while q:
            u = q.popleft()
            cid = node2cluster[u]
            nbrs = self.adj.indices[self.adj.indptr[u]:self.adj.indptr[u + 1]]
            for v in nbrs:
                if node2cluster[v] == -1:
                    node2cluster[v] = cid
                    q.append(int(v))
        unassigned = np.where(node2cluster == -1)[0]
        if unassigned.size > 0:
            node2cluster[unassigned] = rng.integers(0, self.num_clusters, size=unassigned.size)
        members = [np.where(node2cluster == i)[0].astype(np.int64) for i in range(self.num_clusters)]
        return node2cluster, members

    def _build_cluster_feat(self) -> np.ndarray:
        feat = np.zeros((self.num_clusters, self.feat.shape[1]), dtype=np.float32)
        for cid, members in enumerate(self.cluster_members):
            feat[cid] = self.feat[members].mean(axis=0)
        return feat

    def _build_cluster_adj(self) -> sp.csr_matrix:
        row, col = self.adj.nonzero()
        ci = self.node2cluster[row]
        cj = self.node2cluster[col]
        mask = ci != cj
        ci, cj = ci[mask], cj[mask]
        data = np.ones_like(ci, dtype=np.float32)
        coarse = sp.coo_matrix((data, (ci, cj)), shape=(self.num_clusters, self.num_clusters))
        coarse = coarse.maximum(coarse.T)
        coarse.data[:] = 1.0
        coarse.setdiag(0)
        coarse.eliminate_zeros()
        return coarse.tocsr()


# -----------------------------
# Adaptive node sampler
# -----------------------------

class AdaptiveSampler:
    def __init__(self, graph: GraphData, coarsener: GraphCoarsener, knn: KNNSearch,
                 num_sampled_nodes: int = 20, num_sampled_supernodes: int = 4,
                 ppr_alpha: float = 0.15, ppr_steps: int = 8, ppr_topk: int = 128,
                 p_min: float = 0.05, eta: float = 1.0, seed: int = 0):
        self.graph = graph
        self.coarsener = coarsener
        self.knn = knn
        self.num_sampled_nodes = num_sampled_nodes
        self.num_sampled_supernodes = num_sampled_supernodes
        self.ppr_alpha = ppr_alpha
        self.ppr_steps = ppr_steps
        self.ppr_topk = ppr_topk
        self.p_min = p_min
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.weights = np.ones(4, dtype=np.float64)
        self._cache_1hop: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_2hop: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_ppr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get_mixture(self) -> np.ndarray:
        return (1.0 - 4 * self.p_min) * self.weights / self.weights.sum() + self.p_min

    def update(self, rewards: np.ndarray) -> None:
        rewards = np.asarray(rewards, dtype=np.float64)
        if not np.all(np.isfinite(rewards)):
            return
        scale = np.max(np.abs(rewards)) + 1e-12
        rewards = rewards / scale
        self.weights *= np.exp(self.eta * rewards)
        self.weights = np.maximum(self.weights, 1e-8)

    def one_hop(self, u: int) -> Tuple[np.ndarray, np.ndarray]:
        if u in self._cache_1hop:
            return self._cache_1hop[u]
        nbrs = self.graph.adj.indices[self.graph.adj.indptr[u]:self.graph.adj.indptr[u + 1]].astype(np.int64)
        nbrs = nbrs[nbrs != u]
        if nbrs.size == 0:
            out = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        else:
            prob = np.full(nbrs.size, 1.0 / nbrs.size, dtype=np.float32)
            out = (nbrs, prob)
        self._cache_1hop[u] = out
        return out

    def two_hop(self, u: int) -> Tuple[np.ndarray, np.ndarray]:
        if u in self._cache_2hop:
            return self._cache_2hop[u]
        one, _ = self.one_hop(u)
        counter: Dict[int, float] = defaultdict(float)
        for v in one:
            nbrs = self.graph.adj.indices[self.graph.adj.indptr[v]:self.graph.adj.indptr[v + 1]]
            deg_v = max(1, len(nbrs))
            for w in nbrs:
                if w != u:
                    counter[int(w)] += 1.0 / deg_v
        if not counter:
            out = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        else:
            ids = np.fromiter(counter.keys(), dtype=np.int64)
            val = np.fromiter(counter.values(), dtype=np.float32)
            val = np.maximum(val, 0)
            val = val / (val.sum() + 1e-12)
            out = (ids, val)
        self._cache_2hop[u] = out
        return out

    def ppr(self, u: int) -> Tuple[np.ndarray, np.ndarray]:
        if u in self._cache_ppr:
            return self._cache_ppr[u]
        cur = {int(u): 1.0}
        scores = defaultdict(float)
        alpha = self.ppr_alpha
        for _ in range(self.ppr_steps):
            nxt = defaultdict(float)
            for v, mass in cur.items():
                scores[v] += alpha * mass
                nbrs = self.graph.adj.indices[self.graph.adj.indptr[v]:self.graph.adj.indptr[v + 1]]
                deg_v = max(1, len(nbrs))
                share = (1.0 - alpha) * mass / deg_v
                for w in nbrs:
                    nxt[int(w)] += share
            if len(nxt) > self.ppr_topk:
                top = sorted(nxt.items(), key=lambda kv: kv[1], reverse=True)[:self.ppr_topk]
                cur = dict(top)
            else:
                cur = nxt
        if u in scores:
            del scores[u]
        if not scores:
            out = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        else:
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:self.ppr_topk]
            ids = np.asarray([k for k, _ in top], dtype=np.int64)
            val = np.asarray([v for _, v in top], dtype=np.float32)
            val = np.maximum(val, 0)
            val = val / (val.sum() + 1e-12)
            out = (ids, val)
        self._cache_ppr[u] = out
        return out

    def knn_query(self, u: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.knn.query(u)

    def heuristic_distributions(self, u: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.one_hop(u), self.two_hop(u), self.ppr(u), self.knn_query(u)]

    def sample_nodes(self, u: int) -> Dict[str, np.ndarray]:
        mix = self.get_mixture()
        heur = self.heuristic_distributions(u)
        union = []
        q_maps: List[Dict[int, float]] = []
        for ids, probs in heur:
            q = {int(i): float(p) for i, p in zip(ids, probs) if int(i) != u}
            q_maps.append(q)
            union.extend(list(q.keys()))
        if not union:
            return {
                "sampled": np.empty(0, dtype=np.int64),
                "psi": np.empty(0, dtype=np.float32),
                "q": np.empty((4, 0), dtype=np.float32),
                "mix": mix.astype(np.float32),
            }
        union = np.asarray(sorted(set(union)), dtype=np.int64)
        q = np.zeros((4, union.size), dtype=np.float32)
        for k, qk in enumerate(q_maps):
            for j, node in enumerate(union):
                q[k, j] = qk.get(int(node), 0.0)
        psi = mix.astype(np.float32) @ q
        psi = np.maximum(psi, 1e-12)
        psi = psi / psi.sum()
        n = min(self.num_sampled_nodes, union.size)
        chosen_pos = self.rng.choice(union.size, size=n, replace=False, p=psi)
        sampled = union[chosen_pos]
        return {
            "sampled": sampled.astype(np.int64),
            "psi": psi[chosen_pos].astype(np.float32),
            "q": q[:, chosen_pos].astype(np.float32),
            "mix": mix.astype(np.float32),
        }

    def sample_supernodes(self, u: int) -> np.ndarray:
        cid = int(self.coarsener.node2cluster[u])
        row = self.coarsener.cluster_row_adj.indices[
            self.coarsener.cluster_row_adj.indptr[cid]:self.coarsener.cluster_row_adj.indptr[cid + 1]
        ].astype(np.int64)
        if row.size == 0:
            cand = np.arange(self.coarsener.num_clusters, dtype=np.int64)
            cand = cand[cand != cid]
            if cand.size == 0:
                return np.asarray([cid], dtype=np.int64)
            self.rng.shuffle(cand)
            return cand[:min(self.num_sampled_supernodes, cand.size)]
        deg = np.asarray(self.coarsener.cluster_adj.sum(1)).reshape(-1)
        prob = deg[row].astype(np.float64)
        prob = prob / (prob.sum() + 1e-12)
        n = min(self.num_sampled_supernodes, row.size)
        pos = self.rng.choice(row.size, size=n, replace=False, p=prob)
        return row[pos].astype(np.int64)


# -----------------------------
# Sequence dataset
# -----------------------------

class CenterNodeDataset(Dataset):
    def __init__(self, graph: GraphData, centers: np.ndarray, sampler: AdaptiveSampler,
                 coarsener: GraphCoarsener, num_global_nodes: int = 1):
        self.graph = graph
        self.centers = centers.astype(np.int64)
        self.sampler = sampler
        self.coarsener = coarsener
        self.num_global_nodes = num_global_nodes
        self.row_adj = graph.row_adj
        self.adj = graph.adj

    def __len__(self) -> int:
        return len(self.centers)

    def _pair_features_graph(self, node_ids: List[int]) -> np.ndarray:
        n = len(node_ids)
        feats = np.zeros((n, n, 6), dtype=np.float32)
        if n == 0:
            return feats
        # Induced adjacency among actual graph nodes only.
        sub_idx = np.asarray(node_ids, dtype=np.int64)
        sub = self.adj[sub_idx][:, sub_idx].astype(np.float32)
        sub2 = (sub @ sub).astype(np.float32)
        sub_bin = (sub.toarray() > 0).astype(np.float32)
        sub2_bin = (sub2.toarray() > 0).astype(np.float32)
        np.fill_diagonal(sub2_bin, 0.0)
        feats[:, :, 0] = np.eye(n, dtype=np.float32)
        feats[:, :, 1] = sub_bin
        feats[:, :, 2] = sub2_bin
        # Same coarse cluster.
        c = self.coarsener.node2cluster[sub_idx]
        feats[:, :, 3] = (c[:, None] == c[None, :]).astype(np.float32)
        # Cosine similarity.
        x = self.graph.feat[sub_idx]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        feats[:, :, 4] = x @ x.T
        # Token-type equality placeholder for real graph nodes.
        feats[:, :, 5] = 1.0
        return feats

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        center = int(self.centers[idx])
        sampled_info = self.sampler.sample_nodes(center)
        fine_nodes = sampled_info["sampled"]
        super_nodes = self.sampler.sample_supernodes(center)
        center_feat = self.graph.feat[center:center + 1]
        fine_feat = self.graph.feat[fine_nodes] if fine_nodes.size else np.zeros((0, self.graph.num_features), dtype=np.float32)
        super_feat = self.coarsener.cluster_feat[super_nodes] if super_nodes.size else np.zeros((0, self.graph.num_features), dtype=np.float32)
        global_feat = np.zeros((self.num_global_nodes, self.graph.num_features), dtype=np.float32)

        x = np.concatenate([center_feat, fine_feat, super_feat, global_feat], axis=0).astype(np.float32)
        seq_len = x.shape[0]
        token_type = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.full(len(fine_nodes), 1, dtype=np.int64),
            np.full(len(super_nodes), 2, dtype=np.int64),
            np.full(self.num_global_nodes, 3, dtype=np.int64),
        ], axis=0)

        graph_node_ids = [center] + fine_nodes.tolist()
        graph_bias = self._pair_features_graph(graph_node_ids)
        attn_bias = np.zeros((seq_len, seq_len, 6), dtype=np.float32)
        g_len = 1 + len(fine_nodes)
        attn_bias[:g_len, :g_len, :] = graph_bias

        # Center-to-super and super-to-center structural link.
        for i, sid in enumerate(super_nodes, start=g_len):
            members = self.coarsener.cluster_members[int(sid)]
            if center in members:
                attn_bias[0, i, 3] = 1.0
                attn_bias[i, 0, 3] = 1.0
            if fine_nodes.size:
                member_set = set(int(m) for m in members)
                for j, nid in enumerate(fine_nodes, start=1):
                    if int(nid) in member_set:
                        attn_bias[j, i, 3] = 1.0
                        attn_bias[i, j, 3] = 1.0
        # Global token indicator.
        if self.num_global_nodes > 0:
            gs = seq_len - self.num_global_nodes
            attn_bias[gs:, :, 5] = 1.0
            attn_bias[:, gs:, 5] = 1.0

        # Coarse-coarse connectivity.
        if len(super_nodes) > 0:
            coarse_sub = self.coarsener.cluster_adj[super_nodes][:, super_nodes].toarray().astype(np.float32)
            start = g_len
            end = g_len + len(super_nodes)
            attn_bias[start:end, start:end, 1] = coarse_sub
            np.fill_diagonal(attn_bias[start:end, start:end, 0], 1.0)
            attn_bias[start:end, start:end, 2] = ((coarse_sub @ coarse_sub) > 0).astype(np.float32)
            attn_bias[start:end, start:end, 5] = 1.0

        return {
            "x": x,
            "token_type": token_type,
            "attn_bias": attn_bias,
            "label": np.asarray(self.graph.labels[center], dtype=np.int64),
            "center": np.asarray(center, dtype=np.int64),
            "fine_nodes": fine_nodes.astype(np.int64),
            "psi": sampled_info["psi"],
            "q": sampled_info["q"],
        }


def collate_sequences(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    bsz = len(batch)
    x = np.zeros((bsz, max_len, feat_dim), dtype=np.float32)
    token_type = np.zeros((bsz, max_len), dtype=np.int64)
    attn_bias = np.zeros((bsz, max_len, max_len, 6), dtype=np.float32)
    mask = np.zeros((bsz, max_len), dtype=np.bool_)
    labels = np.zeros((bsz,), dtype=np.int64)
    centers = np.zeros((bsz,), dtype=np.int64)

    max_fine = max(item["fine_nodes"].shape[0] for item in batch)
    fine_nodes = np.full((bsz, max_fine), -1, dtype=np.int64)
    psi = np.zeros((bsz, max_fine), dtype=np.float32)
    q = np.zeros((bsz, 4, max_fine), dtype=np.float32)
    fine_mask = np.zeros((bsz, max_fine), dtype=np.bool_)

    for i, item in enumerate(batch):
        l = item["x"].shape[0]
        x[i, :l] = item["x"]
        token_type[i, :l] = item["token_type"]
        attn_bias[i, :l, :l] = item["attn_bias"]
        mask[i, :l] = True
        labels[i] = item["label"]
        centers[i] = item["center"]
        nf = item["fine_nodes"].shape[0]
        if nf > 0:
            fine_nodes[i, :nf] = item["fine_nodes"]
            psi[i, :nf] = item["psi"]
            q[i, :, :nf] = item["q"]
            fine_mask[i, :nf] = True

    return {
        "x": torch.from_numpy(x),
        "token_type": torch.from_numpy(token_type),
        "attn_bias": torch.from_numpy(attn_bias),
        "mask": torch.from_numpy(mask),
        "labels": torch.from_numpy(labels),
        "centers": torch.from_numpy(centers),
        "fine_nodes": torch.from_numpy(fine_nodes),
        "psi": torch.from_numpy(psi),
        "q": torch.from_numpy(q),
        "fine_mask": torch.from_numpy(fine_mask),
    }


# -----------------------------
# Model
# -----------------------------

class BiasedMultiheadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_bias_dim: int = 6, attn_dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bias_proj = nn.Linear(attn_bias_dim, num_heads)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask: torch.Tensor,
                need_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, L, D], attn_bias: [B, L, L, C], mask: [B, L]
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        bias = self.bias_proj(attn_bias).permute(0, 3, 1, 2)
        scores = scores + bias
        # key mask
        key_mask = (~mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)
        out = self.out_proj(out)
        mean_attn = attn.mean(dim=1) if need_attn else None
        return out, mean_attn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int,
                 dropout: float = 0.1, attn_bias_dim: int = 6):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = BiasedMultiheadSelfAttention(hidden_dim, num_heads, attn_bias_dim, attn_dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, mask: torch.Tensor,
                need_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, attn = self.attn(self.norm1(x), attn_bias, mask, need_attn=need_attn)
        x = x + self.drop1(h)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, attn


class ANSGTModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 dim_feedforward: int, dropout: float, num_classes: int,
                 num_global_nodes: int = 1, attn_bias_dim: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.type_emb = nn.Embedding(4, hidden_dim)
        self.global_tokens = nn.Parameter(torch.randn(max(1, num_global_nodes), hidden_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dim_feedforward, dropout=dropout, attn_bias_dim=attn_bias_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.num_global_nodes = num_global_nodes

    def forward(self, batch: Dict[str, torch.Tensor], return_score: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = batch["x"]
        token_type = batch["token_type"]
        attn_bias = batch["attn_bias"]
        mask = batch["mask"]
        h = self.input_proj(x) + self.type_emb(token_type)
        if self.num_global_nodes > 0:
            g = self.global_tokens[:self.num_global_nodes]
            g_start = h.shape[1] - self.num_global_nodes
            h[:, g_start:, :] = h[:, g_start:, :] + g.unsqueeze(0)
        h = self.input_dropout(h)
        attn_score = None
        for i, layer in enumerate(self.layers):
            h, attn = layer(h, attn_bias, mask, need_attn=return_score and i == len(self.layers) - 1)
            if attn is not None:
                attn_score = attn
        h = self.norm(h)
        center_repr = h[:, 0, :]
        logits = self.cls(center_repr)
        if not return_score:
            return logits, None
        # Fine-node significance from center token attention in the last layer.
        fine_mask = batch["fine_mask"].to(h.device)
        max_fine = fine_mask.shape[1]
        if attn_score is None or max_fine == 0:
            return logits, torch.zeros((h.shape[0], max_fine), device=h.device)
        center_to_all = attn_score[:, 0, 1:1 + max_fine]
        vmag = h[:, 1:1 + max_fine, :].norm(dim=-1)
        sig = center_to_all * vmag
        sig = sig * fine_mask.float()
        denom = sig.sum(dim=-1, keepdim=True) + 1e-12
        sig = sig / denom
        return logits, sig


# -----------------------------
# Training / evaluation
# -----------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model: ANSGTModel, loader: DataLoader, device: torch.device,
             metric: str = "acc") -> Tuple[float, float]:
    """Returns (score, loss); score is accuracy or ROC-AUC (see resolve_metric)."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_logits, all_labels = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits, _ = model(batch, return_score=False)
        loss = F.cross_entropy(logits, batch["labels"])
        total_loss += float(loss.item()) * logits.shape[0]
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        total += int(logits.shape[0])
    if total == 0:
        return 0.0, 0.0
    return compute_score(torch.cat(all_logits), torch.cat(all_labels), metric), total_loss / total


@torch.no_grad()
def estimate_rewards(model: ANSGTModel, loader: DataLoader, device: torch.device,
                     max_batches: int = 16) -> np.ndarray:
    model.eval()
    rewards = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        _, sig = model(batch, return_score=True)
        q = batch["q"].float()          # [B, 4, F]
        psi = batch["psi"].float()      # [B, F]
        fine_mask = batch["fine_mask"].float()
        denom = psi.unsqueeze(1) + 1e-12
        contrib = (sig.unsqueeze(1) * q / denom) * fine_mask.unsqueeze(1)
        reward = contrib.sum(dim=-1).mean(dim=0)
        rewards.append(reward.detach().cpu().numpy())
    if not rewards:
        return np.zeros(4, dtype=np.float64)
    return np.mean(np.stack(rewards, axis=0), axis=0)


def train_one_epoch(model: ANSGTModel, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_grad_norm: float = 5.0) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch, return_score=False)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += float(loss.item()) * logits.shape[0]
        total += int(logits.shape[0])
    return total_loss / max(1, total)


# -----------------------------
# Experiment driver
# -----------------------------

def make_loaders(graph: GraphData, sampler: AdaptiveSampler, coarsener: GraphCoarsener,
                 batch_size: int, num_workers: int, num_global_nodes: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CenterNodeDataset(graph, graph.train_idx, sampler, coarsener, num_global_nodes)
    val_ds = CenterNodeDataset(graph, graph.val_idx, sampler, coarsener, num_global_nodes)
    test_ds = CenterNodeDataset(graph, graph.test_idx, sampler, coarsener, num_global_nodes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_sequences, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_sequences, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_sequences, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_once(args: argparse.Namespace, run_seed: int) -> Dict:
    set_seed(run_seed)
    graph = load_graph_dataset(args.data_root, args.resplit_622, args.save_splits and args.run_id == 0, run_seed)

    # one-off preprocessing of ANS-GT: kNN index, graph coarsening, sampler and first loaders
    pre_t0 = time.perf_counter()
    knn = KNNSearch(graph.feat, topk=args.knn_topk, exact_max_nodes=args.knn_exact_max_nodes,
                    candidate_pool=args.knn_candidate_pool, seed=run_seed)
    coarsener = GraphCoarsener(graph.adj, graph.feat, rate=args.coarsen_rate, seed=run_seed)
    sampler = AdaptiveSampler(graph, coarsener, knn,
                              num_sampled_nodes=args.num_sampled_nodes,
                              num_sampled_supernodes=args.num_sampled_supernodes,
                              ppr_alpha=args.ppr_alpha,
                              ppr_steps=args.ppr_steps,
                              ppr_topk=args.ppr_topk,
                              p_min=args.sampler_pmin,
                              eta=args.sampler_eta,
                              seed=run_seed)

    train_loader, val_loader, test_loader = make_loaders(
        graph, sampler, coarsener, args.batch_size, args.num_workers, args.num_global_nodes
    )
    pre_time = time.perf_counter() - pre_t0

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model = ANSGTModel(
        input_dim=graph.num_features,
        hidden_dim=args.hid_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=graph.num_classes,
        num_global_nodes=args.num_global_nodes,
    ).to(device)
    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {
        "epoch": 0,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "train_peak_mem_mb": 0.0,
        "eval_peak_mem_mb": 0.0,
        "update_peak_mem_mb": 0.0,
        "total_peak_mem_mb": 0.0,
        "train_epoch_time_sec": 0.0,
        "eval_epoch_time_sec": 0.0,
        "sampler_update_time_sec": 0.0,
        "total_epoch_time_sec": 0.0,
    }
    best_state = None
    epochs_without_improve = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        timeout_dir = os.environ.get('SCOPE_GT_RUN_MANIFEST_DIR', str(Path(args.out_dir) / 'provenance'))
        with EpochDeadline(args.epoch_timeout_seconds, seed=run_seed, epoch=epoch,
                           record_dir=timeout_dir):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, args.max_grad_norm)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        train_time = time.perf_counter() - t0
        train_peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

        # Evaluate only every args.eval_period epochs by default.
        # The last epoch is always evaluated, so a valid best checkpoint can still be selected.
        do_eval = (epoch % args.eval_period == 0) or (epoch == args.epochs)
        val_acc, val_loss = None, None
        test_acc, test_loss = None, None
        eval_time = 0.0
        val_time = 0.0
        eval_peak_mem = 0.0

        if do_eval:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            eval_t0 = time.perf_counter()
            # val_acc / test_acc hold args.metric (accuracy, or ROC-AUC on the AUC datasets)
            val_acc, val_loss = evaluate(model, val_loader, device, args.metric)
            val_time = time.perf_counter() - eval_t0
            test_acc, test_loss = evaluate(model, test_loader, device, args.metric)
            eval_time = time.perf_counter() - eval_t0
            eval_peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

        update_time = 0.0
        update_peak_mem = 0.0
        if epoch % args.sampler_update_period == 0:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            update_t0 = time.perf_counter()
            # Fair baseline protocol: adaptive sampling rewards are estimated from
            # training centers only; validation/test centers are reserved for model
            # selection and final reporting.
            rewards = estimate_rewards(model, train_loader, device, max_batches=args.reward_eval_batches)
            sampler.update(rewards)
            train_loader, val_loader, test_loader = make_loaders(
                graph, sampler, coarsener, args.batch_size, args.num_workers, args.num_global_nodes
            )
            update_time = time.perf_counter() - update_t0
            update_peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

        total_time = train_time + eval_time + update_time
        total_peak_mem = max(train_peak_mem, eval_peak_mem, update_peak_mem)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "did_eval": do_eval,
            "train_peak_mem_mb": train_peak_mem,
            "eval_peak_mem_mb": eval_peak_mem,
            "update_peak_mem_mb": update_peak_mem,
            "total_peak_mem_mb": total_peak_mem,
            "train_epoch_time_sec": train_time,
            "eval_epoch_time_sec": eval_time,
            "val_epoch_time_sec": val_time,
            "sampler_update_time_sec": update_time,
            "total_epoch_time_sec": total_time,
            "sampler_mix": sampler.get_mixture().tolist(),
        })

        # Best checkpoint selection and early stopping are updated only on evaluation epochs.
        # Therefore, patience now means "number of consecutive evaluations without improvement".
        if do_eval:
            if val_acc > best["val_acc"]:
                best.update({
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "train_peak_mem_mb": train_peak_mem,
                    "eval_peak_mem_mb": eval_peak_mem,
                    "update_peak_mem_mb": update_peak_mem,
                    "total_peak_mem_mb": total_peak_mem,
                    "train_epoch_time_sec": train_time,
                    "eval_epoch_time_sec": eval_time,
                    "sampler_update_time_sec": update_time,
                    "total_epoch_time_sec": total_time,
                })
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_acc, _ = evaluate(model, val_loader, device, args.metric)
    final_test_acc, _ = evaluate(model, test_loader, device, args.metric)

    # time to the best model: training, validation and sampler updates of epochs 1..best_epoch
    # (the test evaluation done on every evaluation epoch is excluded)
    time_to_best = float(sum(h["train_epoch_time_sec"] + h["val_epoch_time_sec"] + h["sampler_update_time_sec"]
                             for h in history if h["epoch"] <= best["epoch"]))

    result = {
        "seed": run_seed,
        "metric": args.metric,
        "params": num_params,
        "pre_time_sec": pre_time,
        "train_time_per_epoch_sec": float(np.mean([h["train_epoch_time_sec"] for h in history])) if history else 0.0,
        "time_to_best_sec": time_to_best,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "test_acc_at_best_val": best["test_acc"],
        "reloaded_val_acc": final_val_acc,
        "reloaded_test_acc": final_test_acc,
        "avg_epoch_time_sec": float(np.mean([h["total_epoch_time_sec"] for h in history])) if history else 0.0,
        "max_peak_mem_mb": float(max((h["total_peak_mem_mb"] for h in history), default=0.0)),
        "best_epoch_train_peak_mem_mb": best["train_peak_mem_mb"],
        "best_epoch_eval_peak_mem_mb": best["eval_peak_mem_mb"],
        "best_epoch_update_peak_mem_mb": best["update_peak_mem_mb"],
        "best_epoch_total_peak_mem_mb": best["total_peak_mem_mb"],
        "best_epoch_train_time_sec": best["train_epoch_time_sec"],
        "best_epoch_eval_time_sec": best["eval_epoch_time_sec"],
        "best_epoch_sampler_update_time_sec": best["sampler_update_time_sec"],
        "best_epoch_total_time_sec": best["total_epoch_time_sec"],
        "num_train": int(len(graph.train_idx)),
        "num_val": int(len(graph.val_idx)),
        "num_test": int(len(graph.test_idx)),
        "num_nodes": int(graph.num_nodes),
        "num_features": int(graph.num_features),
        "num_classes": int(graph.num_classes),
        "history": history,
    }
    return result


def summarize_results(results: List[Dict]) -> Dict:
    test_accs = np.asarray([r["test_acc_at_best_val"] for r in results], dtype=np.float64)
    best_epochs = np.asarray([r["best_epoch"] for r in results], dtype=np.float64)
    peak_mems = np.asarray([r["max_peak_mem_mb"] for r in results], dtype=np.float64)
    epoch_times = np.asarray([r["avg_epoch_time_sec"] for r in results], dtype=np.float64)
    return {
        "runs": len(results),
        "metric": results[0]["metric"] if results else "acc",  # test_acc_* hold this metric
        "test_acc_mean": float(test_accs.mean()) if test_accs.size else 0.0,
        "test_acc_std": float(test_accs.std(ddof=1)) if test_accs.size > 1 else 0.0,
        "best_epoch_mean": float(best_epochs.mean()) if best_epochs.size else 0.0,
        "best_epoch_std": float(best_epochs.std(ddof=1)) if best_epochs.size > 1 else 0.0,
        "max_peak_mem_mb_mean": float(peak_mems.mean()) if peak_mems.size else 0.0,
        "max_peak_mem_mb_std": float(peak_mems.std(ddof=1)) if peak_mems.size > 1 else 0.0,
        "avg_epoch_time_sec_mean": float(epoch_times.mean()) if epoch_times.size else 0.0,
        "avg_epoch_time_sec_std": float(epoch_times.std(ddof=1)) if epoch_times.size > 1 else 0.0,
        "per_run": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANS-GT adaptation for adj.npz/feat.npy/label.npy datasets")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--dataset", type=str, default=None, help="dataset name (default: basename of --data-root)")
    p.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "auc", "roc_auc"],
                   help="auto: ROC-AUC for %s, accuracy otherwise" % ", ".join(AUC_DATASETS))
    p.add_argument("--out-dir", type=str, default="./results_ansgt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epoch-timeout-seconds", type=float, default=43200,
                   help="classification training-epoch limit; default 12 hours, 0 disables")
    p.add_argument("--resplit-622", action="store_true", default=False, help="Deprecated and ignored. This script always reads train.npy/val.npy/test.npy.")
    p.add_argument("--keep-existing-split", action="store_true", help="Deprecated. Fixed split files are always used.")
    p.add_argument("--save-splits", action="store_true", help="Deprecated and ignored.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)

    # Requested project-aligned hyperparameters.
    p.add_argument("--hid-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--eval-period", type=int, default=5, help="Evaluate validation/test once every N epochs. Default: 5.")
    p.add_argument("--patience", type=int, default=100, help="Early stopping patience counted by evaluation rounds, not raw epochs.")
    p.add_argument("--max-grad-norm", type=float, default=5.0)

    # ANS-GT-style sampler / hierarchy settings.
    p.add_argument("--num-sampled-nodes", type=int, default=20)
    p.add_argument("--num-sampled-supernodes", type=int, default=4)
    p.add_argument("--num-global-nodes", type=int, default=1)
    p.add_argument("--coarsen-rate", type=float, default=0.1)
    p.add_argument("--sampler-update-period", type=int, default=20)
    p.add_argument("--sampler-pmin", type=float, default=0.05)
    p.add_argument("--sampler-eta", type=float, default=1.0)
    p.add_argument("--reward-eval-batches", type=int, default=16)
    p.add_argument("--ppr-alpha", type=float, default=0.15)
    p.add_argument("--ppr-steps", type=int, default=8)
    p.add_argument("--ppr-topk", type=int, default=128)
    p.add_argument("--knn-topk", type=int, default=64)
    p.add_argument("--knn-exact-max-nodes", type=int, default=50000)
    p.add_argument("--knn-candidate-pool", type=int, default=4096)
    args = p.parse_args()
    if not math.isfinite(args.epoch_timeout_seconds) or args.epoch_timeout_seconds < 0:
        p.error('--epoch-timeout-seconds must be finite and nonnegative')
    if args.dataset is None:
        args.dataset = Path(args.data_root).name
    args.metric = resolve_metric(args.metric, args.dataset)
    return args


def main() -> None:
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    name = metric_name(args.metric)
    all_results = []
    print(f"==== ANS-GT adaptation ==== dataset={args.dataset} metric={name}")
    for run_id in range(args.runs):
        args.run_id = run_id
        run_seed = args.base_seed + run_id
        print(f"\n[Run {run_id + 1}/{args.runs}] seed={run_seed}")
        result = run_once(args, run_seed)
        all_results.append(result)
        print(
            f"best_epoch={result['best_epoch']}  "
            f"val_{name}={result['best_val_acc']:.4f}  "
            f"test_{name}_at_best={result['test_acc_at_best_val']:.4f}  "
            f"peak_mem_mb={result['best_epoch_total_peak_mem_mb']:.2f}  "
            f"epoch_time_sec={result['best_epoch_total_time_sec']:.4f}  "
            f"time_to_best_sec={result['time_to_best_sec']:.2f}"
        )

    summary = summarize_results(all_results)
    peak_mem = max((r["max_peak_mem_mb"] for r in all_results), default=0.0)

    print("\n==== Summary ====")
    print(f"Test {name}: {summary['test_acc_mean'] * 100:.2f} \u00b1 {summary['test_acc_std'] * 100:.2f}")
    print(f"Tepoch (s): {summary['avg_epoch_time_sec_mean']:.2f}")
    print(f"Mem. (MB): {peak_mem:.2f}")
    print_efficiency(all_results[0]["params"], [r["train_time_per_epoch_sec"] for r in all_results],
                     [r["time_to_best_sec"] for r in all_results], peak_mem,
                     pre_s=[r["pre_time_sec"] for r in all_results], best_epoch=[r["best_epoch"] for r in all_results])


if __name__ == "__main__":
    main()
