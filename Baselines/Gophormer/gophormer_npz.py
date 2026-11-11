#!/usr/bin/env python3
"""Gophormer (Zhao et al., arXiv:2110.13094, "Gophormer: Ego-Graph Transformer for Node Classification") on the
project's NPZ datasets.

This file is a dense, paper-guided re-implementation. The author's repository has since been verified at
https://github.com/AndyJZhao/Gophormer; its DGL disjoint implementation is separate and is not bundled as this baseline. The model implemented in this file follows:
  Node2Seq      S ego-graphs per training node and epoch, GraphSAGE-style uniform sampling to depth D (Sec. 3.1),
                plus n_g learnable global nodes shared by all ego-graphs
  PE-MHA        attention score + phi_ij^T b, phi_ij = [A~^0..A~^(M-2)][i, j] and the global-node indicator
                (Eq. 5-7); A~ = Norm(A + I) on the sampled ego-graph, the "reachable probabilities" -> row
                normalisation (--prox-norm row; sym = previous symmetric version)
  layers        post-LN, H^ = LN(PE-MHA(H) + H), H = LN(FFN(H^) + H^) (Eq. 8; --pre-ln = previous pre-LN version),
                read-out = center node, 1-layer MLP classifier (Eq. 9-10)
  loss          mean CE over the S ego-graphs + lambda * consistency ||sharpened mean - y^(s)||^2 (Eq. 11-13)
  inference     multi-sample: the S' predicted distributions of each node are averaged (Eq. 14)
  optimiser     Adam, linear warm-up from 0 to the peak lr, then ReduceLROnPlateau on the validation metric down
                to the end lr 1e-9 (Sec. 4.1.3); warm-up length and decay factor / patience are not given in the
                paper (--warmup-epochs, --lr-decay, --lr-patience)
Shared comparison protocol (../common/protocol.py), as for every other baseline
  data files and splits (common/data.py), runs / seeds, epochs / patience / validation every --eval-every
  epochs, model selection on the validation metric with the test set evaluated once, ROC-AUC on the binary
  imbalanced datasets, efficiency report (params, train time per epoch, time to best, peak GPU memory).
  Validation and test use a fixed sampling seed per run, so every validation check sees the same ego-graphs.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.protocol import (  # noqa: E402
    ProtocolData, RunResult, add_protocol_args, compute_metric, count_parameters, default_out_dir,
    finalize_args, fit, report, resolve_device, set_seed,
)


def dedup_keep_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for x in values:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ----------------------------
# Dataset + sampling
# ----------------------------


class GraphData:
    """Graph of common/data.py (symmetric, no self loops) with per-node neighbour lists for the sampler."""

    def __init__(self, data: ProtocolData):
        self.adj = data.adj
        self.features = data.features
        self.labels = data.labels
        self.num_nodes = data.num_nodes
        self.feat_dim = data.num_features
        self.num_classes = data.num_classes
        indptr, indices = self.adj.indptr, self.adj.indices
        self.neighbors: List[np.ndarray] = [indices[indptr[i]:indptr[i + 1]] for i in range(self.num_nodes)]


class EgoGraphSampler:
    def __init__(self, graph: GraphData, fanouts: Sequence[int], global_nodes: int, prox_order: int,
                 prox_norm: str = "row"):
        self.graph = graph
        self.fanouts = list(fanouts)
        self.depth = len(self.fanouts)
        self.global_nodes = global_nodes
        self.prox_order = prox_order
        self.prox_norm = prox_norm

    def sample_nodes(self, center: int, rng: np.random.RandomState) -> List[int]:
        hops: List[List[int]] = [[int(center)]]
        seen = {int(center)}
        frontier = [int(center)]

        for fanout in self.fanouts:
            sampled: List[int] = []
            for u in frontier:
                nbrs = self.graph.neighbors[u]
                if nbrs.size == 0:
                    continue
                if fanout < 0 or nbrs.size <= fanout:
                    chosen = nbrs.tolist()
                else:
                    perm = rng.choice(nbrs.size, size=fanout, replace=False)
                    chosen = nbrs[perm].tolist()
                sampled.extend(int(v) for v in chosen)
            next_frontier = [v for v in dedup_keep_order(sampled) if v not in seen]
            hops.append(next_frontier)
            seen.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                # Keep hop count aligned but stop expanding.
                break

        ordered_nodes = [int(center)]
        for hop_nodes in hops[1:]:
            ordered_nodes.extend(hop_nodes)
        ordered_nodes = dedup_keep_order(ordered_nodes)
        return ordered_nodes

    def build_single(self, center: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        nodes = self.sample_nodes(center, rng)
        real_n = len(nodes)
        total_n = real_n + self.global_nodes

        feat = np.zeros((total_n, self.graph.feat_dim), dtype=np.float32)
        feat[:real_n] = self.graph.features[nodes]
        mask = np.ones(total_n, dtype=bool)

        if real_n > 0:
            sub_adj = self.graph.adj[nodes][:, nodes].astype(np.float32).toarray()
            sub_adj = (sub_adj != 0).astype(np.float32)
        else:
            sub_adj = np.zeros((0, 0), dtype=np.float32)

        prox_dim = self.prox_order + 2  # A~^0..A~^K + global-indicator (M = K + 2 views)
        phi = np.zeros((total_n, total_n, prox_dim), dtype=np.float32)

        if real_n > 0:
            a_tilde = sub_adj + np.eye(real_n, dtype=np.float32)
            deg = a_tilde.sum(axis=1)
            if self.prox_norm == "row":       # D^-1 (A + I): multi-step reachable probabilities (Eq. 7)
                a_norm = a_tilde / np.maximum(deg, 1e-12)[:, None]
            else:                             # D^-1/2 (A + I) D^-1/2 (previous version of this file)
                deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
                a_norm = deg_inv_sqrt[:, None] * a_tilde * deg_inv_sqrt[None, :]
            cur = np.eye(real_n, dtype=np.float32)
            for order in range(self.prox_order + 1):
                phi[:real_n, :real_n, order] = cur
                cur = cur @ a_norm

        if self.global_nodes > 0:
            phi[:, :, -1] = 0.0
            phi[real_n:, :, -1] = 1.0
            phi[:, real_n:, -1] = 1.0

        return {
            "center": np.int64(center),
            "nodes": np.asarray(nodes, dtype=np.int64),
            "feat": feat,
            "mask": mask,
            "phi": phi,
            "real_n": np.int64(real_n),
        }

    def collate(self, centers: Sequence[int], samples_per_center: int, rng: np.random.RandomState) -> Dict[str, torch.Tensor]:
        graphs = []
        repeated_centers = []
        for c in centers:
            for _ in range(samples_per_center):
                graphs.append(self.build_single(int(c), rng))
                repeated_centers.append(int(c))

        if not graphs:
            raise ValueError("empty collate")

        # Shared global nodes sit at the same padded positions for every sample of the batch:
        # real ego nodes are padded to max_real_n, global nodes occupy [max_real_n, max_real_n + global_nodes).
        max_real_n = max(int(g["real_n"]) for g in graphs)
        max_n = max_real_n + self.global_nodes
        feat_dim = graphs[0]["feat"].shape[1]
        prox_dim = graphs[0]["phi"].shape[2]

        feat = np.zeros((len(graphs), max_n, feat_dim), dtype=np.float32)
        mask = np.zeros((len(graphs), max_n), dtype=bool)
        phi = np.zeros((len(graphs), max_n, max_n, prox_dim), dtype=np.float32)

        global_start = max_real_n
        global_end = max_real_n + self.global_nodes

        for i, g in enumerate(graphs):
            real_n = int(g["real_n"])
            # Real ego nodes keep their original order, with the center node at 0.
            feat[i, :real_n] = g["feat"][:real_n]
            mask[i, :real_n] = True
            phi[i, :real_n, :real_n, :] = g["phi"][:real_n, :real_n, :]

            if self.global_nodes > 0:
                mask[i, global_start:global_end] = True
                # global-node indicator between valid real nodes and global nodes; padding stays masked out
                phi[i, global_start:global_end, :real_n, -1] = 1.0
                phi[i, :real_n, global_start:global_end, -1] = 1.0
                phi[i, global_start:global_end, global_start:global_end, -1] = 1.0

        centers_np = np.asarray(repeated_centers, dtype=np.int64)
        labels_np = self.graph.labels[centers_np].astype(np.int64)

        return {
            "feat": torch.from_numpy(feat),
            "mask": torch.from_numpy(mask),
            "phi": torch.from_numpy(phi),
            "centers": torch.from_numpy(centers_np),
            "labels": torch.from_numpy(labels_np),
        }


# ----------------------------
# Model
# ----------------------------


class ProximityMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, prox_dim: int, attn_dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prox_bias = nn.Parameter(torch.zeros(prox_dim))
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        bsz, n, _ = x.shape
        q = self.q_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        prox = torch.einsum("bijd,d->bij", phi, self.prox_bias).unsqueeze(1)
        scores = scores + prox

        key_mask = mask[:, None, None, :]
        query_mask = mask[:, None, :, None]
        scores = scores.masked_fill(~key_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn = attn * query_mask.to(attn.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n, self.hidden_dim)
        out = self.o_proj(out)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class GophormerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, prox_dim: int, ff_dim: int, dropout: float,
                 pre_ln: bool = False):
        super().__init__()
        self.pre_ln = pre_ln
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = ProximityMultiheadAttention(hidden_dim, num_heads, prox_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(-1).to(x.dtype)
        if self.pre_ln:          # previous version of this file
            x = (x + self.dropout(self.attn(self.ln1(x), mask, phi))) * keep
            x = (x + self.dropout(self.ff(self.ln2(x)))) * keep
            return x
        # Eq. 8 (GT framework): H^ = Norm(PE-MHA(H) + H), H = Norm(FFN(H^) + H^)
        x = self.ln1(x + self.dropout(self.attn(x, mask, phi))) * keep
        x = self.ln2(x + self.dropout(self.ff(x))) * keep
        return x


class GophormerModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        global_nodes: int,
        prox_order: int,
        pre_ln: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_nodes = global_nodes
        self.prox_dim = prox_order + 2
        self.pre_ln = pre_ln
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        if global_nodes > 0:
            self.global_embed = nn.Parameter(torch.randn(global_nodes, hidden_dim) * 0.02)
        else:
            self.register_parameter("global_embed", None)
        self.layers = nn.ModuleList([
            GophormerBlock(hidden_dim, num_heads, self.prox_dim, ff_dim, dropout, pre_ln)
            for _ in range(num_layers)
        ])
        # pre-LN needs a final norm; with post-LN every layer output is already normalised (Eq. 8)
        self.norm = nn.LayerNorm(hidden_dim) if pre_ln else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)   # 1-layer MLP f_mlp (Eq. 10)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feat)
        bsz, n, _ = x.shape

        if self.global_nodes > 0:
            real_n = n - self.global_nodes
            if real_n < 0:
                raise ValueError("sequence shorter than number of global nodes")
            x = x.clone()
            x[:, real_n:, :] = self.global_embed.unsqueeze(0)

        x = self.input_dropout(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, mask, phi)
        x = self.norm(x)
        center_repr = x[:, 0, :]          # read-out = center node (Eq. 9)
        return self.classifier(center_repr)


def sharpen_distribution(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    power = 1.0 / max(temperature, 1e-6)
    sharp = probs.pow(power)
    return sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# ----------------------------
# Training / evaluation (shared protocol)
# ----------------------------


def parse_fanouts(text: str) -> List[int]:
    text = text.strip()
    return [7, 7] if not text else [int(x) for x in text.split(",")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gophormer re-implementation on NPZ datasets (shared protocol)")
    add_protocol_args(p)
    g = p.add_argument_group("Gophormer (paper Sec. 4.1.4: fixed values + grid-search space)")
    g.add_argument("--hid-dim", type=int, default=64, help="fixed in the paper: 64")
    g.add_argument("--num-heads", type=int, default=8, help="fixed in the paper: 8")
    g.add_argument("--dropout", type=float, default=0.5, help="fixed in the paper: 0.5")
    g.add_argument("--weight-decay", type=float, default=1e-5, help="fixed in the paper: 1e-5")
    g.add_argument("--end-lr", type=float, default=1e-9, help="fixed in the paper: 1e-9")
    g.add_argument("--fanouts", type=parse_fanouts, default=[7, 7],
                   help="neighbours sampled per node and hop; the number of hops is the depth D (paper: D = 2)")
    g.add_argument("--num-layers", type=int, default=3, help="grid {1..5}")
    g.add_argument("--lr", type=float, default=2e-4, help="peak lr, grid {1e-4, 2e-4}")
    g.add_argument("--batch-size", type=int, default=32, help="grid {32, 64, 128}")
    g.add_argument("--global-nodes", type=int, default=2, help="n_g, grid {0..4}")
    g.add_argument("--prox-order", type=int, default=1,
                   help="highest power K of A~ in the proximity encoding; M = K + 2 views, grid M in {2, 3, 4}")
    g.add_argument("--cr-weight", type=float, default=0.5, help="lambda of Eq. 13, grid {0.5, 0.75, 1}")
    g.add_argument("--dim-feedforward", type=int, default=256, help="FFN width (not given in the paper: 4 x hidden)")
    g.add_argument("--train-samples", type=int, default=4, help="S ego-graphs per training node and epoch")
    g.add_argument("--eval-samples", type=int, default=5, help="S' of multi-sample inference (Fig. 5: 5-7 saturate)")
    g.add_argument("--cr-temperature", type=float, default=0.5, help="sharpening temperature of the consistency target")
    g.add_argument("--eval-batch-size", type=int, default=64)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--prox-norm", choices=["row", "sym"], default="row")
    g.add_argument("--pre-ln", action="store_true", help="pre-LN blocks of the previous version (paper: post-LN)")
    g.add_argument("--lr-scheduler", choices=["plateau", "none"], default="plateau")
    g.add_argument("--warmup-epochs", type=int, default=10, help="linear warm-up length (not given in the paper)")
    g.add_argument("--lr-decay", type=float, default=0.5, help="ReduceLROnPlateau factor (not given in the paper)")
    g.add_argument("--lr-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience in validation checks (not given in the paper)")
    return p


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.provenance import record_config
    record_config(args, __file__)
    if args.out_dir is None:
        args.out_dir = default_out_dir(HERE, "Gophormer", args.dataset)
    device = resolve_device(args.device)
    data = ProtocolData(args)
    graph = GraphData(data)
    print("==== Gophormer (re-implementation from the paper; no official code), shared protocol ====")
    print(f"data_root={args.data_root} {data.summary()} metric={args.metric} device={device}")
    print("config: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    results = []
    for run in range(args.runs):
        seed = args.base_seed + run
        suffix = args.split_suffixes[run % len(args.split_suffixes)]
        set_seed(seed)
        splits = data.splits(suffix)
        train_idx = splits["train"].astype(np.int64).copy()
        sampler = EgoGraphSampler(graph, args.fanouts, args.global_nodes, args.prox_order, args.prox_norm)
        model = GophormerModel(graph.feat_dim, graph.num_classes, args.hid_dim, args.num_layers, args.num_heads,
                               args.dim_feedforward, args.dropout, args.global_nodes, args.prox_order,
                               pre_ln=args.pre_ln).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        plateau = None
        warmup_steps = 0
        if args.lr_scheduler == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=args.lr_decay, patience=args.lr_patience, min_lr=args.end_lr, eps=0.0)
            warmup_steps = args.warmup_epochs * math.ceil(len(train_idx) / args.batch_size)
        step = [0]

        def train_epoch(epoch: int) -> float:
            model.train()
            rng = np.random.RandomState((seed * 1000003 + epoch) % (2 ** 32))
            rng.shuffle(train_idx)
            total_loss, total_samples = 0.0, 0
            for start in range(0, len(train_idx), args.batch_size):
                if step[0] < warmup_steps:          # linear warm-up from 0 to the peak lr
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * (step[0] + 1) / warmup_steps
                batch_centers = train_idx[start:start + args.batch_size]
                batch = sampler.collate(batch_centers.tolist(), args.train_samples, rng)
                feat = batch["feat"].to(device)
                mask = batch["mask"].to(device)
                phi = batch["phi"].to(device)
                labels = batch["labels"].to(device)

                logits = model(feat, mask, phi)
                loss = F.cross_entropy(logits, labels)                       # Eq. 11
                if args.train_samples > 1 and args.cr_weight > 0:
                    probs = torch.softmax(logits, dim=-1).view(len(batch_centers), args.train_samples, -1)
                    target = sharpen_distribution(probs.mean(dim=1), args.cr_temperature).detach()
                    consistency = ((probs - target.unsqueeze(1)) ** 2).sum(dim=-1).mean()   # Eq. 12
                    loss = loss + args.cr_weight * consistency                # Eq. 13

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step[0] += 1
                total_loss += float(loss.item()) * int(len(labels))
                total_samples += int(len(labels))
            return total_loss / max(total_samples, 1)

        @torch.no_grad()
        def predict(node_idx: np.ndarray, sample_seed: int) -> torch.Tensor:
            """Multi-sample inference (Eq. 14): mean of the S' predicted distributions."""
            model.eval()
            rng = np.random.RandomState(sample_seed % (2 ** 32))
            out = []
            for start in range(0, len(node_idx), args.eval_batch_size):
                batch_nodes = node_idx[start:start + args.eval_batch_size]
                batch = sampler.collate(batch_nodes.tolist(), args.eval_samples, rng)
                logits = model(batch["feat"].to(device), batch["mask"].to(device), batch["phi"].to(device))
                probs = torch.softmax(logits, dim=-1).view(len(batch_nodes), args.eval_samples, -1).mean(dim=1)
                out.append(probs.cpu())
            return torch.cat(out, dim=0)

        def evaluate(split: str) -> float:
            idx = splits[split].astype(np.int64)
            probs = predict(idx, seed * 7919 + (17 if split == "val" else 31))   # fixed ego-graphs per split
            score = compute_metric(args.metric, probs, graph.labels[idx], kind="probs")
            if split == "val" and plateau is not None and step[0] >= warmup_steps:
                plateau.step(score)
            return score

        tag = f"Gophormer seed={seed} split={suffix or '-'}"
        res = fit(model, train_epoch, evaluate, args, device, tag)
        results.append(RunResult(run=run, seed=seed, split_suffix=suffix, params=count_parameters(model),
                                 extra={"final_lr": float(optimizer.param_groups[0]["lr"])}, **res))
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("Gophormer", results, args,
           extra_config={"implementation": "re-implementation from arXiv:2110.13094 (no official code)",
                         "fanouts": args.fanouts, "prox_norm": args.prox_norm, "pre_ln": args.pre_ln,
                         "lr_scheduler": args.lr_scheduler})


if __name__ == "__main__":
    main()
