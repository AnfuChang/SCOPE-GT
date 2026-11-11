"""Context sampling for the relation-evidence diagnostic (Fig. 1a)."""
from __future__ import annotations

import numpy as np


def sample_sequences(g, centers, k, rng):
    """Uniformly sample at most k contexts per center from its h-hop ball, without replacement.

    Returns (ctx [B,k], -1 for empty slots; dist [B,k], 0 for empty slots). Same procedure as
    A_experiment.sample_contexts: sort the candidates of each row by a random key and keep the first k.
    """
    rows_in_ball = g.ball_row[centers]
    if (rows_in_ball < 0).any():
        raise ValueError("some centers are not inside the precomputed h-hop balls")
    starts, ends = g.ball.indptr[rows_in_ball], g.ball.indptr[rows_in_ball + 1]
    counts = ends - starts
    rows = np.repeat(np.arange(len(centers)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.arange(counts.sum()) - first
    flat = np.repeat(starts, counts) + pos
    cand, cand_dist = g.ball.indices[flat].astype(np.int64), g.ball.data[flat].astype(np.int64)
    order = np.lexsort((rng.random(len(cand)), rows))                  # shuffle within rows; row order/length kept
    keep = pos < k
    ctx = np.full((len(centers), k), -1, np.int64)
    dist = np.zeros((len(centers), k), np.int64)
    ctx[rows[keep], pos[keep]] = cand[order][keep]
    dist[rows[keep], pos[keep]] = cand_dist[order][keep]
    return ctx, dist
