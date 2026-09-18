"""Retrieval: narrative-embedding cosine top-k. Pure numpy; runs in LinkBatch.

Retrieval uses embeddings ONLY and scoring uses structured MO ONLY (CLAUDE.md
rule 1) — filtering on a scored field would double-count it. Restricting
candidates to the same crime type (or to other types) is allowed: crime type
defines the pool, it is not a scored field.

Never materialise the full N × N matrix: 1,000 query rows at a time against
the whole corpus, keep the top k, discard the block.
"""
from __future__ import annotations

import numpy as np


def top_k(query_rows: np.ndarray, vectors: np.ndarray, k: int = 50, groups: np.ndarray | None = None,
          scope: str = "all", chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Top-k neighbours by cosine for each query row (vectors L2-normalised).

    scope: "all", "same" (same group as the query) or "other" (different
    group); groups is the per-row crime type. The query itself is excluded.
    Returns (indices, similarities), each (len(query_rows), k), best first.
    """
    corpus = vectors.astype(np.float32, copy=False)
    out_idx = np.empty((len(query_rows), k), dtype=np.int64)
    out_sim = np.empty((len(query_rows), k), dtype=np.float32)
    for start in range(0, len(query_rows), chunk):
        rows = query_rows[start:start + chunk]
        sims = corpus[rows] @ corpus.T
        sims[np.arange(len(rows)), rows] = -np.inf
        if scope != "all":
            same = groups[None, :] == groups[rows][:, None]
            sims[~same if scope == "same" else same] = -np.inf
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        best = np.take_along_axis(part, order, axis=1)
        out_idx[start:start + len(rows)] = best
        out_sim[start:start + len(rows)] = np.take_along_axis(sims, best, axis=1)
    return out_idx, out_sim
