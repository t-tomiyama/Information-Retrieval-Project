"""k-reciprocal nearest-neighbour re-ranking (Zhong et al., CVPR 2017)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from emotion_reranking.reranking.base import Reranker


def _knn_sets(dist: np.ndarray, k: int) -> list[set[int]]:
    return [set(np.argsort(dist[i])[:k + 1]) - {i} for i in range(dist.shape[0])]


def _reciprocal_sets(dist: np.ndarray, k: int) -> tuple[list[set[int]], list[set[int]]]:
    neighbours = _knn_sets(dist, k)
    reciprocal = [{g for g in neighbours[p] if p in neighbours[g]} for p in range(len(neighbours))]
    return reciprocal, neighbours


class KReciprocalReranker(Reranker):
    """Re-ranks by a blend of Jaccard distance over k-reciprocal sets and cosine distance.

    Final distance: ``(1 - lam) * d_jaccard + lam * d_cosine``.

    Args:
        k: neighbourhood size used to build the k-reciprocal sets.
        lam: weight of the original cosine distance.
    """

    name = "Reciprocal-kNN"

    def __init__(self, k: int = 15, lam: float = 0.3) -> None:
        self.k = k
        self.lam = lam

    def rerank(self, query_vec: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stacked = np.vstack([query_vec, candidates])
        dist = np.clip(1 - cosine_similarity(stacked), 0, None)
        n = dist.shape[0]
        k_eff = min(self.k, n - 1)

        reciprocal, neighbours = _reciprocal_sets(dist, k_eff)
        reciprocal_half, _ = _reciprocal_sets(dist, max(1, k_eff // 2))

        # Expand each k-reciprocal set with the half-size sets of its members
        # when they overlap enough (>= 2/3).
        expanded = []
        for p in range(n):
            r_p = set(reciprocal[p])
            for q in reciprocal[p]:
                rq_half = reciprocal_half[q]
                if len(rq_half) > 0 and len(rq_half & reciprocal_half[p]) >= (2 / 3) * len(rq_half):
                    r_p = r_p | rq_half
            expanded.append(r_p)

        # Gaussian-weighted indicator vectors.
        v = np.zeros((n, n))
        for p in range(n):
            for g in expanded[p]:
                v[p, g] = np.exp(-dist[p, g])

        # Local query expansion: average over the k nearest neighbours.
        v_exp = np.zeros_like(v)
        for p in range(n):
            neigh = neighbours[p] if len(neighbours[p]) > 0 else {p}
            v_exp[p] = v[list(neigh)].mean(axis=0)

        # Jaccard distance between expanded vectors.
        d_jaccard = np.zeros((n, n))
        for p in range(n):
            mins = np.minimum(v_exp[p], v_exp).sum(axis=1)
            maxs = np.maximum(v_exp[p], v_exp).sum(axis=1)
            maxs[maxs == 0] = 1e-9
            d_jaccard[p] = 1 - mins / maxs

        d_final = (1 - self.lam) * d_jaccard[0] + self.lam * dist[0]
        ranking = np.argsort(d_final[1:])
        return ranking, -d_final[1:][ranking]
