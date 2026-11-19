"""RL-Sim: re-ranking by similarity of ranked lists (Pedronette & Torres, 2013)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from emotion_reranking.reranking.base import Reranker


class RLSimReranker(Reranker):
    """Iteratively replaces pairwise similarity by the overlap of top-k neighbour lists.

    Two items are considered similar when their k-nearest-neighbour lists
    share many elements; ``k`` grows by one at each iteration.

    Args:
        iterations: number of refinement iterations.
        neighbors: initial neighbourhood size ``k``.
    """

    name = "RL-Sim"

    def __init__(self, iterations: int = 2, neighbors: int = 10) -> None:
        self.iterations = iterations
        self.neighbors = neighbors

    def rerank(self, query_vec: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = cosine_similarity(np.vstack([query_vec, candidates]))
        n = sim.shape[0]
        corr = sim.copy()

        for t in range(self.iterations):
            kt = min(self.neighbors + t, n - 1)
            top_lists = [set(np.argsort(corr[i])[::-1][1:kt + 1]) for i in range(n)]
            new_corr = np.zeros_like(corr)
            for i in range(n):
                for j in range(n):
                    new_corr[i, j] = 1.0 if i == j else len(top_lists[i] & top_lists[j]) / kt
            corr = new_corr

        # Row 0 is the query; columns 1.. are the candidates.
        ranking = np.argsort(corr[0, 1:])[::-1]
        return ranking, corr[0, 1:][ranking]
