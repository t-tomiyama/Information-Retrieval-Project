"""Late fusion (rank aggregation) strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class LateFusion(ABC):
    """Merges per-space score lists over the same candidates into one score list."""

    name: str

    @abstractmethod
    def fuse(self, score_lists: Sequence[np.ndarray]) -> np.ndarray:
        ...

    def rank(self, score_lists: Sequence[np.ndarray]) -> np.ndarray:
        """Candidate indices ordered by fused score, best first."""
        return np.argsort(self.fuse(score_lists))[::-1]


class CombSumFusion(LateFusion):
    """CombSUM: sum of min-max normalised scores (Fox & Shaw, 1994)."""

    name = "ListAgg-CombSUM"

    def fuse(self, score_lists: Sequence[np.ndarray]) -> np.ndarray:
        normalised = []
        for scores in score_lists:
            scores = np.asarray(scores, dtype=float)
            spread = scores.max() - scores.min()
            normalised.append((scores - scores.min()) / spread if spread > 0 else np.zeros_like(scores))
        return np.sum(normalised, axis=0)


class ReciprocalRankFusion(LateFusion):
    """RRF: ``sum 1 / (k + rank)`` over the lists; uses positions only (Cormack et al., 2009)."""

    name = "ListAgg-RRF"

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, score_lists: Sequence[np.ndarray]) -> np.ndarray:
        n_items = len(score_lists[0])
        fused = np.zeros(n_items)
        for scores in score_lists:
            order = np.argsort(scores)[::-1]
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (rank + 1 + self.k)
        return fused
