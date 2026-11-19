"""Re-ranker interface (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Reranker(ABC):
    """Re-orders a candidate list using the geometry of a vector space.

    Implementations receive the query vector and the candidate vectors in the
    same feature space and return ``(ranking, scores)``: ``ranking`` holds
    candidate indices, best first, and ``scores`` the matching scores
    (higher is better).
    """

    name: str

    @abstractmethod
    def rerank(self, query_vec: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"
