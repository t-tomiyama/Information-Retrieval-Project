"""Relevance judgment interface (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from emotion_reranking.data.queries import Query


class RelevanceJudge(ABC):
    """Produces the ground-truth relevance of each candidate for a query.

    ``judge`` returns one value per candidate (binary or graded in [0, 1]),
    or ``None`` when this notion of relevance is undefined for the query;
    in that case the query is skipped for this judge.
    """

    name: str

    @abstractmethod
    def judge(self, query: Query, candidate_idx: np.ndarray) -> np.ndarray | None:
        ...
