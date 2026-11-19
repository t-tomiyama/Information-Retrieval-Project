"""Rating-based relevance: does the review's star rating agree with the query's valence?"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from emotion_reranking.data.loader import HIGH_RATING, LOW_RATING
from emotion_reranking.data.queries import Query
from emotion_reranking.relevance.base import RelevanceJudge


class RatingRelevance(RelevanceJudge):
    """Binary relevance: 1 if the review is 4-5 stars for a positive query
    (1-3 stars for a negative one), else 0. Undefined for neutral queries.

    Args:
        rating_categories: rating class of every document in the corpus.
    """

    name = "rating"

    def __init__(self, rating_categories: Sequence[str]) -> None:
        self.rating_categories = np.asarray(rating_categories)

    def judge(self, query: Query, candidate_idx: np.ndarray) -> np.ndarray | None:
        if query.valence is None:
            return None
        target = HIGH_RATING if query.valence == "positive" else LOW_RATING
        return (self.rating_categories[candidate_idx] == target).astype(int)
