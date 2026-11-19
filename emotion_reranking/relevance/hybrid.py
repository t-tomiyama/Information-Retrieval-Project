"""Hybrid relevance: emotional intensity + informativeness + aspect mentions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from emotion_reranking.config import ASPECT_TERMS, HybridRelevanceWeights
from emotion_reranking.data.queries import Query
from emotion_reranking.relevance.base import RelevanceJudge


def informativeness(text: str, aspects: Sequence[str] = ASPECT_TERMS) -> float:
    """Informativeness score in [0, 1].

    ``0.4 * length + 0.4 * aspect density + 0.2 * lexical richness``, where
    length saturates at 50 words and aspect density at 5 distinct aspects.
    """
    text = str(text).lower()
    words = text.split()
    if not words:
        return 0.0

    length_score = min(len(words) / 50.0, 1.0)
    aspect_score = min(len({a for a in aspects if a in text}) / 5.0, 1.0)
    lexical_richness = len(set(words)) / len(words)
    return min(0.4 * length_score + 0.4 * aspect_score + 0.2 * lexical_richness, 1.0)


class HybridEmotionRelevance(RelevanceJudge):
    """Graded relevance favouring detailed reviews that express the query's emotions.

    ``relevance = w_e * max(target emotion scores) + w_i * informativeness + w_a * mentions_aspect``

    Args:
        texts: cleaned text of every document in the corpus.
        emotion_matrix: (n_docs x n_emotions) emotion probabilities.
        label_to_idx: emotion label -> column of ``emotion_matrix``.
        weights: weights of the three components.
        aspects: aspect lexicon.
    """

    name = "emotion"

    def __init__(self, texts: Sequence[str], emotion_matrix: np.ndarray,
                 label_to_idx: Mapping[str, int],
                 weights: HybridRelevanceWeights = HybridRelevanceWeights(),
                 aspects: Sequence[str] = ASPECT_TERMS) -> None:
        self.texts = list(texts)
        self.emotion_matrix = emotion_matrix
        self.label_to_idx = label_to_idx
        self.weights = weights
        self.aspects = tuple(aspects)

    def judge(self, query: Query, candidate_idx: np.ndarray) -> np.ndarray:
        targets = {self.label_to_idx[e] for e in query.target_emotions if e in self.label_to_idx}
        return np.array([self.score(int(i), targets) for i in candidate_idx])

    def score(self, doc_idx: int, target_cols: set[int]) -> float:
        emotion_scores = [self.emotion_matrix[doc_idx, c] for c in target_cols
                          if c < self.emotion_matrix.shape[1]]
        emotion_score = max(emotion_scores) if emotion_scores else 0.0

        text = self.texts[doc_idx]
        info_score = informativeness(text, self.aspects)
        aspect_score = 1.0 if any(a in text.lower() for a in self.aspects) else 0.0

        w = self.weights
        return min(w.emotion * emotion_score + w.informativeness * info_score + w.aspect * aspect_score, 1.0)
