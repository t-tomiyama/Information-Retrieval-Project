"""Evaluation queries: hand-written (synthetic) and mined from the corpus (data-driven)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from emotion_reranking.config import EMOTION_VALENCE

TITLE_COLUMN_CANDIDATES = ("Summary", "title", "review_summary", "Review Summary")


@dataclass(frozen=True)
class Query:
    """An evaluation query and the emotions it is expected to retrieve."""

    id: str
    text: str
    target_emotions: tuple[str, ...]
    valence: str | None
    source: str
    source_row_idx: int | None = field(default=None, compare=False)


def _synthetic(idx: int, text: str, emotions: Sequence[str], valence: str) -> Query:
    return Query(f"syn_{idx}", text, tuple(emotions), valence, "synthetic")


SYNTHETIC_QUERIES: tuple[Query, ...] = (
    _synthetic(1, "where is my refund they charged my account", ["anger", "annoyance"], "negative"),
    _synthetic(2, "customer service was extremely helpful and polite", ["admiration", "gratitude", "joy"], "positive"),
    _synthetic(3, "package never arrived and no one is helping me", ["anger", "sadness", "disappointment"], "negative"),
    _synthetic(4, "i absolutely love this product it works perfectly", ["love", "joy", "approval"], "positive"),
    _synthetic(5, "the delivery driver threw my package on the porch", ["anger", "disgust"], "negative"),
    _synthetic(6, "im very confused about how to cancel my prime membership", ["confusion", "nervousness"], "negative"),
    _synthetic(7, "arrived earlier than expected what a nice surprise", ["surprise", "joy", "excitement"], "positive"),
    _synthetic(8, "item was damaged right out of the box very sad", ["sadness", "disappointment"], "negative"),
    _synthetic(9, "thank you for quickly replacing my broken item", ["gratitude", "relief"], "positive"),
    _synthetic(10, "this is a complete scam do not buy from this seller", ["anger", "disapproval", "disgust"], "negative"),
    _synthetic(11, "the quality is much better than i hoped for", ["optimism", "approval", "joy"], "positive"),
    _synthetic(12, "i regret buying this it was a huge waste of money", ["remorse", "disappointment"], "negative"),
    _synthetic(13, "im a bit worried giving them my credit card info", ["fear", "nervousness"], "negative"),
    _synthetic(14, "the refund process was smooth and painless", ["relief", "approval"], "positive"),
    _synthetic(15, "they locked my account and i have no idea why", ["confusion", "anger"], "negative"),
    _synthetic(16, "best shopping experience ive had in years", ["admiration", "joy"], "positive"),
)


class DataDrivenQueryBuilder:
    """Builds queries from real reviews that strongly express each emotion.

    For every emotion, the reviews with the highest classifier score are
    taken as a pool (3x the number of queries wanted per emotion) and a few of
    them are sampled; the review title (or its first 12 words) becomes the
    query text. The source review is later removed from that query's
    candidate list, so a query never retrieves itself.
    """

    def __init__(self, corpus: pd.DataFrame, emotion_matrix: np.ndarray,
                 label_to_idx: Mapping[str, int], n_queries: int = 100,
                 valence: Mapping[str, str | None] = EMOTION_VALENCE,
                 rng: np.random.RandomState | None = None) -> None:
        self.corpus = corpus
        self.emotion_matrix = emotion_matrix
        self.label_to_idx = label_to_idx
        self.n_queries = n_queries
        self.valence = valence
        # Defaults to NumPy's global generator so that a global seed reproduces the queries.
        self.rng = rng if rng is not None else np.random

    def build(self) -> list[Query]:
        emotions = list(self.valence.keys())
        n_per_emotion = max(1, self.n_queries // len(emotions))
        title_col = next((c for c in TITLE_COLUMN_CANDIDATES if c in self.corpus.columns), None)

        queries: list[Query] = []
        counter = 1
        for emotion in emotions:
            col_idx = self.label_to_idx.get(emotion)
            if col_idx is None:
                continue

            scores = np.asarray(self.emotion_matrix[:, col_idx]).flatten()
            pool = np.argsort(scores)[::-1][:n_per_emotion * 3]
            sample_size = min(n_per_emotion, len(pool))
            if sample_size == 0:
                continue

            for row_idx in self.rng.choice(pool, size=sample_size, replace=False):
                queries.append(Query(
                    id=f"data_{emotion}_{counter}",
                    text=self._query_text(self.corpus.iloc[row_idx], title_col),
                    target_emotions=(emotion,),
                    valence=self.valence.get(emotion, "neutral"),
                    source="data_driven",
                    source_row_idx=int(row_idx),
                ))
                counter += 1
        return queries

    @staticmethod
    def _query_text(row: pd.Series, title_col: str | None) -> str:
        if title_col and pd.notna(row[title_col]) and len(str(row[title_col]).split()) >= 3:
            return str(row[title_col])
        words = str(row.get("clean_text", "")).split()
        return " ".join(words[:12]) if words else "empty review"
