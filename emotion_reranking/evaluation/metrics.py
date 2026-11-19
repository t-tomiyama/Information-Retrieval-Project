"""Ranking metrics. ``relevance`` may be binary or graded."""

from __future__ import annotations

import numpy as np


def precision_at_k(ranking: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    return relevance[ranking[:k]].sum() / k


def recall_at_k(ranking: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    total = relevance.sum()
    return relevance[ranking[:k]].sum() / total if total > 0 else 0.0


def ndcg_at_k(ranking: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[ranking[:k]]))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(ranking: np.ndarray, relevance: np.ndarray, k: int = 10) -> dict[str, float]:
    """Precision, recall and NDCG at ``k``, keyed as in the results tables."""
    return {
        f"precision_{k}": precision_at_k(ranking, relevance, k),
        f"recall_{k}": recall_at_k(ranking, relevance, k),
        f"ndcg{k}": ndcg_at_k(ranking, relevance, k),
    }
