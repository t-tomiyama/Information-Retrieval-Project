"""Ranking metrics and significance testing."""

from emotion_reranking.evaluation.metrics import evaluate_ranking, ndcg_at_k, precision_at_k, recall_at_k
from emotion_reranking.evaluation.statistics import (
    EMOTION_FUSED,
    LIST_AGGREGATION,
    TEXT_ONLY,
    paired_bootstrap_ci,
    run_significance_tests,
    summarize_list_aggregation,
)

__all__ = [
    "EMOTION_FUSED",
    "LIST_AGGREGATION",
    "TEXT_ONLY",
    "evaluate_ranking",
    "ndcg_at_k",
    "paired_bootstrap_ci",
    "precision_at_k",
    "recall_at_k",
    "run_significance_tests",
    "summarize_list_aggregation",
]
