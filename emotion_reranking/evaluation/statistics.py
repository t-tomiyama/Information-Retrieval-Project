"""Statistical comparison of early fusion against the text-only baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TEXT_ONLY = "text-only"
EMOTION_FUSED = "emotion-fused"
LIST_AGGREGATION = "list-aggregation"


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 42,
                        ci: float = 0.95) -> tuple[float, float, float]:
    """Mean of paired differences and its percentile bootstrap confidence interval."""
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    low = np.percentile(boots, (1 - ci) / 2 * 100)
    high = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(diffs)), float(low), float(high)


def run_significance_tests(metrics: pd.DataFrame,
                           algorithms: Sequence[str] = ("RL-Sim", "Reciprocal-kNN"),
                           n_boot: int = 2000) -> pd.DataFrame:
    """One-sided Wilcoxon signed-rank test (fused > text-only) on per-query NDCG@10,
    plus a bootstrap CI of the mean difference, for every algorithm and alpha."""
    rows = []
    for rel_type, rel_df in metrics.groupby("relevance_type"):
        for algo in algorithms:
            algo_df = rel_df[rel_df.algorithm == algo]
            base = (algo_df[algo_df.feature_space == TEXT_ONLY][["query_id", "ndcg10"]]
                    .rename(columns={"ndcg10": "ndcg_text"}))
            if base.empty:
                continue

            for alpha, grp in algo_df[algo_df.feature_space == EMOTION_FUSED].groupby("alpha"):
                merged = base.merge(grp[["query_id", "ndcg10"]].rename(columns={"ndcg10": "ndcg_fused"}),
                                    on="query_id", how="inner")
                if len(merged) < 2:
                    continue

                try:
                    _, p_value = wilcoxon(merged.ndcg_fused, merged.ndcg_text, alternative="greater")
                except ValueError:
                    p_value = 1.0

                mean_delta, ci_low, ci_high = paired_bootstrap_ci(
                    (merged.ndcg_fused - merged.ndcg_text).values, n_boot=n_boot)
                rows.append({
                    "relevance_type": rel_type,
                    "algorithm": algo,
                    "alpha": alpha,
                    "n_queries": len(merged),
                    "mean_ndcg_text": merged.ndcg_text.mean(),
                    "mean_ndcg_fused": merged.ndcg_fused.mean(),
                    "mean_delta_ndcg": mean_delta,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "p_value": p_value,
                    "significant_alpha005": p_value < 0.05,
                })
    return pd.DataFrame(rows)


def summarize_list_aggregation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean P@10, R@10 and NDCG@10 of each late-fusion method."""
    late = metrics[metrics["feature_space"] == LIST_AGGREGATION]
    if late.empty:
        return pd.DataFrame()
    return (late.groupby(["relevance_type", "algorithm"])[["precision_10", "recall_10", "ndcg10"]]
            .mean().reset_index())
