"""Figures for the experiment results.

Every figure has one panel per relevance type (emotion | rating) and is built
only from the saved result tables, so figures can be regenerated without
re-running the models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emotion_reranking.evaluation.statistics import EMOTION_FUSED, LIST_AGGREGATION, TEXT_ONLY  # noqa: E402

# Palette (validated for colour-vision deficiency on a white surface).
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
BASELINE_GRAY = "#a3a29c"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"

METHOD_COLORS = {"text": BASELINE_GRAY, "early": BLUE, "combsum": ORANGE, "rrf": AQUA}
ALGORITHM_COLORS = {"Reciprocal-kNN": BLUE, "RL-Sim": MAGENTA}

RELEVANCE_TITLES = {"emotion": "Hybrid emotion relevance", "rating": "Rating relevance"}

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_PRIMARY,
}

REFERENCE_ALGORITHM = "Reciprocal-kNN"


# --------------------------------------------------------------------------- data helpers

def _relevance_types(metrics: pd.DataFrame) -> list[str]:
    order = [r for r in RELEVANCE_TITLES if r in set(metrics["relevance_type"])]
    return order + sorted(set(metrics["relevance_type"]) - set(order))


def best_early_alpha(metrics: pd.DataFrame, relevance_type: str,
                     algorithm: str = REFERENCE_ALGORITHM) -> float:
    """Alpha with the highest mean NDCG@10 for early fusion with ``algorithm``."""
    early = metrics[(metrics.relevance_type == relevance_type)
                    & (metrics.feature_space == EMOTION_FUSED)
                    & (metrics.algorithm == algorithm)]
    return float(early.groupby("alpha")["ndcg10"].mean().idxmax())


def method_ndcg(metrics: pd.DataFrame, relevance_type: str) -> dict[str, tuple[str, np.ndarray]]:
    """Per-query NDCG@10 of the four compared approaches: ``key -> (label, values)``."""
    m = metrics[metrics.relevance_type == relevance_type]
    alpha = best_early_alpha(metrics, relevance_type)
    ref = m.algorithm == REFERENCE_ALGORITHM
    return {
        "text": ("Text-only\nbaseline", m[ref & (m.feature_space == TEXT_ONLY)]["ndcg10"].to_numpy()),
        "early": (f"Early fusion\n(α={alpha:g})",
                  m[ref & (m.feature_space == EMOTION_FUSED) & (m.alpha == alpha)]["ndcg10"].to_numpy()),
        "combsum": ("Late fusion\nCombSUM", m[(m.feature_space == LIST_AGGREGATION)
                                              & (m.algorithm == "ListAgg-CombSUM")]["ndcg10"].to_numpy()),
        "rrf": ("Late fusion\nRRF", m[(m.feature_space == LIST_AGGREGATION)
                                      & (m.algorithm == "ListAgg-RRF")]["ndcg10"].to_numpy()),
    }


# --------------------------------------------------------------------------- panels

def plot_fusion_comparison(ax: plt.Axes, metrics: pd.DataFrame, relevance_type: str) -> None:
    """Mean NDCG@10: text-only vs best early fusion vs late fusion."""
    methods = method_ndcg(metrics, relevance_type)
    labels = [label for label, _ in methods.values()]
    means = [values.mean() for _, values in methods.values()]
    colors = [METHOD_COLORS[key] for key in methods]

    bars = ax.bar(labels, means, color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, means):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", color=TEXT_PRIMARY, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean NDCG@10")
    ax.grid(axis="x", visible=False)


def plot_query_variance(ax: plt.Axes, metrics: pd.DataFrame, relevance_type: str) -> None:
    """Distribution of per-query NDCG@10 for the four approaches."""
    methods = method_ndcg(metrics, relevance_type)
    data = [values for _, values in methods.values()]
    box = ax.boxplot(data, tick_labels=[label for label, _ in methods.values()], patch_artist=True,
                     widths=0.55, showmeans=True,
                     medianprops={"color": TEXT_PRIMARY, "linewidth": 1.6},
                     whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY},
                     flierprops={"marker": "o", "markersize": 4, "markerfacecolor": BASELINE_GRAY,
                                 "markeredgecolor": "none"},
                     meanprops={"marker": "D", "markersize": 6, "markerfacecolor": SURFACE,
                                "markeredgecolor": TEXT_PRIMARY})
    for patch, key in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[key])
        patch.set_edgecolor(TEXT_SECONDARY)
    for pos, values in enumerate(data, start=1):
        ax.annotate(f"mean {values.mean():.3f}", (pos, 1.0), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("NDCG@10 per query")
    ax.grid(axis="x", visible=False)


def plot_confidence_intervals(ax: plt.Axes, stats: pd.DataFrame, relevance_type: str) -> None:
    """Mean ΔNDCG@10 (early fusion − text-only) with 95% bootstrap CIs, per alpha."""
    s = stats[stats.relevance_type == relevance_type]
    algorithms = [a for a in ALGORITHM_COLORS if a in set(s.algorithm)]
    offsets = np.linspace(-0.012, 0.012, len(algorithms)) if len(algorithms) > 1 else [0.0]

    ax.axhline(0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    for algo, offset in zip(algorithms, offsets):
        g = s[s.algorithm == algo].sort_values("alpha")
        mean = g.mean_delta_ndcg.to_numpy()
        yerr = np.vstack([mean - g.delta_ci_low.to_numpy(), g.delta_ci_high.to_numpy() - mean])
        color = ALGORITHM_COLORS[algo]
        ax.errorbar(g.alpha + offset, mean, yerr=yerr, color=color, ecolor=color, linewidth=2,
                    elinewidth=1.6, capsize=4, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=algo)
        significant = g[g.significant_alpha005]
        ax.scatter(significant.alpha + offset, significant.mean_delta_ndcg, marker="*", s=170,
                   color=TEXT_PRIMARY, zorder=5, label=None)
    ax.set_xlabel("Emotion weight α")
    ax.set_ylabel("Δ NDCG@10 vs text-only")
    ax.set_xticks(sorted(s.alpha.unique()))


def plot_alpha_sweep(ax: plt.Axes, metrics: pd.DataFrame, relevance_type: str) -> None:
    """Mean NDCG@10 of early fusion across alpha, against each algorithm's text-only baseline."""
    m = metrics[metrics.relevance_type == relevance_type]
    for algo, color in ALGORITHM_COLORS.items():
        a = m[m.algorithm == algo]
        if a.empty:
            continue
        sweep = a[a.feature_space == EMOTION_FUSED].groupby("alpha")["ndcg10"].mean()
        baseline = a[a.feature_space == TEXT_ONLY]["ndcg10"].mean()
        ax.plot(sweep.index, sweep.values, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{algo} (early fusion)")
        ax.axhline(baseline, color=color, linestyle="--", linewidth=1.2, label=f"{algo} (text-only)")
    ax.set_xlabel("Emotion weight α")
    ax.set_ylabel("Mean NDCG@10")
    ax.set_xticks(sorted(m[m.feature_space == EMOTION_FUSED].alpha.unique()))


# --------------------------------------------------------------------------- figures

def _two_panel_figure(plot_panel, data, relevance_types: Sequence[str], suptitle: str,
                      path: Path, sharey: bool = True) -> Path:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(relevance_types), figsize=(5.6 * len(relevance_types), 4.4),
                                 sharey=sharey, squeeze=False)
        for ax, rel in zip(axes[0], relevance_types):
            plot_panel(ax, data, rel)
            ax.set_title(RELEVANCE_TITLES.get(rel, rel))
        if sharey:
            for ax in axes[0][1:]:
                ax.set_ylabel("")
                ax.tick_params(labelleft=True)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=TEXT_PRIMARY)

        # One shared legend below the panels (series are identical across panels).
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9.5)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
        else:
            fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return path


def generate_figures(metrics: pd.DataFrame, stats: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write every result figure to ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = _relevance_types(metrics)
    return [
        _two_panel_figure(plot_fusion_comparison, metrics, rels,
                          "Early vs. late fusion (mean NDCG@10)", out_dir / "fusion_comparison.png"),
        _two_panel_figure(plot_query_variance, metrics, rels,
                          "Per-query NDCG@10 distribution", out_dir / "query_variance.png"),
        _two_panel_figure(plot_confidence_intervals, stats, rels,
                          "Early-fusion gain over text-only (95% bootstrap CI, ★ = p < 0.05)",
                          out_dir / "confidence_intervals.png", sharey=False),
        _two_panel_figure(plot_alpha_sweep, metrics, rels,
                          "Early fusion: emotion weight sweep", out_dir / "alpha_sweep.png", sharey=False),
    ]


def plot_emotion_vs_informativeness(info_scores: np.ndarray, emotion_matrix: np.ndarray,
                                    label_to_idx: Mapping[str, int], path: Path,
                                    emotions: Sequence[str] = ("anger", "sadness", "joy", "admiration", "neutral"),
                                    ) -> Path:
    """Small multiples: emotion intensity vs informativeness of every review."""
    emotions = [e for e in emotions if e in label_to_idx]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(emotions), figsize=(3.0 * len(emotions), 3.2),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, emotion in zip(axes[0], emotions):
            ax.scatter(info_scores, emotion_matrix[:, label_to_idx[emotion]], s=8, alpha=0.35,
                       color=BLUE, edgecolors="none")
            ax.set_title(emotion.capitalize())
            ax.set_xlabel("Informativeness")
        axes[0][0].set_ylabel("Emotion score (GoEmotions)")
        fig.suptitle("Emotion intensity vs. informativeness", fontsize=13, fontweight="bold",
                     color=TEXT_PRIMARY)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return Path(path)
