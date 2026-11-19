"""End-to-end experiment: BM25 retrieval -> re-ranking / fusion -> evaluation.

:class:`ExperimentPipeline` is the single entry point (a facade over the
components). Every collaborator - loader, encoders, re-rankers, late-fusion
methods - can be injected, which keeps the orchestration testable without
downloading any model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from emotion_reranking.config import ExperimentConfig
from emotion_reranking.data.loader import AmazonReviewsLoader
from emotion_reranking.data.preprocessing import clean_text, tokenize_and_stem
from emotion_reranking.data.queries import SYNTHETIC_QUERIES, DataDrivenQueryBuilder, Query
from emotion_reranking.encoders.base import EmotionEncoder, TextEncoder
from emotion_reranking.evaluation.metrics import evaluate_ranking
from emotion_reranking.evaluation.statistics import (
    EMOTION_FUSED,
    LIST_AGGREGATION,
    TEXT_ONLY,
    run_significance_tests,
    summarize_list_aggregation,
)
from emotion_reranking.fusion import CombSumFusion, EarlyFusion, LateFusion, ReciprocalRankFusion
from emotion_reranking.relevance import HybridEmotionRelevance, RatingRelevance, RelevanceJudge, informativeness
from emotion_reranking.reranking import KReciprocalReranker, Reranker, RLSimReranker
from emotion_reranking.retrieval import BM25Index

logger = logging.getLogger(__name__)

METRICS_FILE = "all_query_metrics.csv"
SIGNIFICANCE_FILE = "significance_tests.csv"
LIST_AGGREGATION_FILE = "list_aggregation_summary.csv"


# --------------------------------------------------------------------------- factories

def default_rerankers(config: ExperimentConfig) -> list[Reranker]:
    return [
        RLSimReranker(iterations=config.rl_sim_iterations, neighbors=config.rl_sim_neighbors),
        KReciprocalReranker(k=config.reciprocal_k, lam=config.reciprocal_lambda),
    ]


def default_late_fusers(config: ExperimentConfig) -> list[LateFusion]:
    return [CombSumFusion(), ReciprocalRankFusion(k=config.rrf_k)]


# --------------------------------------------------------------------------- results

@dataclass
class ExperimentResults:
    """Per-query metrics plus the aggregated tables derived from them."""

    metrics: pd.DataFrame
    significance: pd.DataFrame
    list_aggregation: pd.DataFrame

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.to_csv(out_dir / METRICS_FILE, index=False)
        self.significance.to_csv(out_dir / SIGNIFICANCE_FILE, index=False)
        self.list_aggregation.to_csv(out_dir / LIST_AGGREGATION_FILE, index=False)

    @classmethod
    def load(cls, out_dir: Path) -> "ExperimentResults":
        out_dir = Path(out_dir)
        return cls(
            metrics=pd.read_csv(out_dir / METRICS_FILE),
            significance=pd.read_csv(out_dir / SIGNIFICANCE_FILE),
            list_aggregation=pd.read_csv(out_dir / LIST_AGGREGATION_FILE),
        )


@dataclass
class _SearchSpace:
    """Everything needed to process a query against the indexed corpus."""

    bm25: BM25Index
    text_encoder: TextEncoder
    emotion_encoder: EmotionEncoder
    text_scaler: StandardScaler
    emotion_scaler: StandardScaler
    text_vectors: np.ndarray      # standardised text embeddings (n_docs x d_text)
    emotion_vectors: np.ndarray   # standardised emotion vectors (n_docs x 28)

    def encode_query(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        q_text = self.text_encoder.encode([clean_text(text)])[0]
        q_emotion = self.emotion_encoder.encode_one(text)
        return (self.text_scaler.transform(q_text.reshape(1, -1))[0],
                self.emotion_scaler.transform(q_emotion.reshape(1, -1))[0])


# --------------------------------------------------------------------------- pipeline

class ExperimentPipeline:
    """Runs the full experiment described by an :class:`ExperimentConfig`.

    For each query:

    1. BM25 retrieves the top-``bm25_top_k`` candidates;
    2. candidates are re-ranked in the text-only space (baseline), in the
       early-fused text+emotion space for every alpha, and by late fusion
       of the text and emotion similarity lists;
    3. each ranking is scored against every relevance judgment.
    """

    def __init__(self, config: ExperimentConfig | None = None, *,
                 loader: AmazonReviewsLoader | None = None,
                 emotion_encoder: EmotionEncoder | None = None,
                 text_encoder: TextEncoder | None = None,
                 rerankers: Sequence[Reranker] | None = None,
                 late_fusers: Sequence[LateFusion] | None = None,
                 make_figures: bool = True) -> None:
        self.config = config or ExperimentConfig()
        cfg = self.config
        self.loader = loader or AmazonReviewsLoader(cfg.data_path, cfg.kaggle_dataset, cfg.n_docs,
                                                    min_per_class=cfg.min_docs_per_class, seed=cfg.seed)
        self._emotion_encoder = emotion_encoder
        self._text_encoder = text_encoder
        self.rerankers = list(rerankers) if rerankers is not None else default_rerankers(cfg)
        self.late_fusers = list(late_fusers) if late_fusers is not None else default_late_fusers(cfg)
        self.make_figures = make_figures

    # ---------------------------------------------------------------- public API

    def run(self) -> ExperimentResults:
        cfg = self.config
        # Global seed: data-driven queries are sampled with NumPy's global generator.
        np.random.seed(cfg.seed)

        logger.info("[Data] Loading corpus...")
        corpus = self.loader.load()
        logger.info("[Data] %d reviews after stratified sampling.", len(corpus))

        logger.info("[Encoder] GoEmotions vectors (RoBERTa)...")
        emotion_encoder = self._emotion_encoder or self._default_emotion_encoder()
        truncated = corpus["clean_text"].str.split().str[:cfg.max_words_emotion].str.join(" ").tolist()
        emotion_matrix = emotion_encoder.encode(truncated)

        info_scores = np.array([informativeness(t) for t in corpus["clean_text"]])
        self._log_emotion_informativeness_correlation(emotion_matrix, info_scores, emotion_encoder)

        bm25 = BM25Index(corpus["bm25_tokens"].tolist(), k1=cfg.bm25_k1, b=cfg.bm25_b)

        logger.info("[Encoder] Dense text embeddings (Sentence-BERT)...")
        text_encoder = self._text_encoder or self._default_text_encoder()
        text_matrix = text_encoder.encode(corpus["clean_text"].tolist())
        text_scaler = StandardScaler().fit(text_matrix)
        emotion_scaler = StandardScaler().fit(emotion_matrix)
        space = _SearchSpace(bm25, text_encoder, emotion_encoder, text_scaler, emotion_scaler,
                             text_scaler.transform(text_matrix), emotion_scaler.transform(emotion_matrix))

        judges: list[RelevanceJudge] = [
            HybridEmotionRelevance(corpus["clean_text"].tolist(), emotion_matrix,
                                   emotion_encoder.label_to_idx, weights=cfg.relevance_weights),
            RatingRelevance(corpus["rating_category"].to_numpy()),
        ]

        queries = self.build_queries(corpus, emotion_matrix, emotion_encoder.label_to_idx)
        n_synthetic = sum(q.source == "synthetic" for q in queries)
        logger.info("[Pipeline] Evaluating %d queries (%d synthetic, %d data-driven)...",
                    len(queries), n_synthetic, len(queries) - n_synthetic)

        rows: list[dict] = []
        for query in queries:
            rows.extend(self.evaluate_query(query, space, judges))

        metrics = pd.DataFrame(rows)
        results = ExperimentResults(
            metrics=metrics,
            significance=run_significance_tests(metrics, [r.name for r in self.rerankers], cfg.n_bootstrap),
            list_aggregation=summarize_list_aggregation(metrics),
        )
        results.save(cfg.output_dir)

        if self.make_figures:
            from emotion_reranking.visualization import generate_figures, plot_emotion_vs_informativeness

            generate_figures(results.metrics, results.significance, cfg.figures_dir)
            plot_emotion_vs_informativeness(info_scores, emotion_matrix, emotion_encoder.label_to_idx,
                                            cfg.figures_dir / "emotion_vs_informativeness.png")

        logger.info("\nEarly fusion vs. text-only (Wilcoxon + bootstrap CI)\n%s",
                    results.significance.to_string(index=False))
        logger.info("\nLate fusion summary\n%s", results.list_aggregation.to_string(index=False))
        logger.info("\nResults saved to '%s'.", cfg.output_dir)
        return results

    def build_queries(self, corpus: pd.DataFrame, emotion_matrix: np.ndarray,
                      label_to_idx) -> list[Query]:
        mode = self.config.query_mode
        queries = list(SYNTHETIC_QUERIES) if mode in ("synthetic", "both") else []
        if mode in ("data_driven", "both"):
            queries += DataDrivenQueryBuilder(corpus, emotion_matrix, label_to_idx,
                                              n_queries=self.config.n_data_driven_queries).build()
        return queries

    def evaluate_query(self, query: Query, space: _SearchSpace,
                       judges: Sequence[RelevanceJudge]) -> list[dict]:
        cfg = self.config
        candidates = space.bm25.top_k(tokenize_and_stem(query.text), cfg.bm25_top_k)
        if query.source_row_idx is not None:
            candidates = candidates[candidates != query.source_row_idx]

        relevance = {}
        for judge in judges:
            judged = judge.judge(query, candidates)
            if judged is not None:
                relevance[judge.name] = judged

        cand_text = space.text_vectors[candidates]
        cand_emotion = space.emotion_vectors[candidates]
        q_text, q_emotion = space.encode_query(query.text)

        rows: list[dict] = []

        def record(algorithm: str, feature_space: str, alpha: float, ranking: np.ndarray) -> None:
            for rel_name, rel in relevance.items():
                rows.append({
                    "query_id": query.id,
                    "query_source": query.source,
                    "algorithm": algorithm,
                    "feature_space": feature_space,
                    "alpha": alpha,
                    "relevance_type": rel_name,
                    **evaluate_ranking(ranking, rel, cfg.eval_k),
                })

        # Baseline: re-ranking in the text-only space.
        for reranker in self.rerankers:
            ranking, _ = reranker.rerank(q_text, cand_text)
            record(reranker.name, TEXT_ONLY, 0.0, ranking)

        # Early fusion: re-ranking in the concatenated text + alpha * emotion space.
        for alpha in cfg.alpha_values:
            fusion = EarlyFusion(alpha)
            q_fused = fusion.combine(q_text, q_emotion)
            cand_fused = fusion.combine(cand_text, cand_emotion)
            for reranker in self.rerankers:
                ranking, _ = reranker.rerank(q_fused, cand_fused)
                record(reranker.name, EMOTION_FUSED, alpha, ranking)

        # Late fusion: aggregate the text and emotion similarity lists.
        text_sim = cosine_similarity(q_text.reshape(1, -1), cand_text)[0]
        emotion_sim = cosine_similarity(q_emotion.reshape(1, -1), cand_emotion)[0]
        for fuser in self.late_fusers:
            record(fuser.name, LIST_AGGREGATION, np.nan, fuser.rank([text_sim, emotion_sim]))

        return rows

    # ---------------------------------------------------------------- helpers

    def _default_emotion_encoder(self) -> EmotionEncoder:
        from emotion_reranking.encoders import GoEmotionsEncoder

        return GoEmotionsEncoder(self.config.emotion_model, batch_size=self.config.embed_batch_size)

    def _default_text_encoder(self) -> TextEncoder:
        from emotion_reranking.encoders import SentenceBertEncoder

        return SentenceBertEncoder(self.config.text_model)

    @staticmethod
    def _log_emotion_informativeness_correlation(emotion_matrix: np.ndarray, info_scores: np.ndarray,
                                                 encoder: EmotionEncoder) -> None:
        anger = emotion_matrix[:, encoder.label_to_idx.get("anger", 0)]
        corr = np.corrcoef(anger, info_scores)[0, 1]
        logger.info("[Analysis] Correlation anger x informativeness: %.3f", corr)
