"""End-to-end smoke test on a small synthetic corpus with fake encoders."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_reranking.config import ExperimentConfig
from emotion_reranking.pipeline import ExperimentPipeline, ExperimentResults
from tests.fakes import FakeEmotionEncoder, FakeTextEncoder

WORDS = ("refund delivery package broken great love quality price seller service "
         "late damaged helpful fast cheap scam account cancel worried perfect").split()


def make_corpus(path: Path, n: int = 150) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "Review Text": [" ".join(rng.choice(WORDS, size=rng.integers(5, 30))) for _ in range(n)],
        "Rating": [f"Rated {s} out of 5 stars" for s in rng.integers(1, 6, size=n)],
    }).to_csv(path, index=False)


class PipelineSmokeTest(unittest.TestCase):
    def test_run_produces_consistent_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_corpus(tmp / "reviews.csv")
            config = ExperimentConfig(data_path=tmp / "reviews.csv", output_dir=tmp / "out",
                                      n_docs=150, min_docs_per_class=10, bm25_top_k=20,
                                      n_data_driven_queries=28, n_bootstrap=50)
            results = ExperimentPipeline(config, emotion_encoder=FakeEmotionEncoder(),
                                         text_encoder=FakeTextEncoder(), make_figures=False).run()

            n_queries = results.metrics.query_id.nunique()
            self.assertEqual(n_queries, 16 + 28)
            # 2 text-only + 2 x 5 alphas + 2 late fusion rankings per query and relevance type.
            per_query = results.metrics.groupby(["query_id", "relevance_type"]).size()
            self.assertTrue((per_query == 14).all())
            self.assertTrue(results.metrics.ndcg10.between(0, 1).all())
            self.assertEqual(len(results.significance), 2 * 2 * 5)
            self.assertEqual(set(results.list_aggregation.algorithm), {"ListAgg-CombSUM", "ListAgg-RRF"})

            reloaded = ExperimentResults.load(config.output_dir)
            pd.testing.assert_frame_equal(reloaded.list_aggregation, results.list_aggregation)

    def test_figures_from_saved_results(self):
        from emotion_reranking.visualization import generate_figures

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_corpus(tmp / "reviews.csv")
            config = ExperimentConfig(data_path=tmp / "reviews.csv", output_dir=tmp / "out",
                                      n_docs=150, min_docs_per_class=10, bm25_top_k=20,
                                      query_mode="synthetic", n_bootstrap=50)
            results = ExperimentPipeline(config, emotion_encoder=FakeEmotionEncoder(),
                                         text_encoder=FakeTextEncoder(), make_figures=False).run()
            paths = generate_figures(results.metrics, results.significance, tmp / "figures")
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
