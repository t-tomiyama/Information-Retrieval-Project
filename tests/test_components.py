import unittest

import numpy as np

from emotion_reranking.data.loader import HIGH_RATING, LOW_RATING, parse_stars, rating_category
from emotion_reranking.data.preprocessing import clean_text, tokenize_and_stem
from emotion_reranking.data.queries import Query
from emotion_reranking.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from emotion_reranking.fusion import CombSumFusion, EarlyFusion, ReciprocalRankFusion
from emotion_reranking.relevance import HybridEmotionRelevance, RatingRelevance, informativeness
from emotion_reranking.reranking import KReciprocalReranker, RLSimReranker
from emotion_reranking.retrieval import BM25Index


class PreprocessingTest(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  Great,  PRODUCT!! "), "great product")

    def test_tokenize_removes_stopwords_and_short_tokens(self):
        tokens = tokenize_and_stem("The Amazon delivery was absolutely amazing on my porch")
        self.assertNotIn("the", tokens)
        self.assertNotIn("amazon", tokens)
        self.assertIn("deliveri", tokens)

    def test_rating_parsing(self):
        self.assertEqual(parse_stars("Rated 4 out of 5 stars"), 4)
        self.assertEqual(rating_category("Rated 5 out of 5 stars"), HIGH_RATING)
        self.assertEqual(rating_category("2"), LOW_RATING)
        self.assertTrue(np.isnan(parse_stars("n/a")))


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([["refund", "late"], ["great", "product"], ["refund", "refund", "scam"]])

    def test_documents_without_query_terms_score_zero(self):
        scores = self.index.get_scores(["refund"])
        self.assertEqual(scores[1], 0.0)
        self.assertGreater(scores[2], scores[0])

    def test_empty_and_unknown_queries(self):
        np.testing.assert_array_equal(self.index.get_scores([]), np.zeros(3))
        np.testing.assert_array_equal(self.index.get_scores(["unknown"]), np.zeros(3))

    def test_top_k(self):
        self.assertEqual(self.index.top_k(["refund"], 2).tolist(), [2, 0])


class RerankerTest(unittest.TestCase):
    def test_rankings_are_permutations_and_exact_match_ranks_first(self):
        rng = np.random.default_rng(1)
        candidates = rng.normal(size=(30, 8))
        query = candidates[7] + 1e-6
        for reranker in (RLSimReranker(), KReciprocalReranker()):
            with self.subTest(reranker=reranker.name):
                ranking, scores = reranker.rerank(query, candidates)
                self.assertEqual(sorted(ranking.tolist()), list(range(30)))
                self.assertEqual(len(scores), 30)
                self.assertTrue(np.all(np.diff(scores) <= 1e-12))
        ranking, _ = KReciprocalReranker().rerank(query, candidates)
        self.assertEqual(ranking[0], 7)


class FusionTest(unittest.TestCase):
    def test_early_fusion_concatenates_scaled_emotion(self):
        fused = EarlyFusion(0.5).combine(np.ones((2, 3)), np.ones((2, 2)))
        np.testing.assert_array_equal(fused, [[1, 1, 1, 0.5, 0.5]] * 2)

    def test_combsum_normalises_each_list(self):
        fused = CombSumFusion().fuse([np.array([0.0, 5.0, 10.0]), np.array([3.0, 3.0, 3.0])])
        np.testing.assert_allclose(fused, [0.0, 0.5, 1.0])

    def test_rrf_uses_positions_only(self):
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse([np.array([0.9, 0.1, 0.5]), np.array([100.0, -5.0, 0.0])])
        np.testing.assert_allclose(fused, [2 / 61, 2 / 63, 2 / 62])
        self.assertEqual(rrf.rank([np.array([0.9, 0.1, 0.5])]).tolist(), [0, 2, 1])


class MetricsTest(unittest.TestCase):
    def test_perfect_and_worst_ranking(self):
        relevance = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(ndcg_at_k(np.array([1, 3, 0, 2]), relevance, k=2), 1.0)
        self.assertEqual(ndcg_at_k(np.array([0, 2, 1, 3]), relevance, k=2), 0.0)

    def test_precision_and_recall(self):
        relevance = np.array([1, 0, 1, 1])
        ranking = np.array([0, 1, 2, 3])
        self.assertAlmostEqual(precision_at_k(ranking, relevance, k=2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranking, relevance, k=2), 1 / 3)
        self.assertEqual(recall_at_k(ranking, np.zeros(4), k=2), 0.0)


class RelevanceTest(unittest.TestCase):
    def test_informativeness_bounds(self):
        self.assertEqual(informativeness(""), 0.0)
        score = informativeness("word " * 100)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_rating_relevance(self):
        judge = RatingRelevance([HIGH_RATING, LOW_RATING, HIGH_RATING])
        positive = Query("q", "x", ("joy",), "positive", "synthetic")
        neutral = Query("q", "x", ("neutral",), None, "synthetic")
        self.assertEqual(judge.judge(positive, np.array([0, 1, 2])).tolist(), [1, 0, 1])
        self.assertIsNone(judge.judge(neutral, np.array([0, 1])))

    def test_hybrid_relevance_uses_max_target_emotion(self):
        emotions = np.array([[0.9, 0.1], [0.2, 0.3]])
        judge = HybridEmotionRelevance(["", ""], emotions, {"anger": 0, "joy": 1}, aspects=())
        query = Query("q", "x", ("anger", "joy"), "negative", "synthetic")
        np.testing.assert_allclose(judge.judge(query, np.array([0, 1])), [0.45, 0.15])


if __name__ == "__main__":
    unittest.main()
