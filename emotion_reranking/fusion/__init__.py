"""Early (feature-level) and late (rank-level) fusion of the text and emotion spaces."""

from emotion_reranking.fusion.early import EarlyFusion
from emotion_reranking.fusion.late import CombSumFusion, LateFusion, ReciprocalRankFusion

__all__ = ["CombSumFusion", "EarlyFusion", "LateFusion", "ReciprocalRankFusion"]
