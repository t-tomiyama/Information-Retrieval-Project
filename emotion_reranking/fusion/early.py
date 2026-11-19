"""Early fusion: combine representations before re-ranking."""

from __future__ import annotations

import numpy as np


class EarlyFusion:
    """Concatenates the text vector with the emotion vector scaled by ``alpha``.

    Works for a single vector (1-D) or a matrix of row vectors (2-D).
    ``alpha`` controls how much the emotion space contributes to distances.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def combine(self, text_vecs: np.ndarray, emotion_vecs: np.ndarray) -> np.ndarray:
        return np.hstack([text_vecs, self.alpha * emotion_vecs])
