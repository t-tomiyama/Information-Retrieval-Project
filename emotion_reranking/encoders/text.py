"""Sentence-BERT semantic encoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceBertEncoder:
    """Dense text embeddings from a ``sentence-transformers`` model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        # Progress bar for corpus-sized batches only, not for single queries.
        return self._model.encode(texts, show_progress_bar=len(texts) > 1)
