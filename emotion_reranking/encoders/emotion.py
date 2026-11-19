"""GoEmotions emotion classifier (RoBERTa)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from emotion_reranking.config import GOEMOTIONS_LABELS


class GoEmotionsEncoder:
    """Encodes each text as the 28-dimensional GoEmotions probability vector.

    Adapts a Hugging Face ``text-classification`` pipeline to the
    :class:`~emotion_reranking.encoders.base.EmotionEncoder` interface.
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions",
                 batch_size: int = 16, labels: Sequence[str] = GOEMOTIONS_LABELS) -> None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        print(f"[Encoder] GoEmotions running on {'GPU' if device == 0 else 'CPU'}.")
        self._classifier = pipeline("text-classification", model=model_name,
                                    top_k=None, truncation=True, device=device)
        self.batch_size = batch_size
        self.labels = tuple(labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._classifier(texts[start:start + self.batch_size], batch_size=self.batch_size)
            vectors.extend(self._to_vector(prediction) for prediction in batch)
        return np.vstack(vectors)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _to_vector(self, prediction: list[dict]) -> np.ndarray:
        vec = np.zeros(len(self.labels), dtype=np.float32)
        for item in prediction:
            idx = self.label_to_idx.get(item["label"].lower())
            if idx is not None:
                vec[idx] = item["score"]
        return vec
