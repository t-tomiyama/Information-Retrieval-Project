"""Deterministic stand-ins for the transformer encoders (no model download needed)."""

import hashlib

import numpy as np

from emotion_reranking.config import GOEMOTIONS_LABELS


def _vector(text: str, dim: int, salt: str) -> np.ndarray:
    seed = int(hashlib.md5(f"{salt}|{text}".encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).random(dim).astype(np.float32)


class FakeEmotionEncoder:
    def __init__(self):
        self.label_to_idx = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}

    def encode(self, texts):
        vectors = np.vstack([_vector(t, len(GOEMOTIONS_LABELS), "emotion") for t in texts])
        return vectors / vectors.sum(axis=1, keepdims=True)

    def encode_one(self, text):
        return self.encode([text])[0]


class FakeTextEncoder:
    def encode(self, texts):
        return np.vstack([_vector(t, 16, "text") for t in texts])
