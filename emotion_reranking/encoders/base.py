"""Encoder interfaces.

The pipeline depends on these protocols only, so the transformer models can be
swapped (or replaced by test doubles) without touching the experiment code.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class TextEncoder(Protocol):
    """Maps texts to dense semantic vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class EmotionEncoder(Protocol):
    """Maps texts to a probability vector over a fixed set of emotion labels."""

    label_to_idx: Mapping[str, int]

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...
