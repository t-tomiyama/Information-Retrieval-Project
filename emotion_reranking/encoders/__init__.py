"""Text and emotion encoders.

The concrete encoders import their heavy dependencies (``torch``,
``transformers``, ``sentence-transformers``) only when instantiated.
"""

from emotion_reranking.encoders.base import EmotionEncoder, TextEncoder
from emotion_reranking.encoders.emotion import GoEmotionsEncoder
from emotion_reranking.encoders.text import SentenceBertEncoder

__all__ = ["EmotionEncoder", "GoEmotionsEncoder", "SentenceBertEncoder", "TextEncoder"]
