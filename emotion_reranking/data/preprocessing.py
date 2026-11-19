"""Text normalisation and tokenisation."""

from __future__ import annotations

import re
import string

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from emotion_reranking.config import EXTRA_STOPWORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | frozenset(EXTRA_STOPWORDS)

_STEMMER = PorterStemmer()
_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces and collapse whitespace."""
    text = _PUNCTUATION.sub(" ", str(text).lower())
    return _WHITESPACE.sub(" ", text).strip()


def tokenize_and_stem(text: str) -> list[str]:
    """Tokens for BM25: cleaned, stop words and tokens of length <= 2 removed, Porter-stemmed."""
    tokens = [t for t in clean_text(text).split() if t not in STOPWORDS and len(t) > 2]
    return [_STEMMER.stem(t) for t in tokens]
