"""Okapi BM25 over an inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np


class BM25Index:
    """BM25 scorer backed by an inverted index.

    Scoring a query costs ``O(|query| x postings per term)`` instead of
    ``O(N x |query|)``, because only documents containing a query term are
    visited.

    Args:
        corpus: one list of (stemmed) tokens per document.
        k1: term-frequency saturation.
        b: document-length normalisation.
    """

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)

        if self.n_docs == 0:
            self.avg_doc_len = 1.0
            self.doc_lengths: list[int] = []
            self.postings: dict[str, list[tuple[int, int]]] = {}
            self.idf: dict[str, float] = {}
            return

        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.n_docs

        # term -> [(doc_id, term frequency), ...]
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        for doc_id, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                postings[term].append((doc_id, tf))
                doc_freq[term] += 1

        self.postings = dict(postings)
        self.idf = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def get_scores(self, query_terms: Sequence[str]) -> np.ndarray:
        """BM25 score of every document (0 for documents sharing no term with the query)."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if not query_terms or self.n_docs == 0:
            return scores

        for term in query_terms:
            idf = self.idf.get(term, 0.0)
            if term not in self.postings or idf == 0.0:
                continue
            for doc_id, tf in self.postings[term]:
                norm = (1 - self.b) + self.b * (self.doc_lengths[doc_id] / self.avg_doc_len)
                scores[doc_id] += idf * ((self.k1 + 1) * tf) / (self.k1 * norm + tf)
        return scores

    def top_k(self, query_terms: Sequence[str], k: int) -> np.ndarray:
        """Indices of the ``k`` highest-scoring documents, best first."""
        return np.argsort(self.get_scores(query_terms))[::-1][:k]
