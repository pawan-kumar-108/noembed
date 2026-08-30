"""
TF-IDF vectorizer — no scikit-learn, no numpy. Pure Python arithmetic over
plain dicts.

See STDLIB.md: this replaces sklearn.feature_extraction.text.TfidfVectorizer.

Definitions used (standard, smoothed variant to avoid div-by-zero / log(0)):

    tf(term, doc)  = count of term in doc / total tokens in doc
    idf(term)      = ln( (1 + N) / (1 + df(term)) ) + 1
        where N = number of documents in the corpus,
              df(term) = number of documents containing term at least once
    tfidf(term, doc) = tf(term, doc) * idf(term)

The "+1" smoothing on both numerator and denominator of idf, plus the
trailing "+1", mirrors the standard smooth-idf formula (as used by
scikit-learn's default) so that a term appearing in every document still
gets a small positive weight instead of zero, and no term ever produces a
negative or undefined weight.
"""

import math
from collections import Counter

from src.tokenizer import tokenize


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """Raw term frequency, normalized by document length."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def document_frequencies(tokenized_docs: list[list[str]]) -> dict[str, int]:
    """
    For each term, how many documents (not how many times total) contain it.
    """
    df: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return df


def inverse_document_frequencies(
    df: dict[str, int], num_docs: int
) -> dict[str, float]:
    """Smoothed IDF per term, see module docstring for the formula."""
    return {
        term: math.log((1 + num_docs) / (1 + count)) + 1
        for term, count in df.items()
    }


class Vectorizer:
    """
    Fits a vocabulary + IDF table over a corpus, then produces sparse
    TF-IDF vectors (dict[term] -> weight) for individual documents.
    """

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.num_docs: int = 0

    def fit(self, raw_documents: list[str]) -> list[dict[str, float]]:
        """
        Fit the IDF table on a corpus of raw document strings, and return
        the TF-IDF vector for each document in the same order.
        """
        tokenized = [tokenize(doc) for doc in raw_documents]
        self.num_docs = len(tokenized)
        df = document_frequencies(tokenized)
        self.idf = inverse_document_frequencies(df, self.num_docs)
        return [self._vectorize(tokens) for tokens in tokenized]

    def transform(self, raw_document: str) -> dict[str, float]:
        """
        Vectorize a single document (e.g. a search query) against an
        already-fitted IDF table. Terms not seen during fit contribute
        nothing (their IDF is unknown, so they're dropped) — this is a
        documented limitation, not a bug.
        """
        tokens = tokenize(raw_document)
        return self._vectorize(tokens)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = term_frequencies(tokens)
        vector = {}
        for term, freq in tf.items():
            idf = self.idf.get(term)
            if idf is None:
                # Term unseen at fit time (only relevant for transform()
                # calls on new text, e.g. queries with novel words).
                continue
            vector[term] = freq * idf
        return vector

    def to_dict(self) -> dict:
        """Serializable state, for persistence."""
        return {"idf": self.idf, "num_docs": self.num_docs}

    @classmethod
    def from_dict(cls, data: dict) -> "Vectorizer":
        v = cls()
        v.idf = data["idf"]
        v.num_docs = data["num_docs"]
        return v