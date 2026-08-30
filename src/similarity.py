"""
Cosine similarity over sparse vectors — no numpy.

See STDLIB.md: this replaces numpy-based vector math
(e.g. sklearn.metrics.pairwise.cosine_similarity).
"""

import math


def dot_product(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Dot product over two sparse vectors represented as term -> weight
    dicts. Only shared keys contribute (weight is implicitly 0 for terms
    absent from a vector).
    """
    # Iterate the smaller dict for a small constant-factor speedup — pure
    # Python, so this isn't asymptotically important, just tidy.
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b[term] for term, weight in a.items() if term in b)


def magnitude(vector: dict[str, float]) -> float:
    """Euclidean norm of a sparse vector."""
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Cosine similarity in [0, 1] for TF-IDF vectors (which have
    non-negative weights, so the true range [-1, 1] never goes negative
    here). Returns 0.0 for either empty vector rather than raising, since
    an empty document/query has no defined direction to compare.
    """
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)