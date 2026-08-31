"""
Inverted index + crash-safe persistence.

See STDLIB.md:
  - faiss / chromadb -> hand-rolled inverted index (this file)
  - SQLite / an embedded DB library -> hand-rolled JSON persistence with
    write-to-temp + fsync + os.replace() atomic rename

Durability guarantee (see README "Durability guarantee" for the full
statement): a save() call either fully lands on disk under the real index
path, or the real index path is left completely untouched. There is no
window where a partially-written file could be loaded as if it were valid,
because we only ever `os.replace()` a temp file that was itself fully
written and fsynced first. This protects against process crashes and
power loss between writes; it does NOT protect against disk-level bit rot
or a corrupted underlying filesystem — that's out of scope for this
project and is stated plainly in the README.
"""

import json
import os
import tempfile
from pathlib import Path

from src.similarity import cosine_similarity
from src.tfidf import Vectorizer


class NoEmbedIndex:
    """
    Holds a fitted Vectorizer, per-document TF-IDF vectors, an inverted
    index (term -> set of doc ids) for fast candidate lookup, and document
    metadata (path, original text length). Supports search, and crash-safe
    save/load to a single JSON index file.
    """

    def __init__(self) -> None:
        self.vectorizer = Vectorizer()
        self.doc_vectors: dict[str, dict[str, float]] = {}   # doc_id -> vector
        self.doc_meta: dict[str, dict] = {}                  # doc_id -> {"path": ..., "length": ...}
        self.inverted: dict[str, list[str]] = {}              # term -> [doc_id, ...]

    # ---- building -----------------------------------------------------

    def build(self, documents: dict[str, str]) -> None:
        """
        Build the index from scratch. `documents` maps a stable doc_id
        (e.g. a relative file path) to its raw text content.
        """
        doc_ids = list(documents.keys())
        raw_texts = [documents[doc_id] for doc_id in doc_ids]

        self.vectorizer = Vectorizer()
        vectors = self.vectorizer.fit(raw_texts)

        self.doc_vectors = {}
        self.doc_meta = {}
        self.inverted = {}

        for doc_id, text, vector in zip(doc_ids, raw_texts, vectors):
            self.doc_vectors[doc_id] = vector
            self.doc_meta[doc_id] = {"length": len(text)}
            for term in vector:
                self.inverted.setdefault(term, []).append(doc_id)

    # ---- searching ------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[tuple[str, float, dict[str, float]]]:
        """
        Return up to k (doc_id, score, query_vector) tuples, ranked by
        cosine similarity, highest first. Only documents that share at
        least one term with the query are scored — this is the actual
        "inverted index" payoff, not a brute-force scan of every document.

        query_vector is returned alongside each hit so callers (e.g. the
        CLI's --explain flag) can show which shared terms drove the score
        without recomputing it.
        """
        if not self.doc_vectors:
            return []

        query_vector = self.vectorizer.transform(query)
        if not query_vector:
            return []

        candidate_ids: set[str] = set()
        for term in query_vector:
            candidate_ids.update(self.inverted.get(term, []))

        scored = []
        for doc_id in candidate_ids:
            score = cosine_similarity(query_vector, self.doc_vectors[doc_id])
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(doc_id, score, query_vector) for doc_id, score in scored[:k]]

    def explain(self, doc_id: str, query_vector: dict[str, float]) -> list[tuple[str, float]]:
        """
        For --explain: which terms are shared between the query and this
        document, and each term's contribution (product of the two
        weights) to the dot product, sorted by contribution descending.
        """
        doc_vector = self.doc_vectors.get(doc_id, {})
        shared = [
            (term, query_vector[term] * doc_vector[term])
            for term in query_vector
            if term in doc_vector
        ]
        shared.sort(key=lambda pair: pair[1], reverse=True)
        return shared

    # ---- stats ----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "documents": len(self.doc_vectors),
            "vocabulary_size": len(self.inverted),
        }

    def top_terms(self, n: int = 10) -> list[tuple[str, float]]:
        """
        The n rarest terms in the corpus by IDF — the vocabulary the index
        considers most distinctive (as opposed to common words that appear
        everywhere and score low). Useful for sanity-checking what an
        index actually "knows" without reading raw JSON.
        """
        idf = self.vectorizer.idf
        return sorted(idf.items(), key=lambda pair: pair[1], reverse=True)[:n]

    def top_terms_for_doc(self, doc_id: str, n: int = 8) -> list[tuple[str, float]]:
        """
        The n highest-weighted terms within a single document's TF-IDF
        vector — effectively "what is this document about," according to
        the index.
        """
        vector = self.doc_vectors.get(doc_id, {})
        return sorted(vector.items(), key=lambda pair: pair[1], reverse=True)[:n]

    def document_ids(self) -> list[str]:
        return sorted(self.doc_vectors.keys())

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Crash-safe save: write full content to a temp file in the same
        directory as the target, fsync it, then os.replace() it over the
        real path. os.replace() is atomic on both POSIX and Windows, so
        the real index file is either the old complete version or the new
        complete version — never a half-written intermediate.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vectorizer": self.vectorizer.to_dict(),
            "doc_vectors": self.doc_vectors,
            "doc_meta": self.doc_meta,
            "inverted": self.inverted,
        }

        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)  # atomic on POSIX and Windows
        except BaseException:
            # If anything went wrong before the replace, clean up the temp
            # file so it doesn't linger; the real index path is untouched
            # either way, which is the actual durability guarantee.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self, path: str | Path) -> None:
        """Load a previously-saved index. Raises if the file is missing or invalid."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.vectorizer = Vectorizer.from_dict(payload["vectorizer"])
        self.doc_vectors = payload["doc_vectors"]
        self.doc_meta = payload["doc_meta"]
        self.inverted = payload["inverted"]

    @staticmethod
    def exists(path: str | Path) -> bool:
        return Path(path).exists()