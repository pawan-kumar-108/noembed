import math
import unittest

from src.tfidf import (
    Vectorizer,
    document_frequencies,
    inverse_document_frequencies,
    term_frequencies,
)
from src.tokenizer import tokenize


class TermFrequencyTests(unittest.TestCase):
    def test_basic_counts(self):
        tf = term_frequencies(["a", "a", "b"])
        self.assertAlmostEqual(tf["a"], 2 / 3)
        self.assertAlmostEqual(tf["b"], 1 / 3)

    def test_empty_tokens(self):
        self.assertEqual(term_frequencies([]), {})


class DocumentFrequencyTests(unittest.TestCase):
    def test_counts_documents_not_occurrences(self):
        # "a" appears twice in doc 0 but that's still one document.
        docs = [["a", "a", "b"], ["a", "c"], ["c"]]
        df = document_frequencies(docs)
        self.assertEqual(df["a"], 2)
        self.assertEqual(df["b"], 1)
        self.assertEqual(df["c"], 2)


class InverseDocumentFrequencyTests(unittest.TestCase):
    def test_hand_verified_formula(self):
        # N = 3 documents. Term appears in 1 of them.
        # idf = ln((1+3)/(1+1)) + 1 = ln(2) + 1
        df = {"rare": 1}
        idf = inverse_document_frequencies(df, num_docs=3)
        self.assertAlmostEqual(idf["rare"], math.log(2) + 1)

    def test_term_in_every_document_still_positive(self):
        # N = 5, term appears in all 5 -> idf = ln(6/6) + 1 = ln(1) + 1 = 1
        df = {"common": 5}
        idf = inverse_document_frequencies(df, num_docs=5)
        self.assertAlmostEqual(idf["common"], 1.0)


class VectorizerTests(unittest.TestCase):
    def test_fit_produces_one_vector_per_document(self):
        docs = ["the cat sat", "the dog ran", "cats and dogs"]
        vec = Vectorizer()
        vectors = vec.fit(docs)
        self.assertEqual(len(vectors), 3)

    def test_rare_term_gets_higher_idf_than_common_term(self):
        docs = [
            "common common common rare",
            "common common common",
            "common common common",
        ]
        vec = Vectorizer()
        vec.fit(docs)
        # "rare" appears in only 1 of 3 docs, "common" in all 3 -- rare's
        # IDF must exceed common's. (Its final TF-IDF *weight* in a given
        # document also depends on term frequency within that document, so
        # this checks the IDF term directly rather than the blended weight.)
        self.assertGreater(vec.idf["rare"], vec.idf["common"])

    def test_equal_frequency_rare_term_outweighs_common_term(self):
        # Same term frequency in the document (1 each), so TF-IDF weight
        # differences here come purely from IDF -- this is where "rare
        # term outweighs common term" actually holds.
        docs = [
            "common rare",
            "common only",
            "common only",
        ]
        vec = Vectorizer()
        vectors = vec.fit(docs)
        self.assertIn("rare", vectors[0])
        self.assertIn("common", vectors[0])
        self.assertGreater(vectors[0]["rare"], vectors[0]["common"])

    def test_transform_drops_unseen_terms(self):
        vec = Vectorizer()
        vec.fit(["hello world"])
        query_vector = vec.transform("hello nonexistentterm")
        self.assertIn("hello", query_vector)
        self.assertNotIn("nonexistentterm", query_vector)

    def test_serialization_round_trip(self):
        vec = Vectorizer()
        vec.fit(["hello world", "goodbye world"])
        restored = Vectorizer.from_dict(vec.to_dict())
        self.assertEqual(vec.idf, restored.idf)
        self.assertEqual(vec.num_docs, restored.num_docs)


if __name__ == "__main__":
    unittest.main()