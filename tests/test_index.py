import tempfile
import unittest
from pathlib import Path

from src.index import NoEmbedIndex


class BuildAndSearchTests(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "cats.txt": "cats are wonderful pets and love to nap",
            "dogs.txt": "dogs are loyal pets and love to play fetch",
            "cars.txt": "electric cars use batteries instead of gasoline engines",
        }
        self.index = NoEmbedIndex()
        self.index.build(self.docs)

    def test_search_ranks_relevant_doc_first(self):
        results = self.index.search("pets that love to nap", k=3)
        self.assertTrue(results)
        top_doc_id = results[0][0]
        self.assertEqual(top_doc_id, "cats.txt")

    def test_search_excludes_unrelated_doc_from_top(self):
        results = self.index.search("pets", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn(results[0][0], {"cats.txt", "dogs.txt"})

    def test_search_respects_k(self):
        results = self.index.search("pets cars engines", k=1)
        self.assertLessEqual(len(results), 1)

    def test_empty_query_returns_no_results(self):
        # An empty/whitespace query tokenizes to nothing, so there is no
        # query vector to score against -- this is a deliberate no-match,
        # not an error path (the CLI layer rejects empty queries earlier).
        results = self.index.search("   ", k=5)
        self.assertEqual(results, [])

    def test_search_on_empty_index_returns_no_results(self):
        empty_index = NoEmbedIndex()
        results = empty_index.search("anything", k=5)
        self.assertEqual(results, [])

    def test_single_document_corpus(self):
        single = NoEmbedIndex()
        single.build({"only.txt": "the only document here"})
        results = single.search("only document", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "only.txt")

    def test_duplicate_documents(self):
        dup = NoEmbedIndex()
        dup.build({"a.txt": "identical text here", "b.txt": "identical text here"})
        results = dup.search("identical text", k=5)
        self.assertEqual(len(results), 2)
        # Both should score identically since the content is identical.
        self.assertAlmostEqual(results[0][1], results[1][1])

    def test_stats(self):
        stats = self.index.stats()
        self.assertEqual(stats["documents"], 3)
        self.assertGreater(stats["vocabulary_size"], 0)

    def test_top_terms_returns_highest_idf_first(self):
        top = self.index.top_terms(n=5)
        self.assertTrue(top)
        idfs = [idf for _, idf in top]
        self.assertEqual(idfs, sorted(idfs, reverse=True))

    def test_top_terms_respects_n(self):
        top = self.index.top_terms(n=2)
        self.assertLessEqual(len(top), 2)

    def test_top_terms_for_doc_returns_highest_weight_first(self):
        top = self.index.top_terms_for_doc("cats.txt", n=5)
        self.assertTrue(top)
        weights = [w for _, w in top]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_top_terms_for_unknown_doc_returns_empty(self):
        self.assertEqual(self.index.top_terms_for_doc("nonexistent.txt"), [])

    def test_document_ids_returns_all_docs_sorted(self):
        self.assertEqual(
            self.index.document_ids(), sorted(["cats.txt", "dogs.txt", "cars.txt"])
        )


class PersistenceTests(unittest.TestCase):
    def test_save_then_load_round_trip(self):
        docs = {"a.txt": "hello world", "b.txt": "goodbye world"}
        index = NoEmbedIndex()
        index.build(docs)

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            index.save(index_path)

            reloaded = NoEmbedIndex()
            reloaded.load(index_path)

            self.assertEqual(reloaded.stats(), index.stats())
            original_results = index.search("hello", k=5)
            reloaded_results = reloaded.search("hello", k=5)
            self.assertEqual(
                [r[0] for r in original_results], [r[0] for r in reloaded_results]
            )

    def test_load_missing_file_raises(self):
        index = NoEmbedIndex()
        with self.assertRaises(OSError):
            index.load("/tmp/definitely-does-not-exist-noembed-index.json")

    def test_exists_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            self.assertFalse(NoEmbedIndex.exists(path))
            index = NoEmbedIndex()
            index.build({"a.txt": "hello"})
            index.save(path)
            self.assertTrue(NoEmbedIndex.exists(path))


if __name__ == "__main__":
    unittest.main()