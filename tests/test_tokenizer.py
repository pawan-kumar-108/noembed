import unittest

from src.tokenizer import tokenize


class TokenizerTests(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(tokenize("Hello World", remove_stopwords=False), ["hello", "world"])

    def test_strips_punctuation(self):
        self.assertEqual(
            tokenize("hello, world!", remove_stopwords=False), ["hello", "world"]
        )

    def test_splits_on_whitespace_runs(self):
        self.assertEqual(
            tokenize("hello    world\tfoo\nbar", remove_stopwords=False),
            ["hello", "world", "foo", "bar"],
        )

    def test_removes_stopwords_by_default(self):
        self.assertEqual(tokenize("this is the way"), ["way"])

    def test_keeps_stopwords_when_disabled(self):
        self.assertEqual(
            tokenize("this is the way", remove_stopwords=False),
            ["this", "is", "the", "way"],
        )

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])

    def test_non_ascii_preserved(self):
        # Documented limitation: non-ASCII text is lowercased and split on
        # whitespace, but not otherwise linguistically processed.
        self.assertEqual(tokenize("café résumé", remove_stopwords=False), ["café", "résumé"])

    def test_punctuation_only_input(self):
        self.assertEqual(tokenize("!!! ... ,,,"), [])

    def test_stems_by_default(self):
        self.assertEqual(
            tokenize("cooking cooked cooks", remove_stopwords=False),
            ["cook", "cook", "cook"],
        )

    def test_stemming_can_be_disabled(self):
        self.assertEqual(
            tokenize("cooking cooked", remove_stopwords=False, use_stemming=False),
            ["cooking", "cooked"],
        )

    def test_search_word_matches_document_word_via_stemming(self):
        # The concrete bug this feature fixes: a query for "cook" should
        # now share a token with a document containing "cooked".
        self.assertEqual(tokenize("cook")[0], tokenize("cooked")[0])


if __name__ == "__main__":
    unittest.main()