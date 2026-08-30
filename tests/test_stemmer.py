import unittest

from src.stemmer import stem, stem_tokens


class StemmerTests(unittest.TestCase):
    def test_ing_suffix(self):
        self.assertEqual(stem("cooking"), "cook")

    def test_ed_suffix(self):
        self.assertEqual(stem("cooked"), "cook")

    def test_s_suffix(self):
        self.assertEqual(stem("cats"), "cat")

    def test_es_suffix(self):
        self.assertEqual(stem("matches"), "match")

    def test_ies_suffix(self):
        self.assertEqual(stem("companies"), "company")

    def test_ying_suffix(self):
        self.assertEqual(stem("trying"), "try")

    def test_cook_cooking_cooked_collapse_to_same_root(self):
        # The actual bug this stemmer was written to fix.
        self.assertEqual(stem("cook"), stem("cooking"))
        self.assertEqual(stem("cook"), stem("cooked"))

    def test_short_words_are_not_mangled(self):
        # "gas" ends in "s" but is only 3 chars -- the min-stem-length
        # guard must prevent it from becoming "ga".
        self.assertEqual(stem("gas"), "gas")
        self.assertEqual(stem("as"), "as")
        self.assertEqual(stem("is"), "is")

    def test_word_with_no_matching_suffix_unchanged(self):
        self.assertEqual(stem("space"), "space")
        self.assertEqual(stem("telescope"), "telescope")

    def test_ss_is_us_os_endings_exempt_from_s_rule(self):
        # Regression test: an earlier version of the "-s" rule turned
        # "this" into "thi", which is exactly the over-stemming failure
        # mode the module docstring warns about.
        self.assertEqual(stem("this"), "this")
        self.assertEqual(stem("class"), "class")
        self.assertEqual(stem("bus"), "bus")
        self.assertEqual(stem("chaos"), "chaos")

    def test_empty_string(self):
        self.assertEqual(stem(""), "")

    def test_stem_tokens_applies_to_whole_list(self):
        self.assertEqual(
            stem_tokens(["cooking", "cats", "space"]),
            ["cook", "cat", "space"],
        )


if __name__ == "__main__":
    unittest.main()