import unittest

from src.similarity import cosine_similarity, dot_product, magnitude


class DotProductTests(unittest.TestCase):
    def test_only_shared_keys_contribute(self):
        a = {"x": 2.0, "y": 3.0}
        b = {"y": 4.0, "z": 5.0}
        # only "y" is shared: 3.0 * 4.0 = 12.0
        self.assertAlmostEqual(dot_product(a, b), 12.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(dot_product({"a": 1.0}, {"b": 1.0}), 0.0)

    def test_symmetry(self):
        a = {"x": 1.5, "y": 2.5}
        b = {"x": 3.0, "y": 0.5, "z": 9.0}
        self.assertAlmostEqual(dot_product(a, b), dot_product(b, a))


class MagnitudeTests(unittest.TestCase):
    def test_simple_vector(self):
        # sqrt(3^2 + 4^2) = 5
        self.assertAlmostEqual(magnitude({"x": 3.0, "y": 4.0}), 5.0)

    def test_empty_vector(self):
        self.assertAlmostEqual(magnitude({}), 0.0)


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        v = {"a": 1.0, "b": 2.0, "c": 3.0}
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_disjoint_vocabulary_scores_zero(self):
        a = {"x": 1.0, "y": 1.0}
        b = {"z": 1.0, "w": 1.0}
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_empty_vector_scores_zero_not_error(self):
        self.assertAlmostEqual(cosine_similarity({}, {"a": 1.0}), 0.0)
        self.assertAlmostEqual(cosine_similarity({"a": 1.0}, {}), 0.0)
        self.assertAlmostEqual(cosine_similarity({}, {}), 0.0)

    def test_symmetry(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 2.0, "y": 1.0, "z": 5.0}
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a))

    def test_scaled_vectors_still_score_one(self):
        # Cosine similarity is scale-invariant: doubling every weight in a
        # vector shouldn't change its similarity to itself's direction.
        a = {"x": 1.0, "y": 1.0}
        b = {"x": 2.0, "y": 2.0}
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()