import hashlib
import os
import unittest

import numpy as np

from src.embedding.embedding import Embedding


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.embedding = Embedding(dimension=384)

    def test_split_text(self):
        # Test the split_text method with a sample text
        text = (
            "This is a sample text to test the splitting functionality. " * 20
        )
        chunks = self.embedding.split_text(text, max_length=50, overlap=10)
        # Assert that each chunk is within the specified max_length
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        # Assert that the text is split into multiple chunks
        self.assertTrue(len(chunks) > 1)

    def test_generate_embedding(self):
        # Test the generate_embedding method and caching mechanism
        text = "This is a test sentence."
        embedding = self.embedding.generate_embedding(text)
        # Assert that the embedding has the correct dimension
        self.assertEqual(embedding.shape[0], self.embedding.dimension)

        # Check if the embedding is cached
        cache_dir = ".cache/md5"
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")
        self.assertTrue(os.path.exists(cache_path))

    def test_compute_similarity(self):
        # Test the compute_similarity method with orthogonal vectors
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        similarity = self.embedding.compute_similarity(vector1, vector2)
        # Assert that the similarity is 0 for orthogonal vectors
        self.assertAlmostEqual(similarity, 0.0)

        # Test the compute_similarity method with non-orthogonal vectors
        vector3 = np.array([1, 1, 0])
        similarity = self.embedding.compute_similarity(vector1, vector3)
        # Assert that the similarity is approximately 0.707
        self.assertAlmostEqual(similarity, 0.707, places=3)


if __name__ == "__main__":
    unittest.main()
