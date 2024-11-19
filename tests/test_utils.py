import unittest
from src.search_app.utils import bm25_search, cosine_similarity_search

class TestElasticsearchUtils(unittest.TestCase):

    def setUp(self):
        self.es_host = "http://localhost:9200"
        self.user_index = "user_develop"
        self.item_index = "item_develop"
        self.query_text = "example query"
        self.query_vector = [0.1] * 384  # Example vector

    def test_bm25_search_user_index(self):
        results = bm25_search(self.es_host, self.user_index, self.query_text, top_k=5)
        self.assertIsInstance(results, list)

    def test_bm25_search_item_index(self):
        results = bm25_search(self.es_host, self.item_index, self.query_text, top_k=5)
        self.assertIsInstance(results, list)

    def test_cosine_similarity_search_user_index(self):
        results = cosine_similarity_search(self.es_host, self.user_index, self.query_vector, top_k=5)
        self.assertIsInstance(results, list)

    def test_cosine_similarity_search_item_index(self):
        results = cosine_similarity_search(self.es_host, self.item_index, self.query_vector, top_k=5)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
