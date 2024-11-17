import hashlib
import os
import pytest

import numpy as np

from src.embedding.embedding import Embedding


@pytest.fixture
def embedding():
    return Embedding(dimension=384)


class TestEmbedding:
        self.embedding = Embedding(dimension=384)

    def test_split_text(self, embedding):
        # サンプルテキストでsplit_textメソッドをテスト
        text = (
            "This is a sample text to test the splitting functionality. " * 20
        )
        chunks = embedding.split_text(text, max_length=50, overlap=10)
        # 各チャンクが指定されたmax_length内であることを確認
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        # テキストが複数のチャンクに分割されていることを確認
        self.assertTrue(len(chunks) > 1)

    def test_generate_embedding(self, embedding):
        # generate_embeddingメソッドとキャッシュメカニズムをテスト
        text = "This is a test sentence."
        embedding = self.embedding.generate_embedding(text)
        # 埋め込みが正しい次元を持っていることを確認
        self.assertEqual(embedding.shape[0], self.embedding.dimension)

        # 埋め込みがキャッシュされているか確認
        cache_dir = ".cache/md5"
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")
        self.assertTrue(os.path.exists(cache_path))

    def test_compute_similarity(self, embedding):
        # 直交ベクトルでcompute_similarityメソッドをテスト
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        similarity = self.embedding.compute_similarity(vector1, vector2)
        # 直交ベクトルの類似度が0であることを確認
        self.assertAlmostEqual(similarity, 0.0)

        # 非直交ベクトルでcompute_similarityメソッドをテスト
        vector3 = np.array([1, 1, 0])
        similarity = self.embedding.compute_similarity(vector1, vector3)
        # 類似度が約0.707であることを確認
        self.assertAlmostEqual(similarity, 0.707, places=3)


if __name__ == "__main__":
    unittest.main()
