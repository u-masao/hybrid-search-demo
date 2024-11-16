import hashlib
import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class Embedding:
    def __init__(self, dimension):
        self.dimension = dimension

    def split_text(self, text, max_length=512, overlap=50):
        """
        テキストをmax_lengthのチャンクに分割し、重複させます。

        パラメータ:
        - text: 分割する入力テキスト。
        - max_length: 各テキストチャンクの最大長。
        - overlap: チャンク間の重複文字数。

        戻り値:
        テキストチャンクのリスト。
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append(text[start:end])
            start += max_length - overlap
        return chunks

    def generate_embedding(self, text):
        """
        キャッシュを使用して事前学習済みモデルで指定されたテキストの埋め込みを生成します。

        パラメータ:
        - text: 埋め込みを生成する入力テキスト。

        戻り値:
        入力テキストの埋め込みを表すnumpy配列。
        """
        # キャッシュディレクトリが存在しない場合は作成
        cache_dir = ".cache/md5"
        os.makedirs(cache_dir, exist_ok=True)

        # テキストのMD5ハッシュを計算
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")

        # 埋め込みが既にキャッシュされているか確認
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # キャッシュされていない場合は埋め込みを生成
        tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-small"
        )
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        outputs = model(**inputs)
        embedding = (
            outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        )

        # 埋め込みをキャッシュに保存
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)

        return embedding
        """
        ランダムな埋め込みベクトルを生成します。

        戻り値:
        ランダムな埋め込みベクトルを表すnumpy配列。
        """
        return np.random.rand(self.dimension)

    def vector_search(self, query_vector, vectors, top_k=5):
        """
        Perform vector search using cosine similarity.

        Parameters:
        - query_vector: The vector representation of the query.
        - vectors: A list of vectors to search against.
        - top_k: The number of top results to return.

        Returns:
        A list of indices of the top_k most similar vectors.
        """
        similarities = cosine_similarity([query_vector], vectors)
        return np.argsort(similarities[0])[-top_k:][::-1]

    def compute_similarity(self, vector1, vector2):
        """
        2つのベクトル間のコサイン類似度を計算します。

        パラメータ:
        - vector1: 類似度計算のための最初のベクトル。
        - vector2: 類似度計算のための2番目のベクトル。

        戻り値:
        vector1とvector2の間のコサイン類似度スコア。
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)
