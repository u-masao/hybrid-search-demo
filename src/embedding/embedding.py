import hashlib
import os
import pickle

import numpy as np
from transformers import AutoModel, AutoTokenizer


class Embedding:
    def __init__(self, dimension):
        self.dimension = dimension

    def split_text(self, text, max_length=512, overlap=50):
        """Split text into chunks of max_length with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunks.append(text[start:end])
            start += max_length - overlap
        return chunks

    def generate_embedding(self, text):
        """Generate embeddings for the given text using a pre-trained model with caching."""
        # Create cache directory if it doesn't exist
        cache_dir = ".cache/md5"
        os.makedirs(cache_dir, exist_ok=True)

        # Compute MD5 hash of the text
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")

        # Check if the embedding is already cached
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Generate embedding if not cached
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

        # Save the embedding to cache
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)

        return embedding
        """Generate a random embedding vector."""
        return np.random.rand(self.dimension)

    def compute_similarity(self, vector1, vector2):
        """Compute the cosine similarity between two vectors."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)
