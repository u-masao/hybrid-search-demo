import numpy as np

class Embedding:
    def __init__(self, dimension):
        self.dimension = dimension

    def generate_random_embedding(self):
        """Generate a random embedding vector."""
        return np.random.rand(self.dimension)

    def compute_similarity(self, vector1, vector2):
        """Compute the cosine similarity between two vectors."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)
