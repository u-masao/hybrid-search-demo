from src.model.embedding import Embedding

# Initialize Embedding class
embedding_model = Embedding(dimension=384)

def text_embedding(text):
    return embedding_model.generate_embedding(text)
