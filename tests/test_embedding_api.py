import pytest
from src.embedding_api.embedding_service import EmbeddingService

@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()

def test_embed(embedding_service):
    embedding = embedding_service.embed("test query")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
