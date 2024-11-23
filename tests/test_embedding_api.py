import pytest
import requests

BASE_URL = "http://localhost:5001"

from src.embedding_api.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_embed():
    response = requests.post(f"{BASE_URL}/embed", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    embedding = data.get("embedding", [])
    assert isinstance(embedding, list)
    assert len(embedding) == 384
