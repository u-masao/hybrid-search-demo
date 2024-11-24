from src.embedding_api.client import text_embedding


def test_text_embedding():
    text = "sample text"
    response = text_embedding(text)

    # Assert that the response contains an embedding of the correct dimension
    assert "embedding" in response
    assert len(response["embedding"]) == 384
