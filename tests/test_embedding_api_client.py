import pytest
from unittest.mock import patch
from src.embedding_api.client import text_embedding

def test_text_embedding():
    with patch('src.embedding_api.client.requests.post') as mock_post:
        # Mock the response from the POST request
        mock_post.return_value.json.return_value = {'embedding': [0.1] * 384}

        text = "sample text"
        response = text_embedding(text)

        # Assert that the POST request was made with the correct URL and data
        mock_post.assert_called_once_with(
            "http://localhost:5000/api/vector_search",
            json={"query_text": text}
        )

        # Assert that the response is as expected
        assert response == {'embedding': [0.1] * 384}

        # Check that the embedding has the correct dimension
        assert len(response['embedding']) == 384
