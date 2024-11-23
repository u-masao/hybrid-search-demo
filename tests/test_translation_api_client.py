import pytest
from unittest.mock import patch
from src.translation_api.client import translation_user, translation_item

def test_translation_user():
    with patch('src.translation_api.client.requests.post') as mock_post:
        # Mock the response from the POST request
        mock_post.return_value.json.return_value = {'translation': 'mock_translation'}

        user_id = 123
        response = translation_user(user_id)

        # Assert that the POST request was made with the correct URL and data
        mock_post.assert_called_once_with(
            "http://localhost:5000/api/user_translation_search",
            json={"user_id": user_id}
        )

        # Assert that the response is as expected
        assert response == {'translation': 'mock_translation'}

def test_translation_item():
    with patch('src.translation_api.client.requests.post') as mock_post:
        # Mock the response from the POST request
        mock_post.return_value.json.return_value = {'translation': 'mock_translation'}

        item_id = 456
        response = translation_item(item_id)

        # Assert that the POST request was made with the correct URL and data
        mock_post.assert_called_once_with(
            "http://localhost:5000/api/item_translation_search",
            json={"item_id": item_id}
        )

        # Assert that the response is as expected
        assert response == {'translation': 'mock_translation'}
