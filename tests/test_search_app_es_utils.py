import pytest
from unittest.mock import patch, MagicMock
from src.search_app.es_utils import (
    make_client,
    perform_vector_search,
    get_user_info,
    get_item_info,
    perform_translation_search,
)

@patch('src.search_app.es_utils.Elasticsearch')
def test_make_client(MockElasticsearch):
    mock_es = MockElasticsearch.return_value
    client = make_client("https://localhost:9200")
    assert client == mock_es
    MockElasticsearch.assert_called_once()

@patch('src.search_app.es_utils.make_client')
def test_perform_vector_search(mock_make_client):
    mock_es = MagicMock()
    mock_make_client.return_value = mock_es
    mock_es.search.return_value = {"hits": {"hits": ["result1", "result2"]}}

    item_results, user_results = perform_vector_search(
        query_vector=[0.1, 0.2, 0.3],
        item_index_name="items",
        user_index_name="users",
        top_k=5
    )

    assert item_results == ["result1", "result2"]
    assert user_results == ["result1", "result2"]

@patch('src.search_app.es_utils.make_client')
def test_get_user_info(mock_make_client):
    mock_es = MagicMock()
    mock_make_client.return_value = mock_es
    mock_es.get.return_value = {"_source": {"name": "test_user"}}

    user_info = get_user_info(user_id="123", user_index_name="users")
    assert user_info == {"name": "test_user"}

@patch('src.search_app.es_utils.make_client')
def test_get_item_info(mock_make_client):
    mock_es = MagicMock()
    mock_make_client.return_value = mock_es
    mock_es.get.return_value = {"_source": {"title": "test_item"}}

    item_info = get_item_info(item_id="456", item_index_name="items")
    assert item_info == {"title": "test_item"}

@patch('src.search_app.es_utils.make_client')
def test_perform_translation_search(mock_make_client):
    mock_es = MagicMock()
    mock_make_client.return_value = mock_es
    mock_es.search.return_value = {"hits": {"hits": ["result1", "result2"]}}

    results = perform_translation_search(
        translation_vector=[0.1, 0.2, 0.3],
        index_name="translations",
        top_k=5
    )

    assert results == ["result1", "result2"]
