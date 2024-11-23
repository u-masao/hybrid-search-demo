import pytest
import os
from elasticsearch import Elasticsearch
from src.search_app.es_utils import (
    make_client,
    perform_vector_search,
    get_user_info,
    get_item_info,
    perform_translation_search,
)

def test_make_client():
    es_host = "https://localhost:9200"
    client = make_client(es_host)
    assert isinstance(client, Elasticsearch)

def test_perform_vector_search():
    query_vector = [0.1, 0.2, 0.3]
    item_index_name = "items"
    user_index_name = "users"
    top_k = 5

    item_results, user_results = perform_vector_search(
        query_vector=query_vector,
        item_index_name=item_index_name,
        user_index_name=user_index_name,
        top_k=top_k
    )

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
