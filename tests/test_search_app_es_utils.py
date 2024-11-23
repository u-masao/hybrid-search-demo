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
    query_vector = [0.1] * 384
    item_index_name = "item_develop"
    user_index_name = "user_develop"
    top_k = 5

    item_results, user_results = perform_vector_search(
        query_vector=query_vector,
        item_index_name=item_index_name,
        user_index_name=user_index_name,
        top_k=top_k
    )

    assert len(item_results) <= top_k
    assert len(user_results) <= top_k
def test_get_user_info():
    user_id = 1
    user_index_name = "user_develop"
    user_info = get_user_info(user_id, user_index_name)
    assert user_info is not None
    assert "name" in user_info  # Assuming 'name' is a field in the user document

def test_get_item_info():
    item_id = 5625149
    item_index_name = "item_develop"
    item_info = get_item_info(item_id, item_index_name)
    assert item_info is not None
    assert "title" in item_info  # Assuming 'title' is a field in the item document

def test_perform_translation_search():
    translation_vector = [0.1] * 384
    index_name = "translation_develop"
    top_k = 5
    results = perform_translation_search(translation_vector, index_name, top_k)
    assert len(results) <= top_k
