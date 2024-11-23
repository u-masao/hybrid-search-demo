import pytest
import os
from elasticsearch import Elasticsearch
from src.search_app.es_utils import make_client, perform_vector_search

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

    assert len(item_results) <= top_k
    assert len(user_results) <= top_k
