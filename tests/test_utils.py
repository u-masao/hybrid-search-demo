import pytest

from src.search_app.utils import bm25_search, cosine_similarity_search


@pytest.fixture
def es_config(dims=384):
    return {
        "es_host": "https://localhost:9200",
        "user_index": "user_develop",
        "item_index": "item_develop",
        "query_text": "example query",
        "query_vector": [0.1] * dims,  # Example vector
    }


def test_bm25_search_user_index(es_config):
    results = bm25_search(
        es_config["es_host"],
        es_config["user_index"],
        es_config["query_text"],
        top_k=5,
    )
    assert isinstance(results, list)


def test_bm25_search_item_index(es_config):
    results = bm25_search(
        es_config["es_host"],
        es_config["item_index"],
        es_config["query_text"],
        top_k=5,
    )
    assert isinstance(results, list)


def test_cosine_similarity_search_user_index(es_config):
    results = cosine_similarity_search(
        es_config["es_host"],
        es_config["user_index"],
        es_config["query_vector"],
        top_k=5,
    )
    assert isinstance(results, list)


def test_cosine_similarity_search_item_index(es_config):
    results = cosine_similarity_search(
        es_config["es_host"],
        es_config["item_index"],
        es_config["query_vector"],
        top_k=5,
    )
    assert isinstance(results, list)


def test_cosine_similarity_search_transition(es_config):
    results = cosine_similarity_search(
        es_config["es_host"],
        es_config["user_index"],
        [0.1] * 64,
        top_k=5,
        vector_field="transition",
    )
    assert isinstance(results, list)

    results = cosine_similarity_search(
        es_config["es_host"],
        es_config["item_index"],
        [0.0] * 64,
        top_k=5,
        vector_field="transition",
    )
    assert isinstance(results, list)
