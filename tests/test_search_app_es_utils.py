from elasticsearch import Elasticsearch

from src.search_app.es_utils import (
    get_item_info,
    get_user_info,
    make_client,
    perform_hybrid_search,
    perform_translation_search,
    perform_vector_search,
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

    item_results = perform_vector_search(
        query_vector=query_vector,
        index_name=item_index_name,
        field_name="embedding",
        top_k=top_k,
    )

    assert len(item_results) <= top_k

    user_results = perform_vector_search(
        query_vector=query_vector,
        index_name=user_index_name,
        field_name="embedding",
        top_k=top_k,
    )

    assert len(user_results) <= top_k


def test_get_user_info():
    user_id = "A1WNRJMBau9a4UogThL4"
    user_index_name = "user_develop"
    user_info = get_user_info(user_id, user_index_name)
    assert user_info is not None
    assert (
        "sentence" in user_info
    )  # Assuming 'name' is a field in the user document


def test_get_item_info():
    item_id = "HFWNRJMBau9a4UogZhKy"
    item_index_name = "item_develop"
    item_info = get_item_info(item_id, item_index_name)
    assert item_info is not None
    assert (
        "sentence" in item_info
    )  # Assuming 'title' is a field in the item document


def test_perform_user_translation_search():
    translation_vector = [0.1] * 64
    index_name = "user_develop"
    top_k = 5
    results = perform_translation_search(translation_vector, index_name, top_k)
    assert len(results) <= top_k


def test_perform_item_translation_search():
    translation_vector = [0.1] * 64
    index_name = "item_develop"
    top_k = 5
    results = perform_translation_search(translation_vector, index_name, top_k)
    assert len(results) <= top_k


def test_perform_hybrid_search():
    text = "スポーツ"
    text_vector = [0.1] * 384
    translation_vector = [0.1] * 64
    text_weight = 1.0
    text_vector_weight = 1.0
    translation_vector_weight = 1.0
    text_field_name = "sentence"
    text_vector_field_name = "embedding"
    translation_vector_field_name = "translation"
    index_name = "item_develop"
    top_k = 5

    results = perform_hybrid_search(
        text,
        text_vector,
        translation_vector,
        index_name,
        text_field_name=text_field_name,
        text_vector_field_name=text_vector_field_name,
        translation_vector_field_name=translation_vector_field_name,
        text_weight=text_weight,
        text_vector_weight=text_vector_weight,
        translation_vector_weight=translation_vector_weight,
        top_k=top_k,
    )
    assert len(results) <= top_k
