import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(".credential")


def make_client(es_host):
    elastic_username = os.getenv("ELASTIC_USERNAME")
    elastic_password = os.getenv("ELASTIC_PASSWORD")

    if not elastic_username or not elastic_password:
        raise ValueError(
            "Elasticsearch credentials not set in environment variables."
        )

    return Elasticsearch(
        [es_host],
        basic_auth=(elastic_username, elastic_password),
        ca_certs="certs/http_ca.crt",
    )


def get_user_info(user_id, user_index_name):
    es = make_client("https://localhost:9200")
    response = es.get(index=user_index_name, id=user_id)
    return response["_source"]


def get_item_info(item_id, item_index_name):
    es = make_client("https://localhost:9200")
    response = es.get(index=item_index_name, id=item_id)
    return response["_source"]


def perform_translation_search(translation_vector, index_name, top_k=5):
    es = make_client("https://localhost:9200")
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": "translation",
                    "query_vector": translation_vector,
                    "k": top_k,
                }
            },
        },
    )
    return response["hits"]["hits"]


def perform_vector_search(query_vector, index_name, field_name, top_k=5):
    es = make_client("https://localhost:9200")
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": field_name,
                    "query_vector": query_vector,
                    "k": top_k,
                }
            },
        },
    )
    return response["hits"]["hits"]


def perform_hybrid_search(
    query_text,
    query_text_vector,
    query_translation_vector,
    index_name,
    text_field_name,
    text_vector_field_name,
    translation_vector_field_name,
    text_weight: float = 1.0,
    text_vector_weight: float = 1.0,
    translation_vector_weight: float = 1.0,
    top_k=5,
):
    es = make_client("https://localhost:9200")

    source = f"""
        double bm25_score = _score;
        double text_vector_score = cosineSimilarity(
            params.query_text_vector, '{text_vector_field_name}'
        );
        double translation_vector_score = cosineSimilarity(
            params.query_translation_vector, '{translation_vector_field_name}'
        );
        double weighted_average_score = (bm25_score * params.bm25_wieght) +
            (text_vector_score * params.text_vector_weight) +
            (translation_vector_score * params.translation_vector_weight);
        return weighted_average_score;
    """
    source = f"""
        double bm25_score = _score;
        double text_vector_score = 0.5 * (1.0 + cosineSimilarity(
            params.query_text_vector, '{text_vector_field_name}'
        ));
        double translation_vector_score = 0.5 *  (1.0 + cosineSimilarity(
            params.query_translation_vector, '{translation_vector_field_name}'
        ));
        double weighted_average_score = bm25_score * params.text_weight +
            text_vector_score * params.text_vector_weight +
            translation_vector_score * params.translation_vector_weight ;
        return weighted_average_score;
    """

    params = {
        "text_query": query_text,
        "query_text_vector": query_text_vector,
        "query_translation_vector": query_translation_vector,
        "text_weight": text_weight,
        "text_vector_weight": text_vector_weight,
        "translation_vector_weight": translation_vector_weight,
    }

    query = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": source,
                    "params": params,
                },
            }
        },
    }

    response = es.search(index=index_name, body=query, size=top_k)
    return response["hits"]["hits"]
