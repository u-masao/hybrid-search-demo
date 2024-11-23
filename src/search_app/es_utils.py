import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv(".credential")

def make_client(es_host):
    elastic_username = os.getenv("ELASTIC_USERNAME")
    elastic_password = os.getenv("ELASTIC_PASSWORD")

    if not elastic_username or not elastic_password:
        raise ValueError("Elasticsearch credentials not set in environment variables.")

    return Elasticsearch(
        [es_host],
        basic_auth=(elastic_username, elastic_password),
        ca_certs="certs/http_ca.crt",
    )

def perform_vector_search(query_vector, item_index_name, user_index_name, top_k=5):
    es = make_client("https://localhost:9200")

    def search_index(index_name):
        response = es.search(
            index=index_name,
            body={
                "size": top_k,
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "k": top_k,
                    }
                },
            },
        )
        return response["hits"]["hits"]

    item_results = search_index(item_index_name)
    user_results = search_index(user_index_name)
    return item_results, user_results

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
