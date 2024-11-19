import os
from pprint import pprint

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from src.model.embedding import Embedding

load_dotenv(".credential")


# Initialize Embedding class
embedding_model = Embedding(dimension=384)


def make_client(es_host):

    # Initialize Elasticsearch client
    elastic_username = os.getenv("ELASTIC_USERNAME")
    elastic_password = os.getenv("ELASTIC_PASSWORD")

    if not elastic_username or not elastic_password:
        raise ValueError(
            "Elasticsearch username or password"
            " not set in environment variables."
        )

    es = Elasticsearch(
        [es_host],
        basic_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )
    return es


def bm25_search(es_host, index_name, query, top_k=5):
    es = make_client(es_host)

    body = {"query": {"match": {"sentence": query}}, "size": top_k}
    response = es.search(index=index_name, body=body)
    return response["hits"]["hits"]


def cosine_similarity_search(
    es_host, index_name, query_vector, top_k=5, vector_field="embedding"
):
    es = make_client(es_host)
    body = {
        "size": top_k,
        "query": {
            "knn": {
                "field": vector_field,
                "query_vector": query_vector,
                "k": top_k,
            }
        },
    }
    response = es.search(index=index_name, body=body)
    return response["hits"]["hits"]


def two_vector_search(
    es_host, index_name, query_text_vector, query_translation_vector, top_k=5
):

    es = make_client(es_host)
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_text_vector,
                            "k": top_k,
                        }
                    },
                    {
                        "knn": {
                            "field": "translation",
                            "query_vector": query_translation_vector,
                            "k": top_k,
                        }
                    },
                ]
            }
        },
    }
    response = es.search(index=index_name, body=body)
    return response["hits"]["hits"]


if __name__ == "__main__":

    query_text = "音楽"
    es_host = "https://localhost:9200"
    user_index = "user_develop"
    item_index = "item_develop"

    print("==== bm25 to user")
    result = bm25_search(es_host, user_index, query_text)
    pprint(result)

    print("==== vector to user embedding")
    query_text_vector = embedding_model.generate_embedding(query_text)
    result = cosine_similarity_search(
        es_host, user_index, query_text_vector, vector_field="embedding"
    )
    pprint(result)

    print("==== vector to user translation")
    query_translation_vector = result[1]["_source"]["translation"]
    result = cosine_similarity_search(
        es_host,
        user_index,
        query_translation_vector,
        vector_field="translation",
    )
    pprint(result)

    print("==== two vector search")
    result = two_vector_search(
        es_host, user_index, query_text_vector, query_translation_vector
    )
    pprint(result)
