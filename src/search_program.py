import os
from dotenv import load_dotenv

load_dotenv(".credential")

from elasticsearch import Elasticsearch

from src.embedding.embedding import Embedding


def perform_vector_search(es_host, index_name, query_text, top_k=5):
    # Initialize Elasticsearch client
    elastic_username = os.getenv("ELASTIC_USERNAME")
    elastic_password = os.getenv("ELASTIC_PASSWORD")

    if not elastic_username or not elastic_password:
        raise ValueError("Elasticsearch username or password not set in environment variables.")

    es = Elasticsearch(
        [es_host],
        basic_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )

    # Check if the index exists
    if not es.indices.exists(index=index_name):
        raise ValueError(f"Index '{index_name}' does not exist.")

    # Initialize Embedding class
    embedding_model = Embedding(dimension=384)  # Assuming 384 dimensions

    # Generate embedding for the query text
    query_vector = embedding_model.generate_embedding(query_text)

    # Perform vector search using Elasticsearch's knn query
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": top_k
                }
            }
        }
    )
    return response["hits"]["hits"]


def bm25_search(es, index_name, query, top_k=5):
    """
    Perform BM25 search on the specified index.

    Parameters:
    - es: Elasticsearch client instance.
    - index_name: The name of the index to search.
    - query: The search query string.
    - top_k: The number of top results to return.

    Returns:
    A list of search results.
    """
    search_body = {"query": {"match": {"content": query}}, "size": top_k}
    response = es.search(index=index_name, body=search_body)
    return response["hits"]["hits"]

def perform_bm25_search(es_host, index_name, query_text, top_k=5):
    # Initialize Elasticsearch client
    es = Elasticsearch(
        es_host,
        basic_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )

    # Perform BM25 search
    return bm25_search(es, index_name, query_text, top_k)


def search(es_host, index_name, query_text):
    # Perform vector search
    vector_results = perform_vector_search(es_host, index_name, query_text)
    print("Vector Search Results:")
    for result in vector_results:
        print(result)

    # Perform BM25 search
    bm25_results = perform_bm25_search(es_host, index_name, query_text)
    print("\nBM25 Search Results:")
    for result in bm25_results:
        print(result)


if __name__ == "__main__":
    es_host = "https://localhost:9200"

    index_name = "article_data"
    query_text = "スマホ"
    search(es_host, index_name, query_text)

    index_name = "user_sentences"
    query_text = "スマホ"
    search(es_host, index_name, query_text)
