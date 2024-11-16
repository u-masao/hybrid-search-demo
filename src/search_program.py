import os
from elasticsearch import Elasticsearch
from src.embedding.embedding import Embedding
from src.load_to_elasticsearch import bm25_search

def perform_vector_search(es_host, index_name, query_text, top_k=5):
    # Initialize Elasticsearch client
    es = Elasticsearch(
        es_host,
        http_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )

    # Initialize Embedding class
    embedding_model = Embedding(dimension=768)  # Assuming 768 dimensions

    # Generate embedding for the query text
    query_vector = embedding_model.generate_embedding(query_text)

    # Retrieve all vectors from the index
    response = es.search(index=index_name, body={"query": {"match_all": {}}})
    vectors = [hit["_source"]["embedding"] for hit in response["hits"]["hits"]]

    # Perform vector search
    top_indices = embedding_model.vector_search(query_vector, vectors, top_k)
    return [response["hits"]["hits"][i] for i in top_indices]

def perform_bm25_search(es_host, index_name, query_text, top_k=5):
    # Initialize Elasticsearch client
    es = Elasticsearch(
        es_host,
        http_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )

    # Perform BM25 search
    return bm25_search(es, index_name, query_text, top_k)

if __name__ == "__main__":
    es_host = "https://localhost:9200"
    index_name = "articles"
    query_text = "example search text"

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
