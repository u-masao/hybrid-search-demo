import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from src.model.embedding import Embedding

load_dotenv(".credential")

# Initialize Embedding class
embedding_model = Embedding(dimension=384)

print("load embeddin model")


def perform_vector_search(
    es_host, item_index_name, user_index_name, query_text, target_column, top_k=5
):
    global embedding_model

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

    def search_index(index_name):
        if not es.indices.exists(index=index_name):
            raise ValueError(f"Index '{index_name}' does not exist.")

        query_vector = embedding_model.generate_embedding(query_text)

        if target_column not in ["embedding", "translation"]:
            raise ValueError(
                "Invalid target column. Choose 'embedding' or 'translation'."
            )

        response = es.search(
            index=index_name,
            body={
                "size": top_k,
                "query": {
                    "knn": {
                        "field": target_column,
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
    search_body = {
        "query": {"match": {"content": query}},
        "size": top_k,
    }
    response = es.search(index=index_name, body=search_body)
    return response["hits"]["hits"]


def perform_bm25_search(es_host, item_index_name, user_index_name, query_text, top_k=5):
    # Initialize Elasticsearch client
    es = Elasticsearch(
        es_host,
        basic_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )

    def search_index(index_name):
        return bm25_search(es, index_name, query_text, top_k)

    item_results = search_index(item_index_name)
    user_results = search_index(user_index_name)

    return item_results, user_results


def search(es_host, index_name, query_text, target_column="embedding"):
    # Perform vector search
    vector_results = perform_vector_search(
        es_host, index_name, query_text, target_column
    )
    print("Vector Search Results:")
    for i, result in enumerate(vector_results):
        print(f"{i + 1}: {result['_source']['title'][:79]}")

    while True:
        try:
            selection = (
                int(
                    input(
                        "Select a result to view details"
                        " (0 to skip, -1 to exit): "
                    )
                )
                - 1
            )
            if selection == -2:
                break
            elif 0 <= selection < len(vector_results):
                print("\nSelected Vector Search Result Details:")
                print(
                    "Title: "
                    f"{vector_results[selection]['_source']['title']}"
                )
                print(
                    "Content: "
                    f"{vector_results[selection]['_source']['content']}"
                )
                input("\nPress Enter to continue...")
            elif selection == -1:
                print("Skipping detailed view.")
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Perform BM25 search
    bm25_results = perform_bm25_search(es_host, index_name, query_text)
    print("\nBM25 Search Results:")
    for i, result in enumerate(bm25_results):
        print(f"{i + 1}: {result['_source']['title'][:79]}")

    while True:
        try:
            selection = (
                int(
                    input(
                        "Select a result to view details "
                        "(0 to skip, -1 to exit): "
                    )
                )
                - 1
            )
            if selection == -2:
                break
            elif 0 <= selection < len(bm25_results):
                print("\nSelected BM25 Search Result Details:")
                print(f"Title: {bm25_results[selection]['_source']['title']}")
                print(
                    f"Content: {bm25_results[selection]['_source']['content']}"
                )
                input("\nPress Enter to continue...")
            elif selection == -1:
                print("Skipping detailed view.")
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Removed command-line interaction code
