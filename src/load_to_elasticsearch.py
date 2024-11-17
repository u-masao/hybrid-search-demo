import os

import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv(".credential")


def load_data_to_elasticsearch(es_host, index_name, embedding_file):
    print(f"{es_host=}")
    es = Elasticsearch(
        es_host,
        http_auth=(
            os.getenv("ELASTIC_USERNAME"),
            os.getenv("ELASTIC_PASSWORD"),
        ),
        ca_certs="certs/http_ca.crt",
    )
    # Delete the index if it exists to ensure all data is overwritten
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    df = pd.read_parquet(embedding_file)

    actions = [
        {
            "_index": index_name,
            "_source": row.to_dict(),
        }
        for _, row in df.iterrows()
    ]

    helpers.bulk(es, actions)


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


if __name__ == "__main__":
    import click

    @click.command()
    @click.option(
        "--es_host",
        default="https://localhost:9200",
        help="Elasticsearch host URL (e.g., https://localhost:9200)",
    )
    @click.option(
        "--index_name", default="user_data", help="Elasticsearch index name"
    )
    @click.argument("embedding_file", type=click.Path(exists=True))
    def main(es_host, index_name, embedding_file):
        load_data_to_elasticsearch(es_host, index_name, embedding_file)

    main()
