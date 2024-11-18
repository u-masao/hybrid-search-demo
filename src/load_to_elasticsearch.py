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
    # Define the index mapping with dense_vector fields
    index_mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384 # Adjust the dimension as needed
                },
                "translation": {
                    "type": "dense_vector",
                    "dims": 64  # Adjust the dimension as needed
                }
            }
        }
    }
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Create the index with the specified mapping
    es.indices.create(index=index_name, body=index_mapping)

    df = pd.read_parquet(embedding_file)

    print(f"Uploading data with columns: {df.columns.tolist()}")

    actions = [
        {
            "_index": index_name,
            "_source": row.to_dict(),
        }
        for _, row in df.iterrows()
    ]

    helpers.bulk(es, actions)


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
