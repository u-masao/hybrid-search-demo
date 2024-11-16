import pandas as pd
import os
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv('.credential')

def load_data_to_elasticsearch(es_host, index_name, data_file):
    es = Elasticsearch(
        [es_host],
        http_auth=(os.getenv('ELASTIC_USERNAME'), os.getenv('ELASTIC_PASSWORD')),
        scheme="https",
        port=9200,
        ca_certs="certs/http_ca.crt"
    )
    df = pd.read_parquet(data_file)

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
    @click.option('--es-host', default='localhost:9200', help='Elasticsearch host')
    @click.option('--index-name', default='user_data', help='Elasticsearch index name')
    @click.argument('data_file', type=click.Path(exists=True))
    def main(es_host, index_name, data_file):
        load_data_to_elasticsearch(es_host, index_name, data_file)

    main()
