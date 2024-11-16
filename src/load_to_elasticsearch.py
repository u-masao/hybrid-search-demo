import pandas as pd
from elasticsearch import Elasticsearch, helpers

def load_data_to_elasticsearch(es_host, index_name, data_file):
    es = Elasticsearch([es_host])
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
