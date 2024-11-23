import click
from src.search_app.utils import make_client

@click.command()
@click.option('--es-host', required=True, help='Elasticsearch host URL.')
@click.option('--index-name', required=True, help='Name of the Elasticsearch index.')
@click.option('--query', required=True, help='Search query for the sentence, translation, or embedding fields.')
def search_db(es_host, index_name, query):
    """Search the Elasticsearch index for the given query in sentence, translation, and embedding fields."""
    es = make_client(es_host)

    # Construct the search query
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["sentence", "translation", "embedding"]
            }
        }
    }

    # Perform the search
    response = es.search(index=index_name, body=search_query)

    # Print the results
    for hit in response['hits']['hits']:
        print(hit['_source'])

if __name__ == '__main__':
    search_db()
