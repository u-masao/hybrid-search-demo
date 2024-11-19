from elasticsearch import Elasticsearch
import numpy as np

def bm25_search(es_host, index_name, query, top_k=5):
    es = Elasticsearch(es_host)
    body = {
        "query": {
            "match": {
                "sentence": query
            }
        },
        "size": top_k
    }
    response = es.search(index=index_name, body=body)
    return response['hits']['hits']

def cosine_similarity_search(es_host, index_name, query_vector, top_k=5, vector_field="embedding"):
    es = Elasticsearch(es_host)
    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc[params.vector_field]) + 1.0",
                    "params": {
                        "query_vector": query_vector,
                        "vector_field": vector_field
                    }
                }
            }
        },
        "size": top_k
    }
    response = es.search(index=index_name, body=body)
    return response['hits']['hits']
