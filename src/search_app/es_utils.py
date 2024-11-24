"""
es_utils
Elasticsearchとのインタラクションを行うユーティリティ関数群
"""

import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(".credential")


def make_client(es_host):
    """
    Elasticsearchクライアントを作成する関数

    Parameters
    ----------
    es_host : str
        ElasticsearchのホストURL

    Returns
    -------
    Elasticsearch
        Elasticsearchクライアントインスタンス
    """
    elastic_username = os.getenv("ELASTIC_USERNAME")
    elastic_password = os.getenv("ELASTIC_PASSWORD")

    if not elastic_username or not elastic_password:
        raise ValueError(
            "Elasticsearch credentials not set in environment variables."
        )

    return Elasticsearch(
        [es_host],
        basic_auth=(elastic_username, elastic_password),
        ca_certs="certs/http_ca.crt",
    )


def get_user_info(user_id, user_index_name):
    """
    ユーザー情報を取得する関数

    Parameters
    ----------
    user_id : str
        ユーザーのID
    user_index_name : str
        ユーザー情報が格納されているインデックス名

    Returns
    -------
    dict
        ユーザー情報を含む辞書
    """
    es = make_client("https://localhost:9200")
    response = es.get(index=user_index_name, id=user_id)
    return response["_source"]


def get_item_info(item_id, item_index_name):
    """
    アイテム情報を取得する関数

    Parameters
    ----------
    item_id : str
        アイテムのID
    item_index_name : str
        アイテム情報が格納されているインデックス名

    Returns
    -------
    dict
        アイテム情報を含む辞書
    """
    es = make_client("https://localhost:9200")
    response = es.get(index=item_index_name, id=item_id)
    return response["_source"]


def perform_translation_search(translation_vector, index_name, top_k=5):
    """
    翻訳ベクトルを用いて検索を実行する関数

    Parameters
    ----------
    translation_vector : list
        検索に使用する翻訳ベクトル
    index_name : str
        検索対象のインデックス名
    top_k : int, optional
        取得する上位結果の数 (デフォルトは5)

    Returns
    -------
    list
        検索結果のリスト
    """
    es = make_client("https://localhost:9200")
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": "translation",
                    "query_vector": translation_vector,
                    "k": top_k,
                }
            },
        },
    )
    return response["hits"]["hits"]


def perform_vector_search(query_vector, index_name, field_name, top_k=5):
    """
    ベクトルを用いて検索を実行する関数

    Parameters
    ----------
    query_vector : list
        検索に使用するクエリベクトル
    index_name : str
        検索対象のインデックス名
    field_name : str
        検索対象のフィールド名
    top_k : int, optional
        取得する上位結果の数 (デフォルトは5)

    Returns
    -------
    list
        検索結果のリスト
    """
    es = make_client("https://localhost:9200")
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": field_name,
                    "query_vector": query_vector,
                    "k": top_k,
                }
            },
        },
    )
    return response["hits"]["hits"]


def perform_hybrid_search(
    query_text,
    query_text_vector,
    query_translation_vector,
    index_name,
    text_field_name,
    text_vector_field_name,
    translation_vector_field_name,
    text_weight: float = 1.0,
    text_vector_weight: float = 1.0,
    translation_vector_weight: float = 1.0,
    top_k=5,
):
    """
    ハイブリッド検索を実行する関数

    Parameters
    ----------
    query_text : str
        検索に使用するクエリテキスト
    query_text_vector : list
        検索に使用するテキストベクトル
    query_translation_vector : list
        検索に使用する翻訳ベクトル
    index_name : str
        検索対象のインデックス名
    text_field_name : str
        テキストフィールド名
    text_vector_field_name : str
        テキストベクトルフィールド名
    translation_vector_field_name : str
        翻訳ベクトルフィールド名
    text_weight : float, optional
        テキストの重み (デフォルトは1.0)
    text_vector_weight : float, optional
        テキストベクトルの重み (デフォルトは1.0)
    translation_vector_weight : float, optional
        翻訳ベクトルの重み (デフォルトは1.0)
    top_k : int, optional
        取得する上位結果の数 (デフォルトは5)

    Returns
    -------
    list
        検索結果のリスト
    """
    query_text,
    query_text_vector,
    query_translation_vector,
    index_name,
    text_field_name,
    text_vector_field_name,
    translation_vector_field_name,
    text_weight: float = 1.0,
    text_vector_weight: float = 1.0,
    translation_vector_weight: float = 1.0,
    top_k=5,
):
    es = make_client("https://localhost:9200")

    source = f"""
        double bm25_score = _score;
        double text_vector_score = cosineSimilarity(
            params.query_text_vector, '{text_vector_field_name}'
        );
        double translation_vector_score = cosineSimilarity(
            params.query_translation_vector, '{translation_vector_field_name}'
        );
        double weighted_average_score = (bm25_score * params.bm25_wieght) +
            (text_vector_score * params.text_vector_weight) +
            (translation_vector_score * params.translation_vector_weight);
        return weighted_average_score;
    """
    source = f"""
        double bm25_score = _score;
        double text_vector_score = 0.5 * (1.0 + cosineSimilarity(
            params.query_text_vector, '{text_vector_field_name}'
        ));
        double translation_vector_score = 0.5 *  (1.0 + cosineSimilarity(
            params.query_translation_vector, '{translation_vector_field_name}'
        ));
        double weighted_average_score = bm25_score * params.text_weight +
            text_vector_score * params.text_vector_weight +
            translation_vector_score * params.translation_vector_weight ;
        return weighted_average_score;
    """

    params = {
        "text_query": query_text,
        "query_text_vector": query_text_vector,
        "query_translation_vector": query_translation_vector,
        "text_weight": text_weight,
        "text_vector_weight": text_vector_weight,
        "translation_vector_weight": translation_vector_weight,
    }

    query = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": source,
                    "params": params,
                },
            }
        },
    }

    response = es.search(index=index_name, body=query, size=top_k)
    return response["hits"]["hits"]
