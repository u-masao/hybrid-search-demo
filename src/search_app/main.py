"""
search_app.main
Flaskアプリケーションのエントリーポイント
"""

import io
import random

from flask import Flask, jsonify, make_response, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

from src.embedding_api.client import text_embedding
from src.search_app.es_utils import (
    get_item_info,
    get_user_info,
    perform_hybrid_search,
    perform_translation_search,
    perform_vector_search,
)

app = Flask(__name__)  # Flaskアプリケーションのインスタンスを作成

# Elasticsearchのインデックス名
item_index_name = "item_develop"
user_index_name = "user_develop"

text_embedding_field_name = "embedding"
translated_field_name = "translation"


@app.route("/api/text_embedding", methods=["POST"])
def encode_text():
    """
    テキストをエンコードして埋め込みベクトルを返すAPIエンドポイント

    Returns
    -------
    flask.Response
        埋め込みベクトルを含むJSONレスポンス
    """
def encode_text():
    data = request.json
    query_text = data.get("query_text")
    embedding = text_embedding(query_text)
    if "embedding" not in embedding:
        return jsonify({"status": "error"})
    return jsonify({"embedding": embedding["embedding"]})


@app.route("/api/vector_search", methods=["POST"])
def vector_search():
    """
    ベクトル検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        アイテムとユーザーの検索結果を含むJSONレスポンス
    """
def vector_search():
    data = request.json
    query_text = data.get("query_text")
    print(query_text)
    query_vector = text_embedding(query_text)
    if "embedding" not in query_vector:
        return jsonify({"status": "error"})
    print(query_vector)
    item_results = perform_vector_search(
        query_vector["embedding"], item_index_name, text_embedding_field_name
    )
    user_results = perform_vector_search(
        query_vector["embedding"], user_index_name, text_embedding_field_name
    )
    return jsonify(
        {"item_results": item_results, "user_results": user_results}
    )


@app.route("/api/user_info/", methods=["POST"])
def user_info():
    """
    ユーザー情報を取得するAPIエンドポイント

    Returns
    -------
    flask.Response
        ユーザー情報を含むJSONレスポンス
    """
def user_info():
    data = request.json
    user_id = data.get("id")
    user_data = get_user_info(user_id, user_index_name)
    return jsonify(user_data)


@app.route("/api/item_info/", methods=["POST"])
def item_info():
    """
    アイテム情報を取得するAPIエンドポイント

    Returns
    -------
    flask.Response
        アイテム情報を含むJSONレスポンス
    """
def item_info():
    data = request.json
    item_id = data.get("id")
    item_data = get_item_info(item_id, item_index_name)
    return jsonify(item_data)


@app.route("/api/user_translation_search", methods=["POST"])
def user_translation_search():
    """
    ユーザー翻訳検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def user_translation_search():
    data = request.json
    translation_vector = data.get("translation")
    results = perform_translation_search(translation_vector, user_index_name)
    return jsonify(results)


@app.route("/api/item_translation_search", methods=["POST"])
def item_translation_search():
    """
    アイテム翻訳検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def item_translation_search():
    data = request.json
    translation_vector = data.get("translation")
    results = perform_translation_search(translation_vector, item_index_name)
    return jsonify(results)


@app.route("/api/user_text_embedding_search", methods=["POST"])
def user_text_embedding_search():
    """
    ユーザーテキスト埋め込み検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def user_text_embedding_search():
    data = request.json
    query_vector = data.get("embedding")
    results = perform_vector_search(
        query_vector, user_index_name, text_embedding_field_name
    )
    return jsonify(results)


@app.route("/api/item_text_embedding_search", methods=["POST"])
def item_text_embedding_search():
    """
    アイテムテキスト埋め込み検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def item_text_embedding_search():
    data = request.json
    query_vector = data.get("embedding")
    results = perform_vector_search(
        query_vector, item_index_name, text_embedding_field_name
    )
    return jsonify(results)


def parse_hybrid_search_input():
    """
    ハイブリッド検索の入力を解析する関数

    Returns
    -------
    tuple
        クエリテキスト、クエリベクトル、重みなどのパラメータを含むタプル
    """
    data = request.json
    print(data)
    query_text = data.get("text", None)
    query_text_vector = data.get("embedding", [0.1] * 384)
    query_translation_vector = data.get("translation", [0.1] * 64)
    text_weight = data.get("text_weight", 0.0)
    text_vector_weight = data.get("text_vector_weight", 0.0)
    translation_vector_weight = data.get("translation_vector_weight", 0.0)
    top_k = data.get("top_k", 5)
    text_field_name = "sentence"
    text_vector_field_name = "embedding"
    translation_vector_field_name = "translation"

    return (
        query_text,
        query_text_vector,
        query_translation_vector,
        text_weight,
        text_vector_weight,
        translation_vector_weight,
        top_k,
        text_field_name,
        text_vector_field_name,
        translation_vector_field_name,
    )


@app.route("/api/item_hybrid_search", methods=["POST"])
def item_hybrid_search():
    """
    アイテムのハイブリッド検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def item_hybrid_search():
    global item_index_name

    (
        query_text,
        query_text_vector,
        query_translation_vector,
        text_weight,
        text_vector_weight,
        translation_vector_weight,
        top_k,
        text_field_name,
        text_vector_field_name,
        translation_vector_field_name,
    ) = parse_hybrid_search_input()

    results = perform_hybrid_search(
        query_text,
        query_text_vector,
        query_translation_vector,
        index_name=item_index_name,
        text_field_name=text_field_name,
        text_vector_field_name=text_vector_field_name,
        translation_vector_field_name=translation_vector_field_name,
        text_weight=text_weight,
        text_vector_weight=text_vector_weight,
        translation_vector_weight=translation_vector_weight,
        top_k=top_k,
    )
    return jsonify(results)


@app.route("/api/user_hybrid_search", methods=["POST"])
def user_hybrid_search():
    """
    ユーザーのハイブリッド検索を実行するAPIエンドポイント

    Returns
    -------
    flask.Response
        検索結果を含むJSONレスポンス
    """
def user_hybrid_search():
    global user_index_name

    (
        query_text,
        query_text_vector,
        query_translation_vector,
        text_weight,
        text_vector_weight,
        translation_vector_weight,
        top_k,
        text_field_name,
        text_vector_field_name,
        translation_vector_field_name,
    ) = parse_hybrid_search_input()

    results = perform_hybrid_search(
        query_text,
        query_text_vector,
        query_translation_vector,
        index_name=user_index_name,
        text_field_name=text_field_name,
        text_vector_field_name=text_vector_field_name,
        translation_vector_field_name=translation_vector_field_name,
        text_weight=text_weight,
        text_vector_weight=text_vector_weight,
        translation_vector_weight=translation_vector_weight,
        top_k=top_k,
    )
    return jsonify(results)


@app.route("/favicon.ico")
def favicon():
    """
    ランダムな背景色とロゴを持つファビコンを生成するAPIエンドポイント

    Returns
    -------
    flask.Response
        ICO形式の画像を含むレスポンス
    """
def favicon():
    # 画像サイズ
    size = (32, 32)

    # ランダムな背景色を生成
    r = random.randint(0, 128)
    g = random.randint(0, 128)
    b = random.randint(0, 128)
    img = Image.new("RGB", size, (r, g, b))

    # ロゴを描画
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("ipag.ttf", 25)
    draw.text((5, 5), "Hy", fill=(255, 255, 255), font=font)

    # 画像をバイトストリームに保存
    img_io = io.BytesIO()
    img.save(img_io, "ICO")
    img_io.seek(0)

    # レスポンスを作成
    response = make_response(img_io.getvalue())
    response.headers["Content-Type"] = "image/x-icon"
    return response


@app.route("/<path:filename>")
def serve_file(filename):
    """
    指定されたファイルを提供するAPIエンドポイント

    Parameters
    ----------
    filename : str
        提供するファイルのパス

    Returns
    -------
    flask.Response
        ファイルを含むレスポンス
    """
def serve_file(filename):
    return send_from_directory("./content", filename, as_attachment=False)


@app.route("/")
def serve_index():
    """
    インデックスページを提供するAPIエンドポイント

    Returns
    -------
    flask.Response
        インデックスページを含むレスポンス
    """
def serve_index():
    return send_from_directory("./content", "index.html")


if __name__ == "__main__":
    # デバッグモードでアプリケーションを実行
    app.run(debug=True)
