from flask import Flask, jsonify, request

from src.embedding_api.client import text_embedding
from src.search_app.es_utils import (
    get_item_info,
    get_user_info,
    perform_hybrid_search,
    perform_translation_search,
    perform_vector_search,
)

app = Flask(__name__)

item_index_name = "item_develop"
user_index_name = "user_develop"

text_embedding_field_name = "embedding"
translated_field_name = "translation"


@app.route("/api/vector_search", methods=["POST"])
def vector_search():
    data = request.json
    query_text = data.get("query_text")
    query_vector = text_embedding(query_text)
    if "embedding" not in query_vector:
        return jsonify({"status": "error"})

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
    data = request.json
    user_id = data.get("id")
    user_data = get_user_info(user_id, user_index_name)
    return jsonify(user_data)


@app.route("/api/item_info/", methods=["POST"])
def item_info():
    data = request.json
    item_id = data.get("id")
    item_data = get_item_info(item_id, item_index_name)
    return jsonify(item_data)


@app.route("/api/user_translation_search", methods=["POST"])
def user_translation_search():
    data = request.json
    translation_vector = data.get("translation")
    results = perform_translation_search(translation_vector, user_index_name)
    return jsonify(results)


@app.route("/api/item_translation_search", methods=["POST"])
def item_translation_search():
    data = request.json
    translation_vector = data.get("translation")
    results = perform_translation_search(translation_vector, item_index_name)
    return jsonify(results)


@app.route("/api/user_text_embedding_search", methods=["POST"])
def user_text_embedding_search():
    data = request.json
    query_vector = data.get("embedding")
    results = perform_vector_search(
        query_vector, user_index_name, text_embedding_field_name
    )
    return jsonify(results)


@app.route("/api/item_text_embedding_search", methods=["POST"])
def item_text_embedding_search():
    data = request.json
    query_vector = data.get("embedding")
    results = perform_vector_search(
        query_vector, item_index_name, text_embedding_field_name
    )
    return jsonify(results)


@app.route("/api/item_hybrid_search", methods=["POST"])
def item_hybrid_search():
    global item_index_name
    data = request.json
    query_text = data.get("text")
    query_text_vector = data.get("embedding")
    query_translation_vector = data.get("translation")
    text_weight = data.get("text_weight", 1.0)
    text_vector_weight = data.get("text_vector_weight", 1.0)
    translation_vector_weight = data.get("translation_vector_weight", 1.0)
    top_k = data.get("top_k", 5)
    text_field_name = "sentence"
    text_vector_field_name = "embedding"
    translation_vector_field_name = "translation"

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
    global user_index_name
    data = request.json
    query_text = data.get("text")
    query_text_vector = data.get("embedding")
    query_translation_vector = data.get("translation")
    text_weight = data.get("text_weight", 1.0)
    text_vector_weight = data.get("text_vector_weight", 1.0)
    translation_vector_weight = data.get("translation_vector_weight", 1.0)
    top_k = data.get("top_k", 5)
    text_field_name = "sentence"
    text_vector_field_name = "embedding"
    translation_vector_field_name = "translation"

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


if __name__ == "__main__":
    app.run(debug=True)
