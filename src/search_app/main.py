from flask import Flask, jsonify, request
from src.search_app.es_utils import (
    perform_vector_search,
    get_user_info,
    get_item_info,
    perform_translation_search,
)
from src.embedding_api.client import text_embedding
from src.translation_api.client import translation_user, translation_item

app = Flask(__name__)

item_index_name = "item_develop"
user_index_name = "user_develop"

@app.route("/api/vector_search", methods=["POST"])
def vector_search():
    data = request.json
    query_text = data.get("query_text")
    query_vector = text_embedding(query_text)
    item_results, user_results = perform_vector_search(
        query_vector, item_index_name, user_index_name
    )
    return jsonify({"item_results": item_results, "user_results": user_results})

@app.route("/api/user_info/<int:user_id>", methods=["GET"])
def user_info(user_id):
    user_data = get_user_info(user_id, user_index_name)
    return jsonify(user_data)

@app.route("/api/item_info/<int:item_id>", methods=["GET"])
def item_info(item_id):
    item_data = get_item_info(item_id, item_index_name)
    return jsonify(item_data)

@app.route("/api/user_translation_search", methods=["POST"])
def user_translation_search():
    data = request.json
    user_id = data.get("user_id")
    translation_vector = translation_user(user_id)
    results = perform_translation_search(translation_vector, user_index_name)
    return jsonify(results)

@app.route("/api/item_translation_search", methods=["POST"])
def item_translation_search():
    data = request.json
    item_id = data.get("item_id")
    translation_vector = translation_item(item_id)
    results = perform_translation_search(translation_vector, item_index_name)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
