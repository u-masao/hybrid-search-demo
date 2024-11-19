from flask import Flask, render_template, request

from src.search_app.search_program import (
    perform_bm25_search,
    perform_vector_search,
)

app = Flask(__name__, template_folder="../../templates")


item_index_name = "item_develop"
user_index_name = "user_develop"


@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query_text = request.form.get("query")
        es_host = "https://localhost:9200"
        target_column = request.form.get("target_column", "embedding")

        # Perform search on both item and user indices
        item_vector_results, user_vector_results = perform_vector_search(
            es_host, item_index_name, user_index_name, query_text, target_column
        )
        item_bm25_results, user_bm25_results = perform_bm25_search(
            es_host, item_index_name, user_index_name, query_text
        )

        return render_template(
            "results.html",
            item_vector_results=item_vector_results,
            user_vector_results=user_vector_results,
            target_column=target_column,
            item_bm25_results=item_bm25_results,
            user_bm25_results=user_bm25_results,
        )

    return render_template("search.html")


if __name__ == "__main__":
    app.run(debug=True)
