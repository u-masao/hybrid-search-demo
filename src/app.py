from flask import Flask, render_template, request
from src.search_program import perform_vector_search, perform_bm25_search

app = Flask(__name__, template_folder="../templates")

@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query_text = request.form.get("query")
        es_host = "https://localhost:9200"
        index_name = "article_data"

        vector_results = perform_vector_search(es_host, index_name, query_text)
        bm25_results = perform_bm25_search(es_host, index_name, query_text)

        return render_template("results.html", vector_results=vector_results, bm25_results=bm25_results)

    return render_template("search.html")

if __name__ == "__main__":
    app.run(debug=True)
