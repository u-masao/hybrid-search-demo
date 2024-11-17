import gradio as gr

from src.search_program import perform_bm25_search, perform_vector_search

# Initialize Elasticsearch client
es_host = "https://localhost:9200"


def search_articles(query_text, top_k=5):
    # Perform a BM25 search on the articles index
    result_bm25 = perform_bm25_search(
        es_host, "article_data", query_text, top_k
    )
    result_vector = perform_vector_search(
        es_host, "article_data", query_text, top_k
    )
    formatted_results = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence', '')[:200]}"
        for item in result_vector
    )
    formatted_results_bm25 = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence', '')[:200]}"
        for item in result_bm25
    )
    return formatted_results_bm25, formatted_results


def search_users(query_text, top_k=5):
    # Perform a BM25 search on the users index
    result_bm25 = perform_bm25_search(es_host, "user_sentences", query_text, top_k)
    result_vector = perform_vector_search(
        es_host, "user_sentences", query_text, top_k
    )
    formatted_results = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence', '')[:200]}"
        for item in result_vector
    )
    formatted_results_bm25 = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence', '')[:200]}"
        for item in result_bm25
    )
    return formatted_results_bm25, formatted_results


with gr.Blocks() as demo:
    gr.Markdown("# Article and User Search")

    query_input = gr.Textbox(label="Search Query")

    with gr.Row():
        with gr.Column():
            bm25_user_results = gr.HTML("<div style='border: 3px solid gray;'><h3>BM25 User Results</h3><div id='bm25_user_results'></div></div>")
            vector_user_results = gr.HTML("<div style='border: 3px solid gray;'><h3>Vector Search User Results</h3><div id='vector_user_results'></div></div>")
        with gr.Column():
            bm25_article_results = gr.HTML("<div style='border: 3px solid gray;'><h3>BM25 Article Results</h3><div id='bm25_article_results'></div></div>")
            vector_article_results = gr.HTML("<div style='border: 3px solid gray;'><h3>Vector Search Article Results</h3><div id='vector_article_results'></div></div>")

    query_input.submit(
        fn=search_articles,
        inputs=[query_input],
        outputs=[bm25_article_results, vector_article_results],
    )
    query_input.submit(
        fn=search_users,
        inputs=[query_input],
        outputs=[bm25_user_results, vector_user_results],
    )

if __name__ == "__main__":
    demo.launch()
