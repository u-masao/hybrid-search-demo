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
    print("BM25 Search Result:", result_bm25)  # デバッグ用に追加
    print("Vector Search Result:", result_vector)  # デバッグ用に追加
    formatted_results = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence_x', '')[:200]}"
        for item in result_vector
    )
    return formatted_results


def search_users(query_text, top_k=5):
    # Perform a BM25 search on the users index
    result = perform_bm25_search(es_host, "user_sentences", query_text, top_k)
    result = perform_vector_search(
        es_host, "user_sentences", query_text, top_k
    )
    formatted_results = "\n".join(
        f"- **ID**: {item['_source'].get('id', 'N/A')},"
        f" **Score**: {item['_score']},"
        f" **Sentence**: {item['_source'].get('sentence_x', '')[:200]}"
        for item in result
    )
    return formatted_results


with gr.Blocks() as demo:
    gr.Markdown("# Article and User Search")

    with gr.Tab("Search Articles"):
        article_query = gr.Textbox(label="Search Articles")
        article_results = gr.Markdown()
        article_search_button = gr.Button("Search")
        article_search_button.click(
            fn=search_articles,
            inputs=[article_query],
            outputs=[article_results],
        )

    with gr.Tab("Search Users"):
        user_query = gr.Textbox(label="Search Users")
        user_results = gr.Markdown()
        user_search_button = gr.Button("Search")
        user_search_button.click(
            fn=search_users, inputs=[user_query], outputs=[user_results]
        )

if __name__ == "__main__":
    demo.launch()
