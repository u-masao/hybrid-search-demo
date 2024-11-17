import gradio as gr

from src.search_program import perform_bm25_search, perform_vector_search

# Initialize Elasticsearch client
es_host = "https://localhost:9200"


def search_articles(query_text, top_k=5):
    # Perform a BM25 search on the articles index
    result = perform_bm25_search(es_host, "article_data", query_text, top_k)
    print(result)
    result = perform_vector_search(es_host, "article_data", query_text, top_k)
    print(result)
    print(type(result))
    return str(result)


def search_users(query_text, top_k=5):
    # Perform a BM25 search on the users index
    result = perform_bm25_search(es_host, "user_sentences", query_text, top_k)
    print(result)
    result = perform_vector_search(es_host, "article_data", query_text, top_k)
    print(result)
    print(type(result))
    return str(result)


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
