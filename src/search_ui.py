import gradio as gr
from elasticsearch import Elasticsearch
from src.search_program import perform_vector_search, perform_bm25_search

# Initialize Elasticsearch client
es_host = "http://localhost:9200"

def search_articles(query_text, top_k=5):
    # Perform a BM25 search on the articles index
    return perform_bm25_search(es_host, "articles", query_text, top_k)

def search_users(query_text, top_k=5):
    # Perform a BM25 search on the users index
    return perform_bm25_search(es_host, "users", query_text, top_k)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Article and User Search")

        with gr.Tab("Search Articles"):
            article_query = gr.Textbox(label="Search Articles")
            article_results = gr.Dataframe(headers=["Title", "Snippet"])
            article_search_button = gr.Button("Search")
            article_search_button.click(
                fn=search_articles,
                inputs=[article_query],
                outputs=[article_results]
            )

        with gr.Tab("Search Users"):
            user_query = gr.Textbox(label="Search Users")
            user_results = gr.Dataframe(headers=["Name", "Profile"])
            user_search_button = gr.Button("Search")
            user_search_button.click(
                fn=search_users,
                inputs=[user_query],
                outputs=[user_results]
            )

    demo.launch()

if __name__ == "__main__":
    main()
