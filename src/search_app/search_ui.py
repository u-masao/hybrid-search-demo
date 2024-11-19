import gradio as gr

from src.search_api.search_program import (
    perform_bm25_search,
    perform_vector_search,
)

# Initialize Elasticsearch client
es_host = "https://localhost:9200"


def search_items(query_text, top_k=5):
    # Perform a BM25 search on the items index
    result_bm25 = perform_bm25_search(es_host, "item_data", query_text, top_k)
    result_vector = perform_vector_search(
        es_host, "item_data", query_text, top_k
    )
    return result_bm25, result_vector


def search_users(query_text, top_k=5):
    # Perform a BM25 search on the users index
    result_bm25 = perform_bm25_search(
        es_host, "user_sentences", query_text, top_k
    )
    result_vector = perform_vector_search(
        es_host, "user_sentences", query_text, top_k
    )
    return result_bm25, result_vector


with gr.Blocks() as demo:
    gr.Markdown("# Item and User Search")

    query_input = gr.Textbox(label="Search Query")

    with gr.Row():
        with gr.Column():
            bm25_user_results = gr.Dataset(
                components=["text"], label="BM25 User Results"
            )
            vector_user_results = gr.Dataset(
                components=["text"], label="Vector Search User Results"
            )
        with gr.Column():
            bm25_item_results = gr.Dataset(
                components=["text"], label="BM25 Item Results"
            )
            vector_item_results = gr.Dataset(
                components=["text"], label="Vector Search Item Results"
            )

    details_output = gr.Textbox(label="Details", interactive=False)

    def display_details(item):
        return (
            f"ID: {item['_source'].get('id', 'N/A')}\n"
            f"Score: {item['_score']}\n"
            f"Sentence: {item['_source'].get('sentence', '')}"
        )

    query_input.submit(
        fn=search_items,
        inputs=[query_input],
        outputs=[bm25_item_results, vector_item_results],
    ).then(
        fn=search_users,
        inputs=[query_input],
        outputs=[bm25_user_results, vector_user_results],
    )

    bm25_item_results.select(
        display_details, inputs=[bm25_item_results], outputs=details_output
    )
    vector_item_results.select(
        display_details,
        inputs=[vector_item_results],
        outputs=details_output,
    )
    bm25_user_results.select(
        display_details, inputs=[bm25_user_results], outputs=details_output
    )
    vector_user_results.select(
        display_details, inputs=[vector_user_results], outputs=details_output
    )

if __name__ == "__main__":
    demo.launch()
