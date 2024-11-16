import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def load_data(file_path):
    return pd.read_parquet(file_path)


def load_mermaid_chart(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    # Extract the Mermaid chart from the Markdown content
    mermaid_start = content.find("```mermaid")
    mermaid_end = content.find("```", mermaid_start + 1)
    if mermaid_start != -1 and mermaid_end != -1:
        return content[mermaid_start + len("```mermaid"):mermaid_end].strip()
    return ""

def main():
    st.title("DVC Pipeline Outputs Summary")

    # Load and display the Mermaid chart
    mermaid_chart = load_mermaid_chart("PIPELINE.md")
    if mermaid_chart:
        components.html(
            f"""
            <div class="mermaid">
            {mermaid_chart}
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            """,
            height=500,
        )
    st.title("DVC Pipeline Outputs Summary")

    # Load and display user profiles
    user_profiles = load_data("data/user_profiles.parquet")
    st.header("User Profiles")
    st.write(user_profiles.head())
    st.write(user_profiles.describe())

    # Load and display formatted dataset
    formatted_dataset = load_data("data/formatted_dataset.parquet")
    st.header("Formatted Dataset")
    st.write(formatted_dataset.head())
    st.write(formatted_dataset.describe())

    # Load and display embeddings
    embeddings = load_data("data/embeddings.parquet")
    st.header("Embeddings")
    st.write(embeddings.head())
    st.write(embeddings.describe())

    # Load and display user embeddings
    user_embeddings = load_data("data/user_embeddings.parquet")
    st.header("User Embeddings")
    st.write(user_embeddings.head())
    st.write(user_embeddings.describe())
    user_history = load_data("data/user_history.parquet")
    st.header("User History")
    st.write(user_history.head())
    st.write(user_history.describe())


if __name__ == "__main__":
    main()
