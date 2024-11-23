import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml


def load_data(file_path):
    return pd.read_parquet(file_path)


def load_mermaid_chart(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    # Extract the Mermaid chart from the Markdown content
    mermaid_start = content.find("```mermaid")
    mermaid_end = content.find("```", mermaid_start + 1)
    if mermaid_start != -1 and mermaid_end != -1:
        return content[
            mermaid_start + len("```mermaid") : mermaid_end  # noqa: E203
        ].strip()
    return ""


def load_dvc_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


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
                import mermaid from
                'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            """,
            height=700,
        )

    # Load and display DVC YAML data
    dvc_data = load_dvc_yaml("dvc.yaml")
    st.header("DVC Pipeline Configuration")
    st.json(dvc_data)

    # Load and display user profiles
    user_profiles = load_data("data/users_with_sentences.parquet")
    st.header("User Profiles")
    st.write(user_profiles.head())
    st.write(user_profiles.describe())

    # Load and display formatted dataset
    formatted_dataset = load_data("data/formatted_dataset.parquet")
    st.header("Formatted Dataset")
    st.write(formatted_dataset.head())
    st.write(formatted_dataset.describe())

    # Load and display embeddings
    embeddings = load_data("data/item_embeddings.parquet")
    st.header("Embeddings")
    st.write(embeddings.head())
    st.write(embeddings.describe())

    # Load and display user embeddings
    user_embeddings = load_data("data/user_embeddings.parquet")
    st.header("User Embeddings")
    st.write(user_embeddings.head())
    st.write(user_embeddings.describe())

    # Load and display user history
    user_history = load_data("data/user_history.parquet")
    st.header("User History")
    st.write(user_history.head())
    st.write(user_history.describe())


if __name__ == "__main__":
    main()
