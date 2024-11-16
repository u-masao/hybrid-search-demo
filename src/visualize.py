import streamlit as st
import pandas as pd

def load_data(file_path):
    return pd.read_parquet(file_path)

def main():
    st.title("DVC Pipeline Outputs Summary")

    # Load and display user profiles
    user_profiles = load_data("data/user_profiles.parquet")
    st.header("User Profiles")
    st.write(user_profiles.describe())

    # Load and display formatted dataset
    formatted_dataset = load_data("data/formatted_dataset.parquet")
    st.header("Formatted Dataset")
    st.write(formatted_dataset.describe())

    # Load and display embeddings
    embeddings = load_data("data/embeddings.parquet")
    st.header("Embeddings")
    st.write(embeddings.describe())

    # Load and display user history
    user_history = load_data("data/user_history.parquet")
    st.header("User History")
    st.write(user_history.describe())

if __name__ == "__main__":
    main()
