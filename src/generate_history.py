import datetime
import random

import click
import pandas as pd


def generate_user_history(
    article_df, user_df, max_views=10, label_probability=0.5
):
    """
    Generate browsing history for each article.

    Parameters:
    - article_df: DataFrame containing 'sentence' column.
    - user_df: Dictionary of user profiles.

    Returns:
    DataFrame containing browsing history.
    """
    history = []
    user_ids = list(user_df["id"].values)

    for index, row in article_df.iterrows():
        num_views = random.randint(0, max_views)
        for _ in range(num_views):
            user_id = random.choice(user_ids)
            timestamp = datetime.datetime.now() - datetime.timedelta(
                days=random.randint(0, 365)
            )
            history.append(
                {
                    "article_id": index,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "label": 1 if random.random() < label_probability else -1,
                }
            )

    return pd.DataFrame(history)


@click.command()
@click.argument("article_embeddings_file", type=click.Path(exists=True))
@click.argument("user_embeddings_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--max_views", type=int, default=10)
def main(
    article_embeddings_file, user_embeddings_file, output_file, max_views
):
    """Generate browsing history and save to a parquet file."""
    # Load embeddings
    article_embeddings = pd.read_parquet(article_embeddings_file)
    user_embeddings = pd.read_parquet(user_embeddings_file)

    # Debug output for input columns
    print("Article Embeddings Columns:", article_embeddings.columns)
    print("User Embeddings Columns:", user_embeddings.columns)
    print("Article Embeddings Columns:", article_embeddings.dtypes)
    print("User Embeddings Columns:", user_embeddings.dtypes)
    df_history = generate_user_history(
        article_embeddings, user_embeddings, max_views=max_views
    )

    print(f"{df_history.dtypes=}")

    # Merge embeddings with history
    df_history = df_history.merge(
        article_embeddings[["id", "embedding"]]
        .rename(columns={"embedding": "article_embedding"})
        .set_index("id"),
        left_on="article_id",
        right_index=True,
        how="left",
    )
    df_history = df_history.merge(
        user_embeddings[["id", "embedding"]]
        .rename(columns={"embedding": "user_embedding"})
        .set_index("id"),
        left_on="user_id",
        right_index=True,
        how="left",
    )

    # Select relevant columns
    df_history = df_history[["article_embedding", "user_embedding", "label"]]

    df_history.to_parquet(output_file)


if __name__ == "__main__":
    main()
