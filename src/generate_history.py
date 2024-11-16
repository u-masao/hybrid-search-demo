import datetime
import random

import click
import pandas as pd


def generate_user_history(df, user_profiles, max_views=10):
    """
    Generate browsing history for each article.

    Parameters:
    - df: DataFrame containing 'sentence' column.
    - user_profiles: Dictionary of user profiles.

    Returns:
    DataFrame containing browsing history.
    """
    history = []

    for index, row in df.iterrows():
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
                }
            )

    return pd.DataFrame(history)


@click.command()
@click.argument("articles_file", type=click.Path(exists=True))
@click.argument("user_profiles_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--max_views", type=int, default=10)
def main(articles_file, user_profiles_file, output_file, max_views):
    """Generate browsing history and save to a parquet file."""
    df = pd.read_parquet(articles_file)
    user_profiles = pd.read_parquet(user_profiles_file).to_dict(orient="index")
    df_history = generate_user_history(df, user_profiles, max_views=max_views)
    df_history.to_parquet(output_file)


if __name__ == "__main__":
    main()
