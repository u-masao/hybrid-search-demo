import datetime
import random
import pandas as pd
from generate_users import generate_user_profiles

def generate_user_history(df, user_profiles):
    """
    Generate browsing history for each article.

    Parameters:
    - df: DataFrame containing 'sentence' column.
    - user_profiles: Dictionary of user profiles.

    Returns:
    DataFrame containing browsing history.
    """
    user_ids = list(user_profiles.keys())
    history = []

    for index, row in df.iterrows():
        num_views = random.randint(
            1, 10
        )  # Each article is viewed between 1 to 10 times
        for _ in range(num_views):
            user_id = random.choice(user_ids)
            timestamp = datetime.datetime.now() - datetime.timedelta(
                days=random.randint(0, 365)
            )
            user_profile = user_profiles[user_id]
            history.append(
                {
                    "article_id": index,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "age": user_profile["age"],
                    "gender": user_profile["gender"],
                    "preferences": user_profile["preferences"],
                }
            )

    return pd.DataFrame(history)

if __name__ == "__main__":
    # Example usage
    df = pd.read_parquet("articles.parquet")
    user_profiles = pd.read_parquet("user_profiles.parquet").to_dict(orient='index')
    df_history = generate_user_history(df, user_profiles)
    df_history.to_parquet("user_history.parquet")
