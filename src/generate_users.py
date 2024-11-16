import os
import random

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


def generate_user_profiles(num_users=1000, categories=None):
    """
    Generate fictional user profiles.

    Parameters:
    - num_users: Number of unique users.
    - categories: List of categories for user preferences.

    Returns:
    A dictionary of user profiles.
    """
    if categories is None:
        categories = []

    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    genders = ["male", "female", "non-binary"]

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    user_profiles = {}
    for user_id in tqdm(user_ids, desc="Generating user profiles"):
        age = random.randint(18, 70)
        gender = random.choice(genders)
        preferences = random.sample(
            categories, k=random.randint(1, min(3, len(categories)))
        )
        introduction = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Create a short self-introduction for"
                            f" a {user_id}"
                            f" who is {age} years old, {gender},"
                            f" and likes {', '.join(preferences)}"
                            " in japanese."
                        ),
                    }
                ],
                max_tokens=50,
                temperature=0.9,
            )
            .choices[0]
            .message.content.strip()
        )

        user_profiles[user_id] = {
            "age": age,
            "gender": gender,
            "preferences": preferences,
            "introduction": introduction,
        }

    return user_profiles


@click.command()
@click.option(
    "--num_users", default=1000, help="Number of unique users to generate."
)
@click.option(
    "--categories",
    default="technology,sports,music,art",
    help="Comma-separated list of categories for user preferences.",
)
@click.argument("output_file", type=click.Path())
def main(num_users, categories, output_file):
    """Generate user profiles and save to a parquet file."""
    categories_list = categories.split(",")
    user_profiles = generate_user_profiles(
        num_users=num_users, categories=categories_list
    )
    df_user_profiles = pd.DataFrame.from_dict(user_profiles, orient="index")
    df_user_profiles.to_parquet(output_file)


if __name__ == "__main__":
    main()
