import datetime
import os
import random

import click
import mlflow
import openai
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def create_sentence_column(df):
    """
    タイトル、カテゴリ、コンテンツを組み合わせてDataFrameに'sentence'列を作成します。

    パラメータ:
    - df: 'title'、'category'、'content'列を含むDataFrame。

    戻り値:
    追加の'sentence'列を持つDataFrame。
    """
    df["sentence"] = df.apply(
        lambda row: f"# {row['title']}\n\n"
        f"**Category:** {row['category']}\n\n"
        f"{row['content']}",
        axis=1,
    )
    return df


def generate_user_history(df, num_users=1000):
    """
    各記事に対して架空のユーザー閲覧履歴を生成します。

    パラメータ:
    - df: 'sentence'列を含むDataFrame。
    - num_users: ユニークなユーザーの数。

    戻り値:
    閲覧履歴を含むDataFrame。
    """
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    # Define possible attributes
    genders = ["male", "female", "non-binary"]
    categories = df["category"].unique().tolist()

    # Set OpenAI API key
    openai.api_key = os.environ["OPENAI_API_KEY"]
    user_profiles = {
        user_id: (
            lambda age, gender, preferences: {
                "age": age,
                "gender": gender,
                "preferences": preferences,
                "introduction": openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=(
                        f"Create a short self-introduction for a {user_id}"
                        f" who is {age} years old, {gender}, "
                        f"and likes {', '.join(preferences)}. in japanese"
                    ),
                    max_tokens=50,
                    temperature=0,
                )
                .choices[0]
                .text.strip(),
            }
        )(
            random.randint(18, 70),
            random.choice(genders),
            random.sample(
                categories, k=random.randint(1, min(3, len(categories)))
            ),
        )
        for user_id in user_ids
    }

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


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--num_users", type=int, default=1000)
@click.option("--limit", type=int, default=0)
def main(input_file, output_file, num_users, limit):
    """データセットを読み込み、列名を表示し、フォーマットされたデータセットを保存します。"""
    with mlflow.start_run():
        logger.info(f"Reading dataset from {input_file}")
        df = pd.read_parquet(input_file)

        # 制限が指定されている場合は行数を制限
        if limit > 0:
            df = df.sample(min(limit, len(df)))

        input_length = len(df)
        df = create_sentence_column(df)
        output_length = len(df)
        mlflow.log_metric("input_length", input_length)
        mlflow.log_metric("output_length", output_length)
        logger.info("Columns in the dataset: {}", df.columns.tolist())
        df_history = generate_user_history(df, num_users=num_users)
        df_history.to_parquet(output_file)
        logger.info(f"Formatted dataset saved to {output_file}")
        # 保存されたParquetファイルを読み込み、最初の行を表示
        df_loaded = pd.read_parquet(output_file)
        logger.info(
            "First row, 'sentence' column of the formatted dataset: {}",
            df_loaded.iloc[0]["sentence"],
        )


if __name__ == "__main__":
    main()
