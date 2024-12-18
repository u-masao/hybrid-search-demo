import datetime  # 日付操作用
import random

import click
import pandas as pd


def generate_user_history(
    item_df, user_df, max_views=10, label_probability=0.5
):
    """
    各アイテムの閲覧履歴を生成します。

    Parameters
    ----------
    item_df : pandas.DataFrame
        'sentence'列を含むアイテムのDataFrame。
    user_df : pandas.DataFrame
        ユーザープロファイルを含むDataFrame。

    Returns
    -------
    pandas.DataFrame
        閲覧履歴を含むDataFrame。
    """
    history = []
    user_ids = list(user_df["id"].values)

    for index, row in item_df.iterrows():
        num_views = random.randint(0, max_views)
        for _ in range(num_views):
            user_id = random.choice(user_ids)
            timestamp = datetime.datetime.now() - datetime.timedelta(
                days=random.randint(0, 365)
            )
            history.append(
                {
                    "item_id": row["id"],
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "label": 1 if random.random() < label_probability else -1,
                }
            )

    return pd.DataFrame(history)


@click.command()  # コマンドラインインターフェースを定義
@click.argument("item_embeddings_file", type=click.Path(exists=True))
@click.argument("user_embeddings_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--max_views", type=int, default=10)
def main(item_embeddings_file, user_embeddings_file, output_file, max_views):
    """Generate browsing history and save to a parquet file."""
    # Load embeddings
    # 埋め込みをロード
    item_embeddings = pd.read_parquet(item_embeddings_file)
    user_embeddings = pd.read_parquet(user_embeddings_file)

    # Debug output for input columns
    # 入力列のデバッグ出力
    print("Item Embeddings Columns:", item_embeddings.columns)
    print("User Embeddings Columns:", user_embeddings.columns)
    df_history = generate_user_history(
        item_embeddings, user_embeddings, max_views=max_views
    )

    # Merge embeddings with history
    # 履歴と埋め込みをマージ
    df_history = df_history.merge(
        item_embeddings[["id", "embedding"]]
        .rename(columns={"embedding": "item_embedding"})
        .set_index("id"),
        left_on="item_id",
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
    # 関連する列を選択
    df_history = df_history[
        [
            "item_embedding",
            "user_embedding",
            "item_id",
            "user_id",
            "label",
        ]
    ]

    df_history.to_parquet(output_file)

    print(df_history)
    print(df_history.iloc[0])
    print(df_history.columns)


if __name__ == "__main__":
    main()
