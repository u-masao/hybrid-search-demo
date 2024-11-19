import click
import mlflow
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
        df.to_parquet(output_file)
        logger.info(f"Formatted dataset saved to {output_file}")


if __name__ == "__main__":
    main()
