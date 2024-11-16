import click
import mlflow
import pandas as pd
from loguru import logger


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
def main(input_file, output_file):
    """データセットを読み込み、列名を表示し、フォーマットされたデータセットを保存します。"""
    with mlflow.start_run():
        logger.info(f"Reading dataset from {input_file}")
        df = pd.read_parquet(input_file)
        input_length = len(df)
        df = create_sentence_column(df)
        output_length = len(df)
        mlflow.log_metric("input_length", input_length)
        mlflow.log_metric("output_length", output_length)
        logger.info("Columns in the dataset: {}", df.columns.tolist())
        df.to_parquet(output_file)
        logger.info(f"Formatted dataset saved to {output_file}")
        # 保存されたParquetファイルを読み込み、最初の行を表示
        df_loaded = pd.read_parquet(output_file)
        logger.info(
            "First row, 'sentence' column of the formatted dataset: {}",
            df_loaded.iloc[0]["sentence"],
        )


if __name__ == "__main__":
    main()
