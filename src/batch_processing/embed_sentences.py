import mlflow
import pandas as pd
from loguru import logger
from tqdm import tqdm

from embedding import Embedding


def embed_sentences(input_file, output_file, dimension, model_name, limit):
    """
    データセット内の文の埋め込みを生成します。

    パラメータ:
    - input_file: データセットを含む入力Parquetファイルのパス。
    - output_file: 埋め込みを含む出力Parquetファイルを保存するパス。
    - dimension: 埋め込みベクトルの次元数。
    - model_name: 埋め込みに使用する事前学習済みモデルの名前。
    - limit: データセットから処理する最大行数。
    """
    with mlflow.start_run():
        logger.info(f"Reading formatted dataset from {input_file}")
        df = pd.read_parquet(input_file)
        input_length = len(df)

        logger.info(f"DataFrame loaded with shape: {df.shape} and columns: {df.columns}")

    # 制限が指定されている場合は行数を制限
    if limit > 0:
        df = df.head(min(limit, len(df)))

    # 指定された次元で埋め込みモデルを初期化
    embedding_model = Embedding(dimension=dimension)
    # pandasにtqdmを登録
    tqdm.pandas()

    # DataFrame内の各文に埋め込み生成を適用し、プログレスバーを表示
    df["embedding"] = df["sentence"].progress_apply(
        lambda x: (embedding_model.generate_embedding(x))
    )

    # 埋め込みを含むDataFrameをParquetファイルに保存
    logger.info(f"Saving embeddings to {output_file}")
    df.to_parquet(output_file)
    output_length = len(df)
    mlflow.log_metric("input_length", input_length)
    mlflow.log_metric("output_length", output_length)
    logger.info(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("input_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    @click.option("--dimension", type=int, default=384)
    @click.option(
        "--model_name", type=str, default="intfloat/multilingual-e5-small"
    )
    @click.option("--limit", type=int, default=0)
    def main(input_file, output_file, dimension, model_name, limit):
        embed_sentences(input_file, output_file, dimension, model_name, limit)

    main()
