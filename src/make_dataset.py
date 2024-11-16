import os
import re

import click
import mlflow
import pandas as pd
from datasets import load_dataset


def make_dataset(output_file):
    """
    Hugging Faceからデータセットをダウンロードして作成する関数。

    パラメータ:
    - output_file: データセットをParquet形式で保存するパス。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = load_dataset("llm-book/livedoor-news-corpus")
    # DatasetDictの各分割をPandas DataFrameに変換
    for split_name, split in dataset.items():
        df = pd.DataFrame.from_dict(split)
        # URLから数値IDを抽出し、新しい列として追加
        df["id"] = df["url"].apply(
            lambda x: (
                re.search(r"/(\d+)/", x).group(1)
                if re.search(r"/(\d+)/", x)
                else None
            )
        )

        # 出力ファイル名の'train'を現在の分割名に置き換え
        split_output_file = output_file.replace("train", split_name)
        print(f"Dataset split '{split_name}' as DataFrame:\n{df}")
        # 入力と出力の長さをログに記録
        mlflow.log_params(
            {
                f"input_length.{split_name}": len(split),
                f"output_length.{split_name}": len(df),
            }
        )
        df.to_parquet(split_output_file)
        print(
            f"Dataset split '{split_name}' saved to "
            f"{split_output_file} in Parquet format."
        )


@click.command()
@click.argument("output_file", type=click.Path())
def main(output_file):
    """コマンドライン引数を処理するメイン関数。"""
    mlflow.set_experiment("Dataset Creation")
    with mlflow.start_run():
        # CLIオプションをログに記録
        mlflow.log_params({"output_file": output_file})

        make_dataset(output_file)


if __name__ == "__main__":
    main()
