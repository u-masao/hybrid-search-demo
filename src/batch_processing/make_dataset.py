import os  # OS操作用
import re

import click
import mlflow
import pandas as pd
from datasets import load_dataset


def make_dataset(output_file):
    """
    Hugging Faceからデータセットをダウンロードして作成する関数。

    Parameters
    ----------
    output_file : str
        データセットをParquet形式で保存するパス。
    """
    """
    Hugging Faceからデータセットをダウンロードして作成する関数。

    パラメータ:
    - output_file: データセットをParquet形式で保存するパス。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = load_dataset("llm-book/livedoor-news-corpus")
    # DatasetDictの各分割をPandas DataFrameに変換
    # 各分割を処理
    for split_name, split in dataset.items():
        df = pd.DataFrame.from_dict(split)
        # URLから数値IDを抽出し、新しい列として追加
        # 正規表現を使用してIDを抽出
        df["id"] = (
            df["url"]
            .apply(
                lambda x: (
                    re.search(r"/(\d+)/", x).group(1)
                    if re.search(r"/(\d+)/", x)
                    else None
                )
            )
            .astype(int)
        )

        # 出力ファイル名の'train'を現在の分割名に置き換え
        # 分割名に基づいてファイル名を変更
        split_output_file = output_file.replace("train", split_name)
        print(f"Dataset split '{split_name}' as DataFrame:\n{df}")
        # 入力と出力の長さをログに記録
        # mlflowを使用してパラメータをログ
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


@click.command()  # コマンドラインインターフェースを定義
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
