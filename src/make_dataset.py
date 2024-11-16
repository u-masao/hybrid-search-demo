import os
import re

import click
import pandas as pd
from datasets import load_dataset
import mlflow


def make_dataset(output_file):
    """
    Hugging Faceからデータセットをダウンロードして作成する関数。

    パラメータ:
    - output_file: データセットをParquet形式で保存するパス。
    """
    mlflow.set_experiment("Dataset Creation")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = load_dataset("llm-book/livedoor-news-corpus")
    # Convert each split in the DatasetDict to a Pandas DataFrame
    for split_name, split in dataset.items():
        df = pd.DataFrame.from_dict(split)
        # Extract numeric ID from URL and add as a new column
        df["id"] = df["url"].apply(
            lambda x: (
                re.search(r"/(\d+)/", x).group(1)
                if re.search(r"/(\d+)/", x)
                else None
            )
        )

        # Replace 'train' in the output file name with the current split name
        split_output_file = output_file.replace("train", split_name)
        print(f"Dataset split '{split_name}' as DataFrame:\n{df}")
        # Log input and output lengths
        mlflow.log_params({
            "input_length": len(split),
            "output_length": len(df)
        })
        df.to_parquet(split_output_file)
        print(
            f"Dataset split '{split_name}' saved to "
            f"{split_output_file} in Parquet format."
        )


@click.command()
@click.argument("output_file", type=click.Path())
def main(output_file):
    """コマンドライン引数を処理するメイン関数。"""
    with mlflow.start_run():
        make_dataset(output_file)
        # Log CLI options
        mlflow.log_params({"output_file": output_file})


if __name__ == "__main__":
    main()
