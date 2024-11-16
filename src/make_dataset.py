import os

import click
import pandas as pd
from datasets import load_dataset


def make_dataset(output_file):
    """
    Function to download and create a dataset from Hugging Face.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = load_dataset("llm-book/livedoor-news-corpus")
    # Convert each split in the DatasetDict to a Pandas DataFrame
    for split_name, split in dataset.items():
        df = pd.DataFrame.from_dict(split)
        if split_name == "train":
            df.to_parquet(output_file)
            print(f"Train dataset saved to {output_file} in Parquet format.")
        else:
            print(f"Dataset split '{split_name}' as DataFrame:\n", df)


@click.command()
@click.argument("output_file", type=click.Path())
def main(output_file):
    """Main function to handle command-line arguments."""
    make_dataset(output_file)


if __name__ == "__main__":
    main()
