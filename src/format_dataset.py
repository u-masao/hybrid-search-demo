import pandas as pd
from hello import hello
import click

@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def main(input_file, output_file):
    """Read the dataset, print column names, and save formatted dataset."""
    df = pd.read_parquet(input_file)
    print("Columns in the dataset:", df.columns.tolist())
    df.to_parquet(output_file)
    print(f"Formatted dataset saved to {output_file}")

if __name__ == "__main__":
    main()
