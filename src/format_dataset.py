import pandas as pd
import click

@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def main(input_file):
    """Read the dataset and print column names."""
    df = pd.read_parquet(input_file)
    print("Columns in the dataset:", df.columns.tolist())

if __name__ == "__main__":
    main()
