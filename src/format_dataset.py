import pandas as pd
import click

@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def main(input_file, output_file):
    """Read the dataset, print column names, and save formatted dataset."""
    df = pd.read_parquet(input_file)
    df['sentence'] = df.apply(lambda row: f"# {row['title']}\n\n**Category:** {row['category']}\n\n{row['content']}", axis=1)
    print("Columns in the dataset:", df.columns.tolist())
    df.to_parquet(output_file)
    print(f"Formatted dataset saved to {output_file}")
    # Read the saved Parquet file and print the first row
    df_loaded = pd.read_parquet(output_file)
    print("First row, 'sentence' column of the formatted dataset:", df_loaded.iloc[0]['sentence'])

if __name__ == "__main__":
    main()
