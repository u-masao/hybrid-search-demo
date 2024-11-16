import click
from datasets import load_dataset

def make_dataset(output_file):
    """
    Function to download and create a dataset from Hugging Face.
    """
    dataset = load_dataset('llm-book/livedoor-news-corpus')
    print("Dataset structure:", dataset)
@click.command()
@click.argument('output_file', type=click.Path())
def main(output_file):
    """Main function to handle command-line arguments."""
    make_dataset(output_file)

if __name__ == "__main__":
    main()
