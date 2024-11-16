import pandas as pd
from loguru import logger

from embedding.embedding import Embedding


def embed_sentences(input_file, output_file, dimension, model_name, limit):
    """
    Generate embeddings for sentences in the dataset.

    Parameters:
    - input_file: Path to the input Parquet file containing the dataset.
    - output_file: Path to save the output Parquet file with embeddings.
    - dimension: Dimensionality of the embedding vectors.
    - model_name: Name of the pre-trained model to use for embeddings.
    - limit: Maximum number of rows to process from the dataset.
    """
    logger.info(f"Reading formatted dataset from {input_file}")
    df = pd.read_parquet(input_file)

    # Limit the number of rows if a limit is specified
    if limit > 0:
        df = df.head(max(limit, len(df)))

    # Initialize the embedding model with the specified dimension
    embedding_model = Embedding(dimension=dimension)
    # Apply the embedding generation to each sentence in the DataFrame
    df["embedding"] = df["sentence"].apply(
        lambda x: (embedding_model.generate_embedding(x))
    )

    # Save the DataFrame with embeddings to a Parquet file
    df.to_parquet(output_file)
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
