import pandas as pd
from loguru import logger

from embedding.embedding import Embedding


def embed_sentences(input_file, output_file, dimension, model_name, limit):
    """Generate embeddings for sentences in the dataset."""
    logger.info(f"Reading formatted dataset from {input_file}")
    df = pd.read_parquet(input_file)

    embedding_model = Embedding(dimension=dimension)
    df["embedding"] = df["sentence"].apply(lambda x: embedding_model.generate_embedding(x) if limit == 0 or df.index.get_loc(x) < limit else None)

    df.to_parquet(output_file)
    logger.info(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("input_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    @click.argument("dimension", type=int)
    @click.argument("model_name", type=str)
    @click.argument("limit", type=int)
    def main(input_file, output_file, dimension, model_name, limit):
        embed_sentences(input_file, output_file, dimension, model_name, limit)


    main()
