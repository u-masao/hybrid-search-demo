from loguru import logger
import pandas as pd
from embedding.embedding import Embedding

def embed_sentences(input_file, output_file):
    """Generate embeddings for sentences in the dataset."""
    logger.info(f"Reading formatted dataset from {input_file}")
    df = pd.read_parquet(input_file)
    
    embedding_model = Embedding(dimension=128)  # Assuming a dimension of 128
    df['embedding'] = df['sentence'].apply(embedding_model.generate_embedding)
    
    df.to_parquet(output_file)
    logger.info(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("input_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    def main(input_file, output_file):
        embed_sentences(input_file, output_file)

    main()
