import click
import torch
import pandas as pd
from two_tower_model import train_two_tower_model

def load_embeddings(file_path):
    return pd.read_parquet(file_path).astype('float32').to_numpy()

def load_labels(labels_file):
    # Load labels from a Parquet file
    labels_df = pd.read_parquet(labels_file)
    return torch.tensor(labels_df['label'].astype('float32').values, dtype=torch.float32)

def main(user_embeddings_file, article_embeddings_file, model_output_file, labels_file):
    user_embeddings = torch.tensor(load_embeddings(user_embeddings_file), dtype=torch.float32)
    article_embeddings = torch.tensor(load_embeddings(article_embeddings_file), dtype=torch.float32)
    
    # Load labels for training
    labels = load_labels(labels_file)

    model = train_two_tower_model(user_embeddings, article_embeddings, labels)
    torch.save(model.state_dict(), model_output_file)

@click.command()
@click.argument("user_embeddings_file", type=click.Path(exists=True))
@click.argument("article_embeddings_file", type=click.Path(exists=True))
@click.argument("model_output_file", type=click.Path())
@click.argument("labels_file", type=click.Path(), default="data/user_history.parquet")
def cli(user_embeddings_file, article_embeddings_file, model_output_file, labels_file):
    main(user_embeddings_file, article_embeddings_file, model_output_file, labels_file)

if __name__ == "__main__":
    cli()
