import os

import click
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from two_tower_model import TwoTowerModel


def load_data(file_path):
    df = pd.read_parquet(file_path)

    user_embeddings = np.stack(df["user_embedding"].values)
    article_embeddings = np.stack(df["article_embedding"].values)
    labels = df["label"].values

    return (
        torch.tensor(user_embeddings, dtype=torch.float32, requires_grad=True),
        torch.tensor(
            article_embeddings, dtype=torch.float32, requires_grad=True
        ),
        torch.tensor(labels, dtype=torch.float32),
    )


def main(user_history, model_output_path, epochs):
    user_embeddings, article_embeddings, labels = load_data(user_history)

    with mlflow.start_run():
        model = TwoTowerModel(
            user_embeddings.size(1), article_embeddings.size(1)
        )
        mlflow.log_param("user_embedding_dim", user_embeddings.size(1))
        mlflow.log_param("article_embedding_dim", article_embeddings.size(1))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CosineEmbeddingLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            article_vector, user_vector = model(
                user_embeddings, article_embeddings
            )
            loss = criterion(article_vector, user_vector, labels)
            loss.backward()
            optimizer.step()

            # Log metrics and print loss
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step=epoch)

        # Log the model
        mlflow.pytorch.log_model(model, "two_tower_model")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(model.state_dict(), model_output_path)


@click.command()
@click.argument("user_history", type=click.Path(exists=True))
@click.argument("model_output_path", type=click.Path())
@click.option("--epochs", default=10, help="Number of training epochs")
def cli(user_history, model_output_path, epochs):
    main(user_history, model_output_path, epochs)


if __name__ == "__main__":
    cli()
