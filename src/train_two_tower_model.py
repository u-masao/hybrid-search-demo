import click
import pandas as pd
import torch

from two_tower_model import TwoTowerModel


def load_embeddings(file_path):
    df = pd.read_parquet(file_path)
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    return numeric_df.astype("float32").to_numpy()


def load_labels(labels_file):
    # Load labels from a Parquet file
    labels_df = pd.read_parquet(labels_file)
    print("Columns in labels file:", labels_df.columns)  # Debug statement
    if "label" not in labels_df.columns:
        raise KeyError("The 'label' column is missing from the labels file.")
    return torch.tensor(
        labels_df["label"].astype("float32").values, dtype=torch.float32
    )


def main(
    user_embeddings_file,
    article_embeddings_file,
    model_output_file,
    labels_file,
):
    user_embeddings = torch.tensor(
        load_embeddings(user_embeddings_file), dtype=torch.float32
    )
    article_embeddings = torch.tensor(
        load_embeddings(article_embeddings_file), dtype=torch.float32
    )

    # Load labels for training
    labels = load_labels(labels_file)

    model = TwoTowerModel(user_embeddings.size(1), article_embeddings.size(1))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(30):  # Assuming 30 epochs
        optimizer.zero_grad()
        outputs = model(user_embeddings, article_embeddings)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid to normalize outputs
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_output_file)


@click.command()
@click.argument("user_embeddings_file", type=click.Path(exists=True))
@click.argument("article_embeddings_file", type=click.Path(exists=True))
@click.argument(
    "labels_file", type=click.Path(), default="data/user_history.parquet"
)
@click.argument("model_output_file", type=click.Path())
def cli(
    user_embeddings_file,
    article_embeddings_file,
    labels_file,
    model_output_file,
):
    main(
        user_embeddings_file,
        article_embeddings_file,
        model_output_file,
        labels_file,
    )


if __name__ == "__main__":
    cli()
