import os  # OS操作用

import click
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from src.model.two_tower_model import TwoTowerModel


def load_data(file_path):
    """
    データをファイルからロードし、テンソルに変換します。

    Parameters
    ----------
    file_path : str
        データが保存されているParquetファイルのパス。

    Returns
    -------
    tuple of torch.Tensor
        ユーザー埋め込み、アイテム埋め込み、ラベルを含むテンソルのタプル。
    """
    df = pd.read_parquet(file_path)

    user_embeddings = np.stack(df["user_embedding"].values)
    item_embeddings = np.stack(df["item_embedding"].values)
    labels = df["label"].values

    return (
        torch.tensor(user_embeddings, dtype=torch.float32, requires_grad=True),
        torch.tensor(item_embeddings, dtype=torch.float32, requires_grad=True),
        torch.tensor(labels, dtype=torch.float32),
    )


def main(user_history, model_output_path, epochs, patience=10):
    """
    TwoTowerModelをトレーニングし、結果を保存します。

    Parameters
    ----------
    user_history : str
        ユーザー履歴データが保存されているファイルのパス。
    model_output_path : str
        トレーニング済みモデルを保存するパス。
    epochs : int
        トレーニングのエポック数。
    patience : int, optional
        早期停止のための忍耐値 (デフォルトは10)。
    """
    user_embeddings, item_embeddings, labels = load_data(user_history)

    with mlflow.start_run():
        model = TwoTowerModel(user_embeddings.size(1), item_embeddings.size(1))
        # モデルを初期化
        mlflow.log_param("user_embedding_dim", user_embeddings.size(1))
        mlflow.log_param("item_embedding_dim", item_embeddings.size(1))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # オプティマイザを設定
        criterion = torch.nn.CosineEmbeddingLoss()

        best_loss = float("inf")
        # 最良の損失を無限大に初期化
        patience_counter = 0

        for epoch in range(epochs):
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            optimizer.zero_grad()
            # 勾配をゼロにリセット
            item_vector, user_vector = model(user_embeddings, item_embeddings)
            loss = criterion(item_vector, user_vector, labels)
            loss.backward()
            optimizer.step()

            # Log metrics and print loss
            # 損失をログに記録し、出力
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step=epoch)

            # Check for early stopping
            # 早期停止のチェック
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
        mlflow.pytorch.log_model(model, "two_tower_model")

    # Ensure the directory exists
    # ディレクトリが存在することを確認
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(model.state_dict(), model_output_path)


@click.command()  # コマンドラインインターフェースを定義
@click.argument("user_history", type=click.Path(exists=True))
@click.argument("model_output_path", type=click.Path())
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--patience", default=10, help="Early stopping patience")
def cli(user_history, model_output_path, epochs, patience):
    main(user_history, model_output_path, epochs, patience)


if __name__ == "__main__":
    cli()
