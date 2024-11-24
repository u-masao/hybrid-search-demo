# 必要なライブラリをインポート
import numpy as np  # 数値計算用
import pandas as pd
import torch
from tqdm import tqdm

from src.model.two_tower_model import TwoTowerModel

tqdm.pandas()


def load_user_embeddings(file_path):
    """
    ユーザーの埋め込みをファイルからロードします。

    Parameters
    ----------
    file_path : str
        埋め込みが保存されているParquetファイルのパス。

    Returns
    -------
    torch.Tensor
        ユーザーの埋め込みを含むテンソル。
    """
    df = pd.read_parquet(file_path)
    print("Columns in user embeddings file:", df.columns)
    return torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)


def save_user_translation(df, translations, output_file):
    """
    ユーザーの翻訳をDataFrameに追加し、ファイルに保存します。

    Parameters
    ----------
    df : pandas.DataFrame
        ユーザーの埋め込みを含むDataFrame。
    translations : numpy.ndarray
        ユーザーの翻訳を含む配列。
    output_file : str
        結果を保存するParquetファイルのパス。
    """
    print("DataFrame shape before adding translations:", df.shape)
    print("Translations shape:", translations.shape)
    df = df.reset_index(drop=True)
    df["translation"] = df.progress_apply(
        lambda row: translations[row.name], axis=1
    )
    df.to_parquet(output_file, index=False)


def main(user_embeddings_file, user_translation_file, model_path):
    """
    ユーザーの埋め込みをロードし、モデルを使用して翻訳を生成します。

    Parameters
    ----------
    user_embeddings_file : str
        ユーザーの埋め込みが保存されているファイルのパス。
    user_translation_file : str
        翻訳結果を保存するファイルのパス。
    model_path : str
        学習済みモデルのパス。
    """
    user_embeddings = load_user_embeddings(user_embeddings_file)
    print("Loaded user embeddings shape:", user_embeddings.shape)

    # モデルをロード
    model = TwoTowerModel(user_embeddings.size(1), user_embeddings.size(1))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 翻訳を生成
    with torch.no_grad():
        user_translations = model.user_tower(user_embeddings).numpy()

    print("Generated user translations shape:", user_translations.shape)
    print("Sample user translations:", user_translations[:5])

    df = pd.read_parquet(user_embeddings_file)
    save_user_translation(df, user_translations, user_translation_file)


if __name__ == "__main__":
    import click  # コマンドライン引数を処理するためのライブラリ

    @click.command()  # コマンドラインインターフェースを定義
    @click.argument("user_embeddings_file", type=click.Path(exists=True))
    @click.argument("user_translation_file", type=click.Path())
    @click.argument("model_path", type=click.Path(exists=True))
    def cli(user_embeddings_file, user_translation_file, model_path):
        main(user_embeddings_file, user_translation_file, model_path)

    cli()
