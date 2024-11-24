# 必要なライブラリをインポート
import numpy as np  # 数値計算用
import pandas as pd
import torch
from tqdm import tqdm

from src.model.two_tower_model import TwoTowerModel

tqdm.pandas()


def load_item_embeddings(file_path):
    """
    アイテムの埋め込みをファイルからロードします。

    Parameters
    ----------
    file_path : str
        埋め込みが保存されているParquetファイルのパス。

    Returns
    -------
    torch.Tensor
        アイテムの埋め込みを含むテンソル。
    """
    df = pd.read_parquet(file_path)
    print("Columns in item embeddings file:", df.columns)
    return torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)


def save_item_translation(df, translations, output_file):
    """
    アイテムの翻訳をDataFrameに追加し、ファイルに保存します。

    Parameters
    ----------
    df : pandas.DataFrame
        アイテムの埋め込みを含むDataFrame。
    translations : numpy.ndarray
        アイテムの翻訳を含む配列。
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


def main(item_embeddings_file, item_translation_file, model_path):
    """
    アイテムの埋め込みをロードし、モデルを使用して翻訳を生成します。

    Parameters
    ----------
    item_embeddings_file : str
        アイテムの埋め込みが保存されているファイルのパス。
    item_translation_file : str
        翻訳結果を保存するファイルのパス。
    model_path : str
        学習済みモデルのパス。
    """
    item_embeddings = load_item_embeddings(item_embeddings_file)
    print("Loaded item embeddings shape:", item_embeddings.shape)

    # モデルをロード
    model = TwoTowerModel(item_embeddings.size(1), item_embeddings.size(1))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 翻訳を生成
    with torch.no_grad():
        item_translations = model.item_tower(item_embeddings).numpy()

    print("Generated item translations shape:", item_translations.shape)
    print("Sample item translations:", item_translations[:5])

    df = pd.read_parquet(item_embeddings_file)
    save_item_translation(df, item_translations, item_translation_file)


if __name__ == "__main__":
    import click  # コマンドライン引数を処理するためのライブラリ

    @click.command()  # コマンドラインインターフェースを定義
    @click.argument("item_embeddings_file", type=click.Path(exists=True))
    @click.argument("item_translation_file", type=click.Path())
    @click.argument("model_path", type=click.Path(exists=True))
    def cli(item_embeddings_file, item_translation_file, model_path):
        main(item_embeddings_file, item_translation_file, model_path)

    cli()
