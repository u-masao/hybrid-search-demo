import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model.two_tower_model import TwoTowerModel

tqdm.pandas()


def load_item_embeddings(file_path):
    df = pd.read_parquet(file_path)
    print("Columns in item embeddings file:", df.columns)
    return torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)


def save_item_translation(df, translations, output_file):
    print("DataFrame shape before adding translations:", df.shape)
    print("Translations shape:", translations.shape)
    df = df.reset_index(drop=True)
    df["translation"] = df.progress_apply(
        lambda row: translations[row.name], axis=1
    )
    df.to_parquet(output_file, index=False)


def main(item_embeddings_file, item_translation_file, model_path):
    item_embeddings = load_item_embeddings(item_embeddings_file)
    print("Loaded item embeddings shape:", item_embeddings.shape)

    # Load the model
    model = TwoTowerModel(
        item_embeddings.size(1), item_embeddings.size(1)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate translations
    with torch.no_grad():
        item_translations = model.item_tower(item_embeddings).numpy()

    print("Generated item translations shape:", item_translations.shape)
    print("Sample item translations:", item_translations[:5])

    df = pd.read_parquet(item_embeddings_file)
    save_item_translation(
        df, item_translations, item_translation_file
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2], sys.argv[3])
