import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from two_tower_model import TwoTowerModel

tqdm.pandas()

def load_article_embeddings(file_path):
    df = pd.read_parquet(file_path)
    print("Columns in article embeddings file:", df.columns)
    return torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)

def save_article_translation(df, translations, output_file):
    print("DataFrame shape before adding translations:", df.shape)
    print("Translations shape:", translations.shape)
    df["translation"] = df.index.map(lambda idx: translations[idx]).astype(np.float32)
    df.to_parquet(output_file, index=False)

def main(article_embeddings_file, article_translation_file, model_path):
    article_embeddings = load_article_embeddings(article_embeddings_file)
    print("Loaded article embeddings shape:", article_embeddings.shape)
    
    # Load the model
    model = TwoTowerModel(article_embeddings.size(1), article_embeddings.size(1))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate translations
    with torch.no_grad():
        article_translations = model.article_tower(article_embeddings).numpy()

    print("Generated article translations shape:", article_translations.shape)
    print("Sample article translations:", article_translations[:5])
    
    df = pd.read_parquet(article_embeddings_file)
    save_article_translation(df, article_translations, article_translation_file)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
