import numpy as np
import pandas as pd
import torch
from two_tower_model import TwoTowerModel

def load_user_embeddings(file_path):
    df = pd.read_parquet(file_path)
    print("Columns in user embeddings file:", df.columns)
    return torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)

def save_user_translation(df, translations, output_file):
    print("DataFrame shape before adding translations:", df.shape)
    print("Translations shape:", translations.shape)
    df["user_translation"] = translations.tolist()
    df.to_parquet(output_file, index=False)

def main(user_embeddings_file, user_translation_file, model_path):
    user_embeddings = load_user_embeddings(user_embeddings_file)
    print("Loaded user embeddings shape:", user_embeddings.shape)
    
    # Load the model
    model = TwoTowerModel(user_embeddings.size(1), user_embeddings.size(1))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate translations
    with torch.no_grad():
        user_translations = model.user_tower(user_embeddings).numpy()

    print("Generated user translations shape:", user_translations.shape)
    print("Sample user translations:", user_translations[:5])
    
    df = pd.read_parquet(user_embeddings_file)
    save_user_translation(df, user_translations, user_translation_file)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
