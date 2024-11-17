import torch
import pandas as pd
from src.two_tower_model import train_two_tower_model

def load_embeddings(file_path):
    return pd.read_parquet(file_path).values

def main(user_embeddings_file, article_embeddings_file, model_output_file):
    user_embeddings = torch.tensor(load_embeddings(user_embeddings_file), dtype=torch.float32)
    article_embeddings = torch.tensor(load_embeddings(article_embeddings_file), dtype=torch.float32)
    
    # Load labels for training
    labels = load_labels()  # Implement this function to load your labels

    model = train_two_tower_model(user_embeddings, article_embeddings, labels)
    torch.save(model.state_dict(), model_output_file)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
