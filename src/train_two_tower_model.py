import torch
import pandas as pd
from two_tower_model import train_two_tower_model

def load_embeddings(file_path):
    return pd.read_parquet(file_path).values

def load_labels(labels_file):
    # Implement the logic to load labels from a file
    # For example, assuming labels are stored in a CSV file
    import pandas as pd
    labels_df = pd.read_csv(labels_file)
    return torch.tensor(labels_df['label'].values, dtype=torch.float32)

def main(user_embeddings_file, article_embeddings_file, model_output_file, labels_file):
    user_embeddings = torch.tensor(load_embeddings(user_embeddings_file), dtype=torch.float32)
    article_embeddings = torch.tensor(load_embeddings(article_embeddings_file), dtype=torch.float32)
    
    # Load labels for training
    labels = load_labels(labels_file)

    model = train_two_tower_model(user_embeddings, article_embeddings, labels)
    torch.save(model.state_dict(), model_output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python train_two_tower_model.py <user_embeddings_file> <article_embeddings_file> <model_output_file> <labels_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
