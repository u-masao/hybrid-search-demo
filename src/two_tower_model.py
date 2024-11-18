import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, user_embedding_dim, article_embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
        )
        self.article_tower = nn.Sequential(
            nn.Linear(article_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
        )

    def forward(self, user_embedding, article_embedding):
        # Ensure the embeddings have the same batch size
        min_batch_size = min(user_embedding.size(0), article_embedding.size(0))
        user_vector = self.user_tower(user_embedding[:min_batch_size])
        article_vector = self.article_tower(article_embedding[:min_batch_size])
        return article_vector, user_vector
