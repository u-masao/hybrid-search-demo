import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, embedding_dim):
        super(UserTower, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.network(x)


class ItemTower(nn.Module):
    def __init__(self, embedding_dim):
        super(ItemTower, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.network(x)


class TwoTowerModel(nn.Module):
    def __init__(self, user_embedding_dim, item_embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(user_embedding_dim)
        self.item_tower = ItemTower(item_embedding_dim)

    def forward(self, user_embedding, item_embedding):
        # Ensure the embeddings have the same batch size
        min_batch_size = min(user_embedding.size(0), item_embedding.size(0))
        user_vector = self.user_tower(user_embedding[:min_batch_size])
        item_vector = self.item_tower(item_embedding[:min_batch_size])
        return item_vector, user_vector
