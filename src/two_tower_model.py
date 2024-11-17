import torch.nn as nn
import torch.optim as optim


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
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, user_embedding, article_embedding):
        # Ensure the embeddings have the same batch size
        min_batch_size = min(user_embedding.size(0), article_embedding.size(0))
        user_vector = self.user_tower(user_embedding[:min_batch_size])
        article_vector = self.article_tower(article_embedding[:min_batch_size])
        return self.cosine_similarity(user_vector, article_vector)


def train_two_tower_model(
    user_embeddings, article_embeddings, labels, epochs=30, lr=0.0001
):
    # Ensure the number of user, article embeddings, and labels match
    min_samples = min(user_embeddings.shape[0], article_embeddings.shape[0], labels.shape[0])
    user_embeddings = user_embeddings[:min_samples]
    article_embeddings = article_embeddings[:min_samples]
    labels = labels[:min_samples]

    model = TwoTowerModel(
        user_embeddings.shape[1], article_embeddings.shape[1]
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(user_embeddings, article_embeddings)
        loss = criterion(outputs.view(-1, 1), labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model
