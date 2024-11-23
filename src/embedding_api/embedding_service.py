import torch
from transformers import AutoModel, AutoTokenizer

class EmbeddingService:
    def __init__(self, model_name="intfloat/multilingual-e5-small"):
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer initialized.")

        print("Initializing model...")
        self.model = AutoModel.from_pretrained(model_name)
        print("Model initialized.")

    def embed(self, query_text):
        prefixed_query = f"query: {query_text}"
        inputs = self.tokenizer(prefixed_query, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embedding.tolist()
