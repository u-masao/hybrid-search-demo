from flask import Flask, request, jsonify
import time
from transformers import AutoModel, AutoTokenizer
import torch

app = Flask(__name__)
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
print("Tokenizer initialized.")

print("Initializing model...")
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
print("Model initialized.")

# Perform a dummy embedding trial
test_input = "This is a test input for embedding."
print("Performing dummy embedding trial...")
inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
dummy_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
print("Dummy embedding trial completed. Embedding size:", dummy_embedding.shape)

@app.route('/embed', methods=['POST'])
def embed():
    start_time = time.time()
    data = request.json
    query_text = data.get('query', '')
    prefixed_query = f"query: {query_text}"
    inputs = tokenizer(prefixed_query, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    response = jsonify({'embedding': embedding.tolist()})
    elapsed_time = time.time() - start_time
    print(f"Response elapsed time: {elapsed_time:.4f} seconds")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
