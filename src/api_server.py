from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('multilingual-e5-small')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    query_text = data.get('query', '')
    prefixed_query = f"query: {query_text}"
    embedding = model.encode(prefixed_query)
    return jsonify({'embedding': embedding.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
