import requests

def text_embedding(text):
    response = requests.post("http://localhost:5000/api/vector_search", json={"query_text": text})
    return response.json()
