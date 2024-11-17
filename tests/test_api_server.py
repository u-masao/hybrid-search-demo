import requests

def test_embed_endpoint():
    url = "http://localhost:5000/embed"
    payload = {"query": "test query"}
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert 'embedding' in data
    assert isinstance(data['embedding'], list)
    assert len(data['embedding']) > 0

if __name__ == "__main__":
    test_embed_endpoint()
