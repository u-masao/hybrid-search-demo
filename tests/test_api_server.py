import requests

def is_server_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def test_embed_endpoint():
    url = "http://localhost:5000/embed"
    if not is_server_running("http://localhost:5000"):
        raise RuntimeError("API server is not running. Please start the server and try again.")
    
    payload = {"query": "test query"}
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert 'embedding' in data
    assert isinstance(data['embedding'], list)
    assert len(data['embedding']) > 0

if __name__ == "__main__":
    test_embed_endpoint()
