import requests
import subprocess
import time

def is_server_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def test_embed_endpoint():
    url = "http://localhost:5000/embed"
    if not is_server_running("http://localhost:5000"):
        # Start the server in a subprocess
        server_process = subprocess.Popen(
            ["poetry", "run", "python", "src/api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait for the server to start
        time.sleep(5)
    
    payload = {"query": "test query"}
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert 'embedding' in data
    assert isinstance(data['embedding'], list)
    assert len(data['embedding']) > 0

    if 'server_process' in locals():
        # Terminate the server process after the test
        server_process.terminate()
        server_process.wait()
