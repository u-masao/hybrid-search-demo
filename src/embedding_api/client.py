import requests


def text_embedding(
    text: str, schema: str = "http", host: str = "localhost", port: int = 5001
):
    BASE_URL = f"{schema}://{host}:{port}"
    response = requests.post(f"{BASE_URL}/embed", json={"query": text})
    return response.json()
