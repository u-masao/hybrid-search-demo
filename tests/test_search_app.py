import requests
import json

BASE_URL = "http://localhost:5000"

def test_vector_search():
    response = requests.post(
        f"{BASE_URL}/api/vector_search",
        json={"query_text": "example query"},
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "item_results" in data
    assert "user_results" in data

def test_user_info():
    response = requests.get(f"{BASE_URL}/api/user_info/1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_item_info():
    response = requests.get(f"{BASE_URL}/api/item_info/1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_user_translation_search():
    response = requests.post(
        f"{BASE_URL}/api/user_translation_search",
        json={"user_id": 1},
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_item_translation_search():
    response = requests.post(
        f"{BASE_URL}/api/item_translation_search",
        json={"item_id": 1},
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
