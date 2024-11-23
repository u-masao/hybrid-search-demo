import pytest
from flask import json
from src.search_app.main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_vector_search(client):
    response = client.post(
        "/api/vector_search",
        data=json.dumps({"query_text": "example query"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "item_results" in data
    assert "user_results" in data

def test_user_info(client):
    response = client.get("/api/user_info/1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_item_info(client):
    response = client.get("/api/item_info/1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_user_translation_search(client):
    response = client.post(
        "/api/user_translation_search",
        data=json.dumps({"user_id": 1}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_item_translation_search(client):
    response = client.post(
        "/api/item_translation_search",
        data=json.dumps({"item_id": 1}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
