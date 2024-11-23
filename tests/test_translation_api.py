import pytest
import requests

BASE_URL = "http://localhost:5002"


@pytest.fixture
def user_embedding():
    return {"embedding": [0.1] * 384}


@pytest.fixture
def item_embedding():
    return {"embedding": [0.1] * 384}


def test_translate_user(user_embedding):
    response = requests.post(f"{BASE_URL}/translate/user", json=user_embedding)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 64


def test_translate_item(item_embedding):
    response = requests.post(f"{BASE_URL}/translate/item", json=item_embedding)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 64
