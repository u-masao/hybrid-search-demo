import requests

BASE_URL = "http://localhost:5000"


def test_vector_search():
    response = requests.post(
        f"{BASE_URL}/api/vector_search",
        json={"query_text": "example query"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "item_results" in data
    assert "user_results" in data


def test_user_info():
    response = requests.get(f"{BASE_URL}/api/user_info/A1WNRJMBau9a4UogThL4")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_item_info():
    response = requests.get(f"{BASE_URL}/api/item_info/HFWNRJMBau9a4UogZhKy")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_user_translation_search():
    response = requests.post(
        f"{BASE_URL}/api/user_translation_search",
        json={"translation": [0.1] * 64},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_item_translation_search():
    response = requests.post(
        f"{BASE_URL}/api/item_translation_search",
        json={"translation": [0.1] * 64},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_user_text_embedding_search():
    response = requests.post(
        f"{BASE_URL}/api/user_text_embedding_search",
        json={"embedding": [0.1] * 384},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_item_text_embedding_search():
    response = requests.post(
        f"{BASE_URL}/api/item_text_embedding_search",
        json={"embedding": [0.1] * 384},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_item_hybrid_search():
    response = requests.post(
        f"{BASE_URL}/api/item_hybrid_search",
        json={
            "text": "スポーツ",
            "embedding": [0.1] * 384,
            "translation": [0.2] * 64,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_user_hybrid_search():
    response = requests.post(
        f"{BASE_URL}/api/user_hybrid_search",
        json={
            "text": "スポーツ",
            "embedding": [0.1] * 384,
            "translation": [0.2] * 64,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
