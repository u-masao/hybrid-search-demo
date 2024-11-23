from src.translation_api.client import translate_item, translate_user


def test_user_translation():
    query_vector = [0.1] * 384
    response = translate_user(query_vector)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 64


def test_item_translation():
    query_vector = [0.2] * 384
    response = translate_item(query_vector)
    assert response.status_code == 200
    data = response.json()
    assert "translation" in data
    assert isinstance(data["translation"], list)
    assert len(data["translation"]) == 64
