import pytest
import json
from flask import Flask
from src.translation_api.api import app, main

@pytest.fixture(scope='module', autouse=True)
def setup_model():
    main(['--model-path', 'models/two_tower_model.pth'])

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_translate_user(client):
    response = client.post('/translate/user', json={
        "embedding": [0.1] * 384
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "translation" in data

def test_translate_item(client):
    response = client.post('/translate/item', json={
        "embedding": [0.1] * 384
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "translation" in data
