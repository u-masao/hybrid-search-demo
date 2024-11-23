import requests

def translation_user(user_id):
    response = requests.post("http://localhost:5000/api/user_translation_search", json={"user_id": user_id})
    return response.json()

def translation_item(item_id):
    response = requests.post("http://localhost:5000/api/item_translation_search", json={"item_id": item_id})
    return response.json()
