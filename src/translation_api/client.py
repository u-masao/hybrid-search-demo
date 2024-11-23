from typing import List

import requests


def _build_url(schema, host, port, api):
    return f"{schema}://{host}:{port}/{api}"


def _post_request(url, data):
    return requests.post(url, json=data)


def translate_user(
    text_embedding: List[float],
    host: str = "localhost",
    port: int = 5002,
    schema="http",
):
    url = _build_url(schema, host, port, "translate/user")
    return _post_request(url, {"embedding": text_embedding})


def translate_item(
    text_embedding: List[float],
    host: str = "localhost",
    port: int = 5002,
    schema="http",
):
    url = _build_url(schema, host, port, "translate/user")
    return _post_request(url, {"embedding": text_embedding})
