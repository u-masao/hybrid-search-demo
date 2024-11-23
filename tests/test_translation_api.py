import pytest
from src.translation_api.translator import Translator

@pytest.fixture(scope="module")
def translator():
    return Translator("models/two_tower_model.pth", 384)

def test_translate_user(translator):
    translation = translator.translate_user([0.1] * 384)
    assert isinstance(translation, list)
    assert len(translation) > 0

def test_translate_item(translator):
    translation = translator.translate_item([0.1] * 384)
    assert isinstance(translation, list)
    assert len(translation) > 0
