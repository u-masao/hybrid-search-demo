import pandas as pd
from src.format_dataset import create_sentence_column


def test_create_sentence_column():
    data = {
        "title": ["Test Title"],
        "category": ["Test Category"],
        "content": ["Test Content"]
    }
    df = pd.DataFrame(data)
    df = create_sentence_column(df)
    expected_sentence = "# Test Title\n\n**Category:** Test Category\n\nTest Content"
    assert df.iloc[0]["sentence"] == expected_sentence
