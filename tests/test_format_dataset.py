import pandas as pd

from src.format_dataset import create_sentence_column


def test_create_sentence_column():
    # Prepare a sample DataFrame for testing
    data = {
        "title": ["Test Title"],
        "category": ["Test Category"],
        "content": ["Test Content"],
    }
    df = pd.DataFrame(data)
    # Apply the create_sentence_column function
    df = create_sentence_column(df)
    # Define the expected sentence format
    expected_sentence = (
        "# Test Title\n\n**Category:** Test Category\n\nTest Content"
    )
    # Assert that the 'sentence' column matches the expected format
    assert df.iloc[0]["sentence"] == expected_sentence
