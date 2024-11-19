import pandas as pd

from src.batch_processing.format_dataset import create_sentence_column


def test_create_sentence_column():
    # テスト用のサンプルDataFrameを準備
    data = {
        "title": ["Test Title"],
        "category": ["Test Category"],
        "content": ["Test Content"],
    }
    df = pd.DataFrame(data)
    # create_sentence_column関数を適用
    df = create_sentence_column(df)
    # 期待される文の形式を定義
    expected_sentence = (
        "# Test Title\n\n**Category:** Test Category\n\nTest Content"
    )
    # 'sentence'列が期待される形式と一致することを確認
    assert df.iloc[0]["sentence"] == expected_sentence
