import os  # OS操作用
import random

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# モジュールの説明:
# このモジュールは、架空のユーザープロファイルを生成し、指定されたファイルに保存するための機能を提供します。

# 環境変数をロード
# .credentialファイルから環境変数を読み込む
# .envファイルから環境変数を読み込む
load_dotenv()


def generate_user_profiles(num_users=1000, categories=None):
    """
    架空のユーザープロファイルを生成します。

    Parameters
    ----------
    num_users : int, optional
        ユニークなユーザーの数 (デフォルトは1000)。
    categories : list of str, optional
        ユーザーの好みのカテゴリのリスト (デフォルトはNone)。

    Returns
    -------
    dict
        ユーザープロファイルの辞書。
    """
    if categories is None:
        categories = []

    # ユーザーIDのリストを生成
    # 1からnum_usersまでのIDを生成
    user_ids = [i for i in range(1, num_users + 1)]
    # 性別の選択肢
    genders = ["male", "female", "non-binary"]

    # OpenAIクライアントを初期化
    # OpenAI APIを使用して自己紹介文を生成
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    user_profiles = {}
    for user_id in tqdm(user_ids, desc="Generating user profiles"):
        # 年齢をランダムに決定
        # 18から70の間でランダムに選択
        age = random.randint(18, 70)
        # 性別をランダムに選択
        # 性別の選択肢からランダムに選択
        gender = random.choice(genders)
        # 好みのカテゴリをランダムに選択
        # カテゴリからランダムに選択
        preferences = random.sample(
            categories, k=random.randint(1, min(3, len(categories)))
        )
        # 自己紹介文を生成
        # OpenAI APIを使用して生成
        introduction = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"50文字以内の短い自己紹介文を作って。"
                            "具体例を挙げて個性的な趣味を語って。"
                            f" 年齢は {age} 歳。性別は {gender}。"
                            f" 好きなものは{', '.join(preferences)}"
                        ),
                    }
                ],
                max_tokens=50,
                temperature=0.9,
            )
            .choices[0]
            .message.content.strip()
        )

        # プロファイルの文を作成
        # プロファイル情報をフォーマット
        sentence = (
            f"**Age**: {age}\n\n"
            f"**Gender**: {gender}\n\n"
            f"**Preferences**: {', '.join(preferences)}\n\n"
            f"**Introduction**: {introduction}"
        )

        # ユーザープロファイルを辞書に追加
        # 辞書にプロファイルを追加
        user_profiles[user_id] = {
            "id": user_id,
            "sentence": sentence,
            "age": age,
            "gender": gender,
            "preferences": preferences,
            "introduction": introduction,
        }

    return user_profiles


@click.command()  # コマンドラインインターフェースを定義
@click.option(
    "--num_users", default=1000, help="生成するユニークなユーザーの数。"
)
@click.option(
    "--categories",
    default="technology,sports,music,art",
    help="ユーザーの好みのカテゴリのカンマ区切りリスト。",
)
@click.argument("output_file", type=click.Path())
def main(num_users, categories, output_file):
    """ユーザープロファイルを生成し、Parquetファイルに保存します。"""
    # カテゴリをリストに変換
    categories_list = categories.split(",")
    # ユーザープロファイルを生成
    user_profiles = generate_user_profiles(
        num_users=num_users, categories=categories_list
    )
    # データフレームに変換
    df_user_profiles = pd.DataFrame.from_dict(user_profiles, orient="index")
    # Parquetファイルに保存
    df_user_profiles.to_parquet(output_file)


if __name__ == "__main__":
    main()
