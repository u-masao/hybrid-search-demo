
## todo

### opened


### stock

- UI案
  - ユーザーをテキスト埋め込みで検索
  - 着目するユーザーを決定
    - とりあえず1人とするが複数混ぜてもOK
  - スコアのブレンド比率を設定
    - アイテムをテキスト埋め込みで検索して候補を取得
    - アイテムをユーザーのTranslationで検索して候補を取得
- embedding, translation api で起動時に初回エンコーディングを実施
- bm25 のスコアがすべて1.0になってしまう問題をなんとかして

### closed

- API
  - search api を作成
    - item, item vector, item translation, item hybrid
    - user, user vector, user translation, user hybrid

- 基本的な検索画面
  - キーワードで検索 ->  text embed
  - User translation で検索 -> similar user, similar item
  - User embedding で検索 -> similar user, similar item
  - User id で検索 -> user
  - Item translation で検索 -> similar user, similar item
  - Item embedding で検索 -> similar user, similar item
  - Item id で検索 -> item
