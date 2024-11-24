/**
 * performSearch
 * 検索を実行する関数
 * 
 * @returns {Promise<void>}
 */
export async function performSearch() {

    // クエリテキストを取得
    const queryText = document.getElementById('query-input').value;
    // クエリテキストの埋め込みを取得
    const queryVector = await fetchEmbedding(queryText);

    // 各種検索を実行
    userTextEmbeddingSearch(queryVector);
    itemTextEmbeddingSearch(queryVector);
    userBM25Search(queryText);
    itemBM25Search(queryText);
}
window.performSearch = performSearch;

/**
 * userTranslationSearch
 * ユーザー翻訳検索を実行する関数
 * 
 * @param {Array} queryVector - クエリの翻訳ベクトル
 * @returns {Promise<void>}
 */
async function userTranslationSearch(queryVector) {
    const results = await fetchResults('/api/user_translation_search', { translation: queryVector });
    displayResults('user-translation-results', results);
}
window.userTranslationSearch = userTranslationSearch;

/**
 * itemTranslationSearch
 * アイテム翻訳検索を実行する関数
 * 
 * @param {Array} queryVector - クエリの翻訳ベクトル
 * @returns {Promise<void>}
 */
async function itemTranslationSearch(queryVector) {
    const results = await fetchResults('/api/item_translation_search', { translation: queryVector });
    displayResults('item-translation-results', results);
}
window.itemTranslationSearch = itemTranslationSearch;

/**
 * userTextEmbeddingSearch
 * ユーザーテキスト埋め込み検索を実行する関数
 * 
 * @param {Array} queryVector - クエリの埋め込みベクトル
 * @returns {Promise<void>}
 */
async function userTextEmbeddingSearch(queryVector) {
    const result = await fetchResults('/api/user_text_embedding_search', { embedding: queryVector });
    displayResults('user-text-embedding-results', result);
}
window.userTextEmbeddingSearch = userTextEmbeddingSearch;

/**
 * itemTextEmbeddingSearch
 * アイテムテキスト埋め込み検索を実行する関数
 * 
 * @param {Array} queryVector - クエリの埋め込みベクトル
 * @returns {Promise<void>}
 */
async function itemTextEmbeddingSearch(queryVector) {
    const itemTextEmbeddingResults = await fetchResults('/api/item_text_embedding_search', { embedding: queryVector });
    displayResults('item-text-embedding-results', itemTextEmbeddingResults);
}
window.itemTextEmbeddingSearch = itemTextEmbeddingSearch;

/**
 * userBM25Search
 * ユーザーBM25検索を実行する関数
 * 
 * @param {string} queryText - クエリテキスト
 * @returns {Promise<void>}
 */
async function userBM25Search(queryText) {
    const userBM25Results = await fetchResults('/api/user_hybrid_search', { query_text: queryText , text_weight: 1.0});
    displayResults('user-bm25-results', userBM25Results);
}
window.userBM25Search= userBM25Search;

/**
 * itemBM25Search
 * アイテムBM25検索を実行する関数
 * 
 * @param {string} queryText - クエリテキスト
 * @returns {Promise<void>}
 */
async function itemBM25Search(queryText) {
    const itemBM25Results = await fetchResults('/api/item_hybrid_search', { query_text: queryText , text_weight: 1.0});
    displayResults('item-bm25-results', itemBM25Results);
}
window.itemBM25Search= itemBM25Search;

/**
 * fetchEmbedding
 * テキストの埋め込みを取得する関数
 * 
 * @param {string} queryText - クエリテキスト
 * @returns {Promise<Array>} - 埋め込みベクトル
 */
async function fetchEmbedding(queryText) {
    const response = await fetch('/api/text_embedding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: queryText })
    });
    const data = await response.json();
    return data.embedding;
}

/**
 * fetchResults
 * 検索結果を取得する関数
 * 
 * @param {string} endpoint - APIエンドポイント
 * @param {Object} body - リクエストボディ
 * @returns {Promise<Object>} - 検索結果
 */
async function fetchResults(endpoint, body) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return await response.json();
}

/**
 * displayResults
 * 検索結果を表示する関数
 * 
 * @param {string} elementId - 結果を表示する要素のID
 * @param {Array} results - 検索結果
 */
function displayResults(elementId, results) {
    const container = document.getElementById(elementId);
    container.innerHTML = `<p>${elementId}</p>`+ results.map(result => `
        <div class='item-summary'>
            <dl>
                <dt>sentence</dt>
                <dd>${result._source.sentence.slice(0,100)}</dd>

                <dt>score</dt>
                <dd>${result._score}</dd>

                <dt>translation</dt>
                <dd>
                    <span class="button"
                        onclick="userTranslationSearch([${result._source.translation}])">user search</span>
                    <span class="button"
                        onclick="itemTranslationSearch([${result._source.translation}])">item search</span>
                </dd>

                <dt>text_embedding</dt>
                <dd>
                    <span class="button"
                        onclick="userTextEmbeddingSearch([${result._source.embedding}])">user search</span>
                    <span class="button"
                        onclick="itemTextEmbeddingSearch([${result._source.embedding}])">item search</span>
                </dd>

                <dt>action</dt><dd>
                    <span class="button"
                        onclick="displayUserItemDetail('${result._index}','${result._id}')">view detail</span>
                    <span class="button"
                        onclick="alert('未実装','${result._index}','${result._id}')">select</span>
                </dd>

            </dl>
        </div>
    `).join('');
}

/**
 * displayUserItemDetail
 * ユーザーまたはアイテムの詳細を表示する関数
 * 
 * @param {string} index - インデックス名
 * @param {string} id - ユーザーまたはアイテムのID
 * @returns {Promise<void>}
 */
async function displayUserItemDetail(index, id) {
    let api;

    if (index==='item_develop') {
        api = '/api/item_info/';
    } else {
        api = '/api/user_info/';
    }

    const info = await fetchResults(api, { id: id});
    document.getElementById('popupContent').innerHTML = objectToString(info);
    document.getElementById('detailPopup').style.display='block';
}
window.displayUserItemDetail = displayUserItemDetail;

/**
 * hidePopup
 * ポップアップを非表示にする関数
 */
function hidePopup() {
    document.getElementById('detailPopup').style.display = "none";
}
window.hidePopup = hidePopup;

/**
 * objectToString
 * オブジェクトを文字列に変換する関数
 * 
 * @param {Object} obj - 変換するオブジェクト
 * @param {string} indent - インデント用の文字列
 * @returns {string} - 変換された文字列
 */
function objectToString(obj, indent = "") {
  let message = "";
  for (let key in obj) {
    if (typeof obj[key] === 'object') {
      message += indent + key + ": {<br>";
      message += objectToString(obj[key], indent + "  ");
      message += indent + "}<br>";
    } else if (Array.isArray(obj[key])) {
      message += indent + key + ": " + obj[key].join(", ") + "<br>";
    } else {
      message += indent + key + ": " + obj[key] + "<br>";
    }
  }
  return message;
}

