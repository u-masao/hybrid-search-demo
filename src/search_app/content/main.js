export async function performSearch() {
    const queryText = document.getElementById('query-input').value;
    const queryVector = await fetchEmbedding(queryText);

    const userTextEmbeddingResults = await fetchResults('/api/user_text_embedding_search', { embedding: queryVector });
    const itemTextEmbeddingResults = await fetchResults('/api/item_text_embedding_search', { embedding: queryVector });
    const userBM25Results = await fetchResults('/api/user_hybrid_search', { text: queryText });
    const itemBM25Results = await fetchResults('/api/item_hybrid_search', { text: queryText });

    displayResults('user-text-embedding-results', userTextEmbeddingResults);
    displayResults('item-text-embedding-results', itemTextEmbeddingResults);
    displayResults('user-bm25-results', userBM25Results);
    displayResults('item-bm25-results', itemBM25Results);
}

async function fetchEmbedding(queryText) {
    const response = await fetch('/api/vector_search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: queryText })
    });
    const data = await response.json();
    return data.item_results[0].embedding; // Assuming the first result contains the embedding
}

async function fetchResults(endpoint, body) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return await response.json();
}

function displayResults(elementId, results) {
    const container = document.getElementById(elementId);
    container.innerHTML = results.map(result => `
        <div>
            <p>${result.name || result.id}</p>
            <span class="translation-link" onclick="performTranslationSearch('${result.id}')">Translation Search</span>
        </div>
    `).join('');
}

async function performTranslationSearch(id) {
    const userTranslationResults = await fetchResults('/api/user_translation_search', { translation: id });
    const itemTranslationResults = await fetchResults('/api/item_translation_search', { translation: id });

    alert(`User Translation Results: ${JSON.stringify(userTranslationResults)}\nItem Translation Results: ${JSON.stringify(itemTranslationResults)}`);
}
