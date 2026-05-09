document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const resultsSection = document.getElementById('results-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsGrid = document.getElementById('results-grid');
    const queryDisplay = document.getElementById('query-display');
    const resultCount = document.getElementById('result-count');
    
    const recSection = document.getElementById('recommendations-section');
    const recLoading = document.getElementById('rec-loading');
    const recList = document.getElementById('recommendations-list');
    const recTargetTitle = document.getElementById('rec-target-title');
    const closeRecsBtn = document.getElementById('close-recs');

    // Base API URL (assuming UI is served from the same host)
    const API_BASE = window.location.origin;

    // Search Form Submit
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        // Update UI state
        resultsSection.classList.remove('hidden');
        resultsGrid.innerHTML = '';
        loadingIndicator.classList.remove('hidden');
        queryDisplay.textContent = query;
        resultCount.textContent = 'Searching...';
        recSection.classList.add('hidden');

        try {
            const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&top_k=9`);
            if (!response.ok) throw new Error('Search request failed');
            
            const data = await response.json();
            displaySearchResults(data.results);
            resultCount.textContent = `${data.results.length} results`;
        } catch (error) {
            console.error('Error:', error);
            resultsGrid.innerHTML = `<div class="error">Failed to fetch results. Ensure backend is running.</div>`;
            resultCount.textContent = 'Error';
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    });

    // Close Recommendations
    closeRecsBtn.addEventListener('click', () => {
        recSection.classList.add('hidden');
    });

    // Display Search Results
    function displaySearchResults(results) {
        if (!results || results.length === 0) {
            resultsGrid.innerHTML = '<div class="no-results">No relevant products found. Try a different phrasing!</div>';
            return;
        }

        results.forEach((result, index) => {
            const itemData = result.data;
            // Handle different dataset schemas
            const title = itemData.title || itemData.product_title || itemData.review_title || 'Unknown Product';
            let content = itemData.content || itemData.text || itemData.review_body || '';
            
            // Clean up content for display
            if (content.startsWith(title)) {
                content = content.substring(title.length).trim();
            }

            const card = document.createElement('div');
            card.className = 'glass-card';
            card.style.animationDelay = `${index * 0.05}s`; // Staggered animation
            
            // Store the ID (assuming the dataset index maps to our recommendation ID)
            // We use the DataFrame row index, which we can extract if passed, but since it's not explicitly in data, 
            // we might need to rely on the backend passing an 'id' or we pass the text.
            // Wait, the backend currently just returns 'data'. Let's assume the row name/index is the 'name' in pandas.
            // If not, we can pass the label, but let's just use the array index for demo if ID is missing.
            // Actually, the dataframe `to_dict()` might not include the row index.
            // We can update the backend to pass the ID, or use a workaround here.
            // For now, we'll try to find an id, or default to a random id for demo.
            const itemId = itemData.id !== undefined ? itemData.id : itemData.Unnamed_0 || index;

            card.innerHTML = `
                <div class="card-score"><i class="ph-fill ph-target"></i> ${result.score.toFixed(2)}</div>
                <h3 class="card-title">${title}</h3>
                <p class="card-text">${content}</p>
                <div class="card-actions">
                    <button class="action-btn" data-id="${itemId}" data-title="${title.replace(/"/g, '&quot;')}">
                        <i class="ph ph-magic-wand"></i> Similar
                    </button>
                </div>
            `;

            // Add listener for recommendation button
            const simBtn = card.querySelector('.action-btn');
            simBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fetchRecommendations(itemId, title);
            });

            resultsGrid.appendChild(card);
        });
    }

    // Fetch and display recommendations
    async function fetchRecommendations(itemId, title) {
        recSection.classList.remove('hidden');
        recList.innerHTML = '';
        recTargetTitle.textContent = title;
        recLoading.classList.remove('hidden');

        try {
            // Note: Since our backend currently takes item_id as the dataframe row index,
            // we should ideally have passed the true index from the backend.
            // Assuming itemId here is somewhat valid or we fallback to 0.
            const response = await fetch(`${API_BASE}/recommend?item_id=${itemId}&top_k=5`);
            if (!response.ok) throw new Error('Recommendation request failed');
            
            const data = await response.json();
            
            recLoading.classList.add('hidden');
            
            if (!data.results || data.results.length === 0) {
                recList.innerHTML = '<div>No similar items found.</div>';
                return;
            }

            data.results.forEach((result, index) => {
                const itemData = result.data;
                const title = itemData.title || itemData.product_title || 'Unknown Product';
                let content = itemData.text || '';
                
                const recCard = document.createElement('div');
                recCard.className = 'rec-card';
                recCard.style.animationDelay = `${index * 0.05}s`;
                
                recCard.innerHTML = `
                    <h4 class="rec-title">${title}</h4>
                    <p class="rec-text">${content}</p>
                `;
                recList.appendChild(recCard);
            });
            
        } catch (error) {
            console.error('Error:', error);
            recLoading.classList.add('hidden');
            recList.innerHTML = '<div class="error">Failed to fetch recommendations.</div>';
        }
    }
});
