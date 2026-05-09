import os
import sys
import numpy as np

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from search.engine import SemanticSearchEngine

class ProductRecommender:
    def __init__(self, engine: SemanticSearchEngine = None):
        """
        Initializes the recommender.
        If an engine is provided, it reuses the loaded models and data to save memory.
        """
        self.engine = engine if engine else SemanticSearchEngine()
        
        # Load the raw embeddings to avoid having to re-encode the target item
        embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "product_embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
            
        print("Loading embeddings for fast recommendation lookup...")
        self.embeddings = np.load(embeddings_path)

    def recommend(self, item_id: int, top_k: int = 5):
        """
        Recommends similar items based on a given item ID (row index).
        
        Args:
            item_id (int): The index of the item in the dataset.
            top_k (int): Number of recommendations to return.
            
        Returns:
            list: Recommended products with scores.
        """
        if item_id < 0 or item_id >= len(self.engine.df):
            raise ValueError(f"Invalid item_id {item_id}. Must be between 0 and {len(self.engine.df)-1}")
            
        # Get the pre-computed embedding for the target item
        target_embedding = self.embeddings[item_id:item_id+1]
        target_text = self.engine.df.iloc[item_id]['text']
        
        # We search for top_k + 1 candidates because the item itself will be in the results
        retrieve_k = min(config.TOP_K_CANDIDATES, len(self.engine.df))
        distances, corpus_indices = self.engine.index.search(target_embedding, retrieve_k)
        
        candidates = []
        for idx in corpus_indices[0]:
            if idx != -1 and idx != item_id: # Exclude the target item itself
                candidates.append(self.engine.df.iloc[idx])
                
        if not candidates:
            return []
            
        # Re-rank using Cross-Encoder to ensure high contextual similarity
        cross_inp = [[target_text, candidate['text']] for candidate in candidates]
        cross_scores = self.engine.cross_encoder.predict(cross_inp)
        
        results = []
        for i, candidate in enumerate(candidates):
            results.append({
                'score': float(cross_scores[i]),
                'data': candidate.to_dict()
            })
            
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    recommender = ProductRecommender()
    print("\nTest Recommendation for item_id=0")
    print("Original Item:", recommender.engine.df.iloc[0]['title'])
    
    recs = recommender.recommend(0, top_k=3)
    for i, r in enumerate(recs, 1):
        print(f"\n{i}. Score: {r['score']:.4f}")
        print(f"Title: {r['data'].get('title', 'Unknown')}")
