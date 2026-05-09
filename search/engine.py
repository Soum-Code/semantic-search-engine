import os
import sys
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class SemanticSearchEngine:
    def __init__(self):
        """
        Initializes the semantic search engine by loading data, the FAISS index,
        and the necessary models (bi-encoder and cross-encoder).
        """
        print("Initializing Semantic Search Engine...")
        start_time = time.time()
        
        # Load dataset
        data_path = os.path.join(config.DATA_DIR, "preprocessed_products.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Run preprocess.py first.")
        
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        # Ensure we have clean text
        self.df['text'] = self.df['text'].fillna('')
        
        # Load FAISS index
        index_path = os.path.join(config.SEARCH_DIR, "product.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run indexer.py first.")
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        # Load models
        print(f"Loading Bi-Encoder ({config.ENCODER_MODEL_NAME})...")
        self.bi_encoder = SentenceTransformer(config.ENCODER_MODEL_NAME)
        
        print(f"Loading Cross-Encoder ({config.CROSS_ENCODER_MODEL_NAME})...")
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
        
        print(f"Engine initialized in {time.time() - start_time:.2f} seconds.")

    def search(self, query, top_k=5):
        """
        Performs a semantic search for the given query.
        1. Encodes the query using the bi-encoder.
        2. Retrieves the top candidate vectors from FAISS.
        3. Re-ranks the candidates using the cross-encoder.
        
        Args:
            query (str): The search query.
            top_k (int): Number of final results to return.
            
        Returns:
            list: A list of dictionaries containing product information and scores.
        """
        if not query or not query.strip():
            return []
            
        print(f"\nSearching for: '{query}'")
        search_start = time.time()
        
        # Step 1: Encode the query (must normalize for IndexFlatIP)
        query_embedding = self.bi_encoder.encode([query], normalize_embeddings=True)
        
        # Step 2: Retrieve top candidates from FAISS
        # We retrieve more candidates than top_k for the re-ranking phase
        retrieve_k = min(config.TOP_K_CANDIDATES, len(self.df))
        
        faiss_start = time.time()
        distances, corpus_indices = self.index.search(query_embedding, retrieve_k)
        faiss_time = time.time() - faiss_start
        
        # Gather the candidate texts and data
        candidates = []
        for idx in corpus_indices[0]:
            if idx != -1: # -1 means not enough results in index
                candidates.append(self.df.iloc[idx])
                
        if not candidates:
            return []
            
        # Step 3: Re-rank using Cross-Encoder
        cross_inp = [[query, candidate['text']] for candidate in candidates]
        
        rerank_start = time.time()
        cross_scores = self.cross_encoder.predict(cross_inp)
        rerank_time = time.time() - rerank_start
        
        # Combine results and sort by cross-encoder score
        results = []
        for i, candidate in enumerate(candidates):
            results.append({
                'score': float(cross_scores[i]),
                'data': candidate.to_dict()
            })
            
        # Sort in descending order of score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Take the top_k
        final_results = results[:top_k]
        
        total_time = time.time() - search_start
        print(f"Search completed in {total_time:.3f}s (FAISS: {faiss_time:.3f}s, Re-rank: {rerank_time:.3f}s)")
        
        return final_results

# Simple interactive test mode if run directly
if __name__ == "__main__":
    try:
        engine = SemanticSearchEngine()
        print("\n=== Semantic Search Engine ===")
        print("Type 'exit' or 'quit' to stop.")
        
        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            results = engine.search(query, top_k=3)
            
            print(f"\nTop {len(results)} Results:")
            for i, res in enumerate(results, 1):
                data = res['data']
                title = data.get('title', data.get('product_title', 'Unknown Product'))
                print(f"{i}. Score: {res['score']:.4f} | Title: {title}")
                # Print a snippet of text
                text_snippet = data['text'][:150] + "..." if len(data['text']) > 150 else data['text']
                print(f"   Text: {text_snippet}")
                
    except Exception as e:
        print(f"Error starting search engine: {e}")
