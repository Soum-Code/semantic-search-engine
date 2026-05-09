import os
import sys
import numpy as np
import faiss
import time

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_index():
    """
    Loads product embeddings and builds a FAISS index for fast similarity search.
    """
    embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "product_embeddings.npy")
    
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found. Run embeddings/encoder.py first.")
        return None
        
    print(f"Loading embeddings from {embeddings_path}...")
    start_time = time.time()
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings of shape {embeddings.shape} in {time.time() - start_time:.2f} seconds.")
    
    dimension = embeddings.shape[1]
    
    # Using IndexFlatIP since embeddings are normalized (Cosine Similarity)
    print(f"Building FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    
    start_time = time.time()
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to the FAISS index in {time.time() - start_time:.2f} seconds.")
    
    # Create search directory if it doesn't exist
    os.makedirs(config.SEARCH_DIR, exist_ok=True)
    
    # Save the index
    index_path = os.path.join(config.SEARCH_DIR, "product.index")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved successfully to {index_path}")
    
    return index_path

if __name__ == "__main__":
    build_index()
