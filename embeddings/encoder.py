import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_embeddings():
    """
    Loads preprocessed products, generates embeddings using Sentence-Transformers,
    and saves the embeddings to disk.
    """
    input_path = os.path.join(config.DATA_DIR, "preprocessed_products.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run data/preprocess.py first.")
        return None
        
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Fill any potential NaNs that slipped through
    df['text'] = df['text'].fillna('')
    
    texts = df['text'].tolist()
    print(f"Loaded {len(texts)} products for embedding.")
    
    print(f"Loading encoder model: {config.ENCODER_MODEL_NAME}...")
    # This will download the model if not cached
    model = SentenceTransformer(config.ENCODER_MODEL_NAME)
    
    print(f"Starting embedding generation for {len(texts)} items (this may take a few minutes)...")
    start_time = time.time()
    
    # Generate embeddings
    # Using batch_size from config to optimize memory and speed
    # normalize_embeddings=True is crucial for FAISS IndexFlatIP (Cosine Similarity)
    embeddings = model.encode(
        texts, 
        batch_size=config.BATCH_SIZE, 
        show_progress_bar=True,
        normalize_embeddings=True 
    )
    
    elapsed_time = time.time() - start_time
    print(f"Generated embeddings of shape {embeddings.shape} in {elapsed_time:.2f} seconds.")
    
    # Create directory if it doesn't exist
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    
    # Save embeddings
    output_path = os.path.join(config.EMBEDDINGS_DIR, "product_embeddings.npy")
    np.save(output_path, embeddings)
    print(f"Embeddings saved successfully to {output_path}")
    
    return output_path

if __name__ == "__main__":
    generate_embeddings()
