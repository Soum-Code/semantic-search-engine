import os
import sys
import pandas as pd
from datasets import load_dataset

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_and_preprocess_data():
    """
    Loads the Amazon reviews dataset from HuggingFace, preprocesses it, 
    extracts unique products, and limits to MAX_PRODUCTS.
    """
    print(f"Loading dataset: {config.DATASET_NAME} (config: {config.DATASET_CONFIG})...")
    try:
        # Load the dataset
        dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT)
        
        # Convert to pandas dataframe
        df = dataset.to_pandas()
        print(f"Original dataset size (reviews): {len(df)}")
        
        # Drop duplicates based on product_id or title to ensure we are searching over unique products
        if 'product_id' in df.columns:
            df = df.drop_duplicates(subset=['product_id'])
            print(f"Size after deduplicating products: {len(df)}")
        elif 'title' in df.columns:
            df = df.drop_duplicates(subset=['title'])
            print(f"Size after deduplicating by title: {len(df)}")
        
        # Sample down to MAX_PRODUCTS
        if len(df) > config.MAX_PRODUCTS:
            df = df.sample(n=config.MAX_PRODUCTS, random_state=42).reset_index(drop=True)
            print(f"Sampled down to: {len(df)} products")
            
        # Create a combined 'text' column for embedding
        text_parts = []
        if 'product_category' in df.columns:
            text_parts.append(df['product_category'].fillna(''))
        if 'review_title' in df.columns:
            text_parts.append(df['review_title'].fillna(''))
        if 'review_body' in df.columns:
            text_parts.append(df['review_body'].fillna(''))
        if 'title' in df.columns and 'content' in df.columns:
            text_parts = [df['title'].fillna(''), df['content'].fillna('')]
            
        if text_parts:
            df['text'] = pd.concat(text_parts, axis=1).agg(' '.join, axis=1)
        else:
            # Fallback for unexpected schema
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            df['text'] = df[string_cols].fillna('').agg(' '.join, axis=1)
            
        # Clean up whitespace
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Remove empty texts
        df = df[df['text'].astype(bool)]
        print(f"Final preprocessed dataset size: {len(df)}")
        
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # Save to csv for future steps
        output_path = os.path.join(config.DATA_DIR, "preprocessed_products.csv")
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error loading or preprocessing dataset: {e}")
        return None

if __name__ == "__main__":
    load_and_preprocess_data()
