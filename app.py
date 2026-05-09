import gradio as gr
import pandas as pd
import os
import sys
from huggingface_hub import hf_hub_download

# Ensure models and paths are accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Checking for required data files...")
repo_id = "Soum25/semantic-search-engine"
revision = "data"

# Download the heavy data files if they aren't present locally
try:
    if not os.path.exists("data/preprocessed_products.csv"):
        print("Downloading dataset...")
        hf_hub_download(repo_id=repo_id, filename="data/preprocessed_products.csv", repo_type="space", revision=revision, local_dir=".")
    if not os.path.exists("search/product.index"):
        print("Downloading FAISS index...")
        hf_hub_download(repo_id=repo_id, filename="search/product.index", repo_type="space", revision=revision, local_dir=".")
    if not os.path.exists("embeddings/product_embeddings.npy"):
        print("Downloading embeddings...")
        hf_hub_download(repo_id=repo_id, filename="embeddings/product_embeddings.npy", repo_type="space", revision=revision, local_dir=".")
except Exception as e:
    print(f"Error downloading data files: {e}")

from search.engine import SemanticSearchEngine

print("Starting Gradio App...")

# Initialize the Search Engine
# Note: HF Spaces might take a few moments to load the models initially
engine = SemanticSearchEngine()

def search_products(query):
    if not query.strip():
        return "Please enter a valid query."
    
    results = engine.search(query, top_k=5)
    
    if not results:
        return "No results found."
    
    html_output = "<div style='display:flex; flex-direction:column; gap:15px;'>"
    for res in results:
        data = res['data']
        title = data.get('title', data.get('product_title', 'Unknown Product'))
        text = data.get('text', '')
        score = res['score']
        
        # Highlight snippet
        text_snippet = text[:300] + "..." if len(text) > 300 else text
        
        html_output += f"""
        <div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9; color: #333;'>
            <h3 style='margin-top: 0; color: #2c3e50;'>{title} <span style='font-size: 0.8em; color: #888;'>(Score: {score:.2f})</span></h3>
            <p style='margin-bottom: 0;'>{text_snippet}</p>
        </div>
        """
    html_output += "</div>"
    
    return html_output

# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Nexus Semantic Search Engine
        A highly accurate, two-stage product discovery engine utilizing **Bi-Encoders** and **Cross-Encoders**.
        Type a descriptive query below to find exactly what you're looking for!
        
        *Example queries: "device to capture memories", "compact laptop for students", "something to clean carpets"*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(placeholder="Enter search query here...", show_label=False, lines=1)
        with gr.Column(scale=1):
            search_button = gr.Button("Search", variant="primary")
            
    results_output = gr.HTML(label="Results")
    
    search_button.click(fn=search_products, inputs=query_input, outputs=results_output)
    query_input.submit(fn=search_products, inputs=query_input, outputs=results_output)

if __name__ == "__main__":
    demo.launch()
