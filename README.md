# Semantic Search Engine

Live Demo: [Try it here](https://huggingface.co/spaces/Soum25/semantic-search-engine)

A state-of-the-art semantic search and recommendation engine designed to deliver highly relevant product discovery. This project implements a sophisticated two-stage retrieval architecture (Bi-Encoder retrieval followed by Cross-Encoder re-ranking) and provides a fast REST API alongside a dynamic web user interface.

## About

Traditional keyword-based search engines often fail to capture the true intent of a user's query, especially when dealing with complex or conversational phrasing. If a user searches for "a device to capture memories," a traditional system might struggle if the exact word "camera" is not used. 

This Semantic Search Engine solves that problem by understanding the underlying meaning and context of queries. By leveraging deep learning models, it maps both products and user queries into a shared mathematical vector space. Products that are conceptually similar are located closer together in this space. 

Furthermore, this engine goes beyond simple vector similarity by implementing a powerful two-stage pipeline:
1. **Initial Retrieval (Speed)**: A fast Bi-Encoder generates an initial pool of relevant candidates from a database of thousands of products in milliseconds using FAISS.
2. **Re-ranking (Accuracy)**: A highly accurate Cross-Encoder re-evaluates the relationship between the query and each candidate, effectively reading them side-by-side to ensure the final results are contextually perfect.

This makes the engine highly resilient to typos, synonyms, and descriptive queries, providing an unparalleled user experience for e-commerce discovery and content retrieval.

This project is proudly developed and maintained by [@Soum-Code](https://github.com/Soum-Code) as an open-source initiative. You can find the complete source code and documentation in the [semantic-search-engine](https://github.com/Soum-Code/semantic-search-engine) repository.

## Results

| Metric | Bi-Encoder Only | + Cross-Encoder Re-ranking |
|---|---|---|
| Query latency | ~38ms | ~195ms |
| Products indexed | 9,671 | 9,671 |
| Top-5 relevance (manual eval) | Good | Excellent |

Tested on the preprocessed Amazon product dataset.
Cross-encoder re-ranking adds ~157ms latency but significantly improves contextual accuracy on complex queries like "device to capture memories."

## Architecture

This project is built using Python, FastAPI, FAISS, and Sentence-Transformers, structured into four distinct phases of operation:

1. **Preprocessing**: Cleans and merges product datasets, combining titles, features, and reviews into a unified semantic text format.
2. **Embedding Generation & Indexing**: Uses a high-speed Bi-Encoder (`all-MiniLM-L6-v2`) to convert product descriptions into dense vector embeddings. These embeddings are stored in a highly optimized FAISS index (`IndexFlatIP`) for sub-millisecond similarity search.
3. **Search & Re-ranking**: When a query is submitted, the engine retrieves top candidate vectors via FAISS and then re-evaluates them using a powerful Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to dramatically improve contextual accuracy.
4. **API & Frontend UI**: A FastAPI service exposes the engine via REST endpoints, while a responsive, modern HTML/CSS/JS frontend provides a rich user experience featuring instant search and product recommendations.

## Project Structure

```text
semantic-search-engine/
├── api/
│   └── main.py              # FastAPI server and endpoints
├── data/
│   └── preprocess.py        # Data cleaning and aggregation pipeline
├── embeddings/
│   └── encoder.py           # Generates product embeddings
├── recommendation/
│   └── recommender.py       # Recommendation engine utilizing the FAISS index
├── search/
│   ├── engine.py            # Core search pipeline (Bi-Encoder + Cross-Encoder)
│   └── indexer.py           # FAISS index compilation
├── ui/
│   ├── app.js               # Frontend JavaScript logic
│   ├── index.html           # Main user interface
│   └── style.css            # UI styling and glassmorphism design
├── config.py                # Global configuration and hyperparameter settings
└── requirements.txt         # Project dependencies
```

## Installation

Ensure you have Python 3.10+ installed.

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup and Execution

To run the full pipeline from scratch, execute the following commands in order:

### 1. Preprocess the Data
Prepare the raw product dataset for embedding:
```bash
python data/preprocess.py
```

### 2. Generate Embeddings
Encode the product data into dense vectors (this process is computationally intensive):
```bash
python embeddings/encoder.py
```

### 3. Build the Index
Compile the embeddings into a FAISS index for high-speed retrieval:
```bash
python search/indexer.py
```

### 4. Launch the Web App
Start the FastAPI server, which also serves the frontend UI:
```bash
uvicorn api.main:app --reload
```

Once the server has started and initialized the ML models, open your web browser and navigate to `http://127.0.0.1:8000/` to access the Semantic Search Interface. You can view the raw API documentation at `http://127.0.0.1:8000/docs`.

## Key Features

- **Two-Stage Retrieval**: Balances speed and accuracy by using FAISS for broad candidate retrieval and a Cross-Encoder for precise reranking.
- **Dynamic Recommendations**: Leverages the existing vector space to instantly find contextually similar items.
- **Premium Interface**: A modern, responsive web application featuring dark mode, glassmorphism, and seamless API integration.
- **Memory Efficient**: Shares model instances and FAISS indices across both the Search and Recommendation modules to optimize hardware utilization.
