import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.engine import SemanticSearchEngine
from recommendation.recommender import ProductRecommender

# Global references
engine = None
recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, recommender
    print("Starting up: Loading ML models and FAISS index (this may take a few seconds)...")
    engine = SemanticSearchEngine()
    recommender = ProductRecommender(engine=engine)
    print("Application startup complete.")
    yield
    print("Shutting down API...")

app = FastAPI(title="Semantic Search Engine API", lifespan=lifespan)

class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[Dict[str, Any]]

class RecommendResponse(BaseModel):
    item_id: int
    top_k: int
    results: List[Dict[str, Any]]

@app.get("/health")
async def health_check():
    if engine is None or recommender is None:
        return {"status": "loading"}
    return {"status": "ok"}

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="The search query text"), 
    top_k: int = Query(5, description="Number of results to return", ge=1, le=50)
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string 'q' cannot be empty")
        
    try:
        results = engine.search(q, top_k=top_k)
        return SearchResponse(query=q, top_k=top_k, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    item_id: int = Query(..., description="The ID (row index) of the target product"),
    top_k: int = Query(5, description="Number of recommendations to return", ge=1, le=50)
):
    try:
        results = recommender.recommend(item_id, top_k=top_k)
        return RecommendResponse(item_id=item_id, top_k=top_k, results=results)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static UI directory
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")
    
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(ui_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
