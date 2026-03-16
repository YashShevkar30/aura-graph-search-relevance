"""
FastAPI Search Interface
=========================
RESTful API for graph-powered search and node similarity queries.
"""
import time
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

app = FastAPI(title="Aura Graph Search Engine", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    query_node: int = None
    top_k: int = 10


class SearchResult(BaseModel):
    results: list[dict]
    latency_ms: float


@app.post("/api/v1/search", response_model=SearchResult)
async def search(request: SearchRequest):
    # In production, ranker would be loaded at startup
    start = time.time()
    # Placeholder: return empty until model is loaded
    latency = (time.time() - start) * 1000
    return SearchResult(results=[], latency_ms=round(latency, 2))


@app.get("/api/v1/similar/{node_id}")
async def similar_nodes(node_id: int, top_k: int = Query(10)):
    return {"node_id": node_id, "similar": []}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aura-graph-search"}
