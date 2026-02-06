from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.cluster import DBSCAN
import os

app = FastAPI()

# --- API key guard (same as tone model) ---
def require_api_key(x_api_key: str = Header(default=None, alias="x-api-key")):
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# --- Versioning (same idea as tone) ---
MODEL_VERSION = os.getenv("DBSCAN_V", "v1")

# --- Request schema ---
class Item(BaseModel):
    id: str
    embedding: List[float]

class ClusterRequest(BaseModel):
    items: List[Item]
    eps: float = 0.7
    min_samples: int = 3

# --- Health ---
@app.get("/health")
def health():
    return {"status": "ok", "dbscan_v": MODEL_VERSION}

# --- Clustering ---
@app.post("/cluster", dependencies=[Depends(require_api_key)])
def cluster(req: ClusterRequest):
    X = np.array([i.embedding for i in req.items])

    dbscan = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    labels = dbscan.fit_predict(X)

    # compute cluster sizes
    cluster_sizes = {}
    for lbl in labels:
        if lbl == -1:
            continue
        cluster_sizes[lbl] = cluster_sizes.get(lbl, 0) + 1

    results = []
    for item, lbl in zip(req.items, labels):
        results.append({
            "id": item.id,
            "event_cluster_id": int(lbl),
            "cluster_size": cluster_sizes.get(lbl, 1 if lbl != -1 else 0)
        })

    return {
        "dbscan_v": MODEL_VERSION,
        "results": results
    }
