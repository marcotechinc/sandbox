from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
import os

app = FastAPI()

def require_api_key(x_api_key: str = Header(default=None, alias="x-api-key")):
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    clean_expected = expected.strip('"').strip("'")
    if x_api_key != clean_expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

MODEL_VERSION = os.getenv("DBSCAN_V", "v1")

class Item(BaseModel):
    id: str
    embedding: List[float]
    source: Optional[str] = "unknown"

class ClusterRequest(BaseModel):
    items: List[Item]
    eps: float = 1.0               # ← increased default
    min_samples: int = 2           # ← lowered default (allows pairs)
    require_multi_source: bool = False  # ← new: make diversity optional

@app.get("/health")
def health():
    return {"status": "ok", "dbscan_v": MODEL_VERSION}

@app.post("/cluster", dependencies=[Depends(require_api_key)])
def cluster(req: ClusterRequest):
    if not req.items:
        return {"dbscan_v": MODEL_VERSION, "results": []}

    X = np.array([i.embedding for i in req.items])
    dbscan = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    labels = dbscan.fit_predict(X)

    # Group items by temporary cluster labels
    temp_clusters = {}
    for item, lbl in zip(req.items, labels):
        if lbl == -1:
            continue
        if lbl not in temp_clusters:
            temp_clusters[lbl] = []
        temp_clusters[lbl].append(item)

    # Diversity logic (now optional)
    final_labels = list(labels)
    valid_cluster_sizes = {}

    if req.require_multi_source:
        for lbl, items in temp_clusters.items():
            unique_sources = {i.source for i in items if i.source}
            if len(unique_sources) <= 1:
                # Discard mono-source clusters
                for idx, original_lbl in enumerate(labels):
                    if original_lbl == lbl:
                        final_labels[idx] = -1
            else:
                valid_cluster_sizes[lbl] = len(items)
    else:
        # Keep all clusters (even single-source)
        for lbl, items in temp_clusters.items():
            valid_cluster_sizes[lbl] = len(items)

    # Build results
    results = []
    for item, lbl in zip(req.items, final_labels):
        results.append({
            "id": item.id,
            "event_cluster_id": int(lbl) if lbl != -1 else -1,
            "cluster_size": valid_cluster_sizes.get(lbl, 0)
        })

    return {
        "dbscan_v": MODEL_VERSION,
        "results": results,
        "params_used": {
            "eps": req.eps,
            "min_samples": req.min_samples,
            "require_multi_source": req.require_multi_source
        }
    }