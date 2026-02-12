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
    # Clean quotes just in case Railway adds them
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
    eps: float = 0.85
    min_samples: int = 3

@app.get("/health")
def health():
    return {"status": "ok", "dbscan_v": MODEL_VERSION}

@app.post("/cluster", dependencies=[Depends(require_api_key)])
def cluster(req: ClusterRequest):
    if not req.items:
        return {"dbscan_v": MODEL_VERSION, "results": []}

    # Parse embeddings safely
    embeddings = []
    valid_items = []
    for item in req.items:
        try:
            emb = item.embedding if isinstance(item.embedding, list) else []
            if len(emb) > 0:
                embeddings.append(emb)
                valid_items.append(item)
        except:
            continue  # skip bad embedding

    if not embeddings:
        return {"dbscan_v": MODEL_VERSION, "results": []}

    X = np.array(embeddings)
    dbscan = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    labels = dbscan.fit_predict(X)

    # Group valid items by cluster labels
    temp_clusters = {}
    for item, lbl in zip(valid_items, labels):
        if lbl == -1:
            continue
        if lbl not in temp_clusters:
            temp_clusters[lbl] = []
        temp_clusters[lbl].append(item)

    # Diversity logic (kept as original)
    final_labels = [-1] * len(req.items)  # default all noise
    valid_cluster_sizes = {}

    for lbl, items in temp_clusters.items():
        unique_sources = {i.source for i in items if i.source}
        
        if len(unique_sources) <= 1:
            # Discard mono-source clusters
            pass
        else:
            valid_cluster_sizes[lbl] = len(items)
            # Map back to original indices (this is approximate; assumes order preserved)
            for orig_idx, orig_item in enumerate(req.items):
                if orig_item.id == items[0].id:  # rough match - improve if needed
                    for i, it in enumerate(valid_items):
                        if it.id == orig_item.id:
                            final_labels[orig_idx] = lbl
                            break

    results = []
    for idx, item in enumerate(req.items):
        lbl = final_labels[idx]
        results.append({
            "id": item.id,
            "event_cluster_id": int(lbl) if lbl != -1 else -1,
            "cluster_size": valid_cluster_sizes.get(lbl, 0)
        })

    return {
        "dbscan_v": MODEL_VERSION,
        "results": results
    }