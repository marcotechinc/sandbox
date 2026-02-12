from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
import os
from collections import Counter

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
    eps: float = 0.85
    min_samples: int = 3

@app.get("/health")
def health():
    return {"status": "ok", "dbscan_v": MODEL_VERSION}

@app.post("/cluster", dependencies=[Depends(require_api_key)])
def cluster(req: ClusterRequest):
    if not req.items:
        return {"dbscan_v": MODEL_VERSION, "results": []}

    # Safe embedding parsing
    embeddings = []
    valid_items = []
    for item in req.items:
        try:
            emb = item.embedding if isinstance(item.embedding, list) else []
            if len(emb) > 0:
                embeddings.append(emb)
                valid_items.append(item)
        except:
            continue

    if not embeddings:
        return {"dbscan_v": MODEL_VERSION, "results": []}

    X = np.array(embeddings)
    dbscan = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    labels = dbscan.fit_predict(X)

    # Group items by cluster label
    temp_clusters = {}
    for item, lbl in zip(valid_items, labels):
        if lbl == -1:
            continue
        temp_clusters.setdefault(lbl, []).append(item)

    # === STRONG FILTERING ===
    valid_cluster_ids = set()

    for lbl, items in temp_clusters.items():
        sources = [i.source.strip().lower() for i in items if i.source and i.source.strip()]
        unique_sources = set(sources)
        source_counts = Counter(sources)

        # Rule 1: Must have at least 2 different sources
        if len(unique_sources) < 2:
            continue

        # Rule 2: No single source dominates (>50% of cluster)
        max_source_share = max(source_counts.values()) / len(items)
        if max_source_share > 0.5:
            continue

        # If both rules passed → keep this cluster
        valid_cluster_ids.add(lbl)

    # Build final results
    results = []
    for orig_item in req.items:
        cluster_lbl = -1
        cluster_size = 0

        for lbl, items in temp_clusters.items():
            if any(it.id == orig_item.id for it in items):
                if lbl in valid_cluster_ids:
                    cluster_lbl = lbl
                    cluster_size = len(items)
                break

        results.append({
            "id": orig_item.id,
            "event_cluster_id": int(cluster_lbl),
            "cluster_size": cluster_size
        })

    return {
        "dbscan_v": MODEL_VERSION,
        "results": results
    }