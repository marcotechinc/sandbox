from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import os

# =========================================================
# App
# =========================================================
app = FastAPI()


# =========================================================
# API key guard
# =========================================================
def require_api_key(x_api_key: str = Header(default=None, alias="x-api-key")):
    """
    Simple API key protection.
    Keeps the OR API private and deterministic.
    """
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY not configured")

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =========================================================
# Versioning
# =========================================================
MODEL_VERSION = os.getenv("OR_V", "v1")


# =========================================================
# Request schemas
# =========================================================
class Item(BaseModel):
    """
    One incident + its measurable features.
    These are INPUTS to optimization, not decisions.
    """
    incident_id: str

    topic_weight: float       # importance of topic domain
    engine_weight: float      # strength of correlation engine
    source_count: int         # number of independent sources
    story_size: int           # size / complexity of narrative


class ORRequest(BaseModel):
    """
    Optimization request.
    """
    items: List[Item]
    max_items: int = 5        # constraint: how many incidents we can surface


# =========================================================
# Health
# =========================================================
@app.get("/health")
def health():
    """
    Simple health check.
    """
    return {
        "status": "ok",
        "or_v": MODEL_VERSION
    }


# =========================================================
# Optimization endpoint
# =========================================================
@app.post("/select", dependencies=[Depends(require_api_key)])
def select_items(req: ORRequest):
    """
    OR v1 — feature-based optimization.

    What this does:
    - combines multiple features into a single priority score
    - applies a hard constraint on how many items can be selected
    - returns the selected incidents (NOT explanations)
    """

    scored_items = []

    for item in req.items:
        # -------------------------------------------------
        # Objective function (explicit + explainable)
        # -------------------------------------------------
        # Plain English:
        # - Topic importance matters most
        # - Strong correlations matter a lot
        # - More sources help, but saturate
        # - Big stories matter, but can't dominate
        priority = (
            item.topic_weight * 0.4 +
            item.engine_weight * 0.3 +
            min(item.source_count, 5) * 0.05 +
            min(item.story_size, 5) * 0.05
        )

        scored_items.append({
            "incident_id": item.incident_id,
            "priority": round(priority, 4),
            "features": {
                "topic_weight": item.topic_weight,
                "engine_weight": item.engine_weight,
                "source_count": item.source_count,
                "story_size": item.story_size
            }
        })

    # -----------------------------------------------------
    # Constraint: limited bandwidth
    # -----------------------------------------------------
    selected = sorted(
        scored_items,
        key=lambda x: x["priority"],
        reverse=True
    )[: req.max_items]

    return {
        "or_v": MODEL_VERSION,
        "selected": selected
    }
