from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import joblib
from pathlib import Path
import os

app = FastAPI()

def require_api_key(x_api_key: str = Header(default=None, alias="x-api-key")):
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# --- Model loading ---
MODEL_PATH = Path(__file__).resolve().parents[1] / "tone_model" / "tone_model.pkl"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# --- Model version (from env) ---
MODEL_VERSION = os.getenv("TONE_MODEL_V", "unknown")

# --- Request schema ---
class PredictRequest(BaseModel):
    text: str

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok", "tone_model_v": MODEL_VERSION}

# --- Prediction ---
@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    prediction = model.predict([req.text])[0]
    return {
        "tone": prediction,
        "tone_model_v": MODEL_VERSION
    }
