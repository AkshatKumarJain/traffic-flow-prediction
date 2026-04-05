from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURE_COUNT = 13
MODEL_PATH = Path(__file__).with_name("traffic_model.pkl")


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)


app = FastAPI()
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "ML Service Running"}


@app.get("/health")
def health():
    return {"success": True}


@app.post("/predict")
def predict(payload: PredictionRequest):
    features = np.array(payload.features, dtype=float)

    if not np.isfinite(features).all():
        raise HTTPException(status_code=400, detail="All feature values must be finite numbers")

    try:
        prediction = model.predict(features.reshape(1, FEATURE_COUNT))[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    return {"prediction": float(prediction)}
