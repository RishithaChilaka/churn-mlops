"""
FastAPI serving layer for the churn model.

Endpoints:
  GET  /health           liveness + model status
  GET  /model-info        currently loaded model metadata + test metrics
  POST /predict            single customer prediction
  POST /predict/batch      batch predictions
  GET  /metrics             prediction log summary (feeds monitoring/drift job)

Every prediction is appended to `logs/prediction_log.csv` (customer_id, features,
prediction, probability, timestamp) so the monitoring pipeline can compare
against the reference distribution and Evidently can compute drift/performance.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.api.model_loader import LoadedModel, load_model
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.features.build_features import split_X_y

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

LOG_DIR = config.ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_PATH = LOG_DIR / "prediction_log.csv"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style serving API for the XGBoost churn model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = config.ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

_state: dict[str, LoadedModel] = {}


@app.on_event("startup")
def _startup() -> None:
    _state["model"] = load_model()
    logger.info("Model loaded: source=%s version=%s", _state["model"].source, _state["model"].version)


def _risk_tier(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def _predict_df(df: pd.DataFrame) -> list[PredictionResponse]:
    model: LoadedModel = _state["model"]
    X, _ = split_X_y(df)
    probs = model.pipeline.predict_proba(X)[:, 1]
    results = []
    for cid, p in zip(df["customer_id"], probs):
        results.append(
            PredictionResponse(
                customer_id=cid,
                churn_probability=round(float(p), 4),
                churn_prediction=int(p >= 0.5),
                risk_tier=_risk_tier(p),
                model_version=model.version,
            )
        )
    _log_predictions(df, probs)
    return results


def _log_predictions(df: pd.DataFrame, probs) -> None:
    write_header = not PREDICTION_LOG_PATH.exists()
    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(list(df.columns) + ["churn_probability", "predicted_at"])
        now = datetime.now(timezone.utc).isoformat()
        for (_, row), p in zip(df.iterrows(), probs):
            writer.writerow(list(row.values) + [round(float(p), 4), now])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = _state.get("model")
    return HealthResponse(
        status="ok" if model else "degraded",
        model_loaded=model is not None,
        model_source=model.source if model else "none",
        model_version=model.version if model else "none",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = _state.get("model")
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=config.MODEL_REGISTRY_NAME,
        model_version=model.version,
        model_source=model.source,
        metrics=model.metrics,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([customer.model_dump()])
    return _predict_df(df)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([c.model_dump() for c in request.customers])
    return BatchPredictionResponse(predictions=_predict_df(df))


@app.get("/metrics")
def metrics() -> dict:
    if not PREDICTION_LOG_PATH.exists():
        return {"total_predictions": 0}
    df = pd.read_csv(PREDICTION_LOG_PATH)
    return {
        "total_predictions": len(df),
        "mean_churn_probability": round(float(df["churn_probability"].mean()), 4),
        "high_risk_count": int((df["churn_probability"] >= 0.7).sum()),
        "log_path": str(PREDICTION_LOG_PATH),
    }
