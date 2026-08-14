"""
Loads the churn model for serving.

Tries, in order:
  1. MLflow Model Registry "Production" stage (real production setup)
  2. Local joblib artifact written by src/models/train.py (works fully offline,
     used for the local demo so nobody needs a running MLflow server)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib

from src import config

logger = logging.getLogger("churn-api")


@dataclass
class LoadedModel:
    pipeline: object
    version: str
    source: str
    metrics: dict


def load_model() -> LoadedModel:
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(config.MODEL_REGISTRY_NAME, stages=["Production"])
        if versions:
            mv = versions[0]
            model_uri = f"models:/{config.MODEL_REGISTRY_NAME}/Production"
            pipeline = mlflow.sklearn.load_model(model_uri)
            run = client.get_run(mv.run_id)
            metrics = dict(run.data.metrics)
            logger.info("Loaded model from MLflow registry: version %s", mv.version)
            return LoadedModel(
                pipeline=pipeline, version=str(mv.version), source="mlflow_registry", metrics=metrics
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Falling back to local model artifact (MLflow registry unavailable: %s)", e)

    if config.LOCAL_MODEL_PATH.exists():
        pipeline = joblib.load(config.LOCAL_MODEL_PATH)
        metrics = {}
        version = "local-unknown"
        if config.LOCAL_METRICS_PATH.exists():
            with open(config.LOCAL_METRICS_PATH) as f:
                metrics = json.load(f)
            version = metrics.get("run_id", "local-unknown")[:8]
        logger.info("Loaded model from local artifact: %s", config.LOCAL_MODEL_PATH)
        return LoadedModel(pipeline=pipeline, version=version, source="local_artifact", metrics=metrics)

    raise RuntimeError(
        "No model available. Run `python -m src.models.train` first to produce a model."
    )
