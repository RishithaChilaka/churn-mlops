"""Central configuration for the churn MLOps project."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- MLflow ---
# Use `or` (not dict.get's default) so an unset/empty-string CI secret falls
# back to the local file store instead of passing "" to MLflow.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"file://{ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME") or "churn-prediction"
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME") or "churn-xgboost"

# --- Data ---
TRAIN_PATH = ROOT / "data" / "reference" / "train.csv"
TEST_PATH = ROOT / "data" / "reference" / "test.csv"
CURRENT_PATH = ROOT / "data" / "current" / "incoming.csv"

# --- Model quality gate (CI + retraining pipeline enforce this) ---
MIN_ROC_AUC = float(os.environ.get("MIN_ROC_AUC", 0.75))
MIN_F1 = float(os.environ.get("MIN_F1", 0.55))

# --- Drift monitoring ---
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", 0.3))  # fraction of drifted columns that triggers retraining

# --- Local model artifact fallback (used by API if MLflow registry unavailable) ---
LOCAL_MODEL_DIR = ROOT / "models"
LOCAL_MODEL_PATH = LOCAL_MODEL_DIR / "latest_model.joblib"
LOCAL_METRICS_PATH = LOCAL_MODEL_DIR / "latest_metrics.json"
