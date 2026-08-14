"""
Train an XGBoost churn classifier with MLflow experiment tracking and
Model Registry integration.

Usage:
    python -m src.models.train
    python -m src.models.train --n-trials 15 --promote

What it does:
  1. Loads reference train/test data
  2. Builds the feature-engineering + preprocessing pipeline
  3. Runs a small randomized hyperparameter search for XGBoost, logging each
     trial as an MLflow run (nested)
  4. Picks the best trial by ROC-AUC, refits on full train data
  5. Evaluates on held-out test data via the automated evaluation gate
  6. Logs the final pipeline as an MLflow model artifact and registers it in
     the MLflow Model Registry (churn-xgboost), optionally promoting it to
     "Production" if it passes the quality gate and beats the current
     Production model
  7. Also saves a local joblib copy + metrics.json so the FastAPI service can
     serve even without a running MLflow tracking server (useful for the demo)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.features.build_features import build_preprocessor, split_X_y
from src.models.evaluate import evaluate

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}


def _make_pipeline(params: dict) -> Pipeline:
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", clf)])


def _get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
    else:
        exp_id = exp.experiment_id
    return exp_id


def train(n_trials: int = 8, promote: bool = False, train_path=None, test_path=None) -> dict:
    train_path = train_path or config.TRAIN_PATH
    test_path = test_path or config.TEST_PATH

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full, y_train_full = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)

    # inner split for hyperparameter selection (keeps test set untouched)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    exp_id = _get_or_create_experiment()
    sampler = list(ParameterSampler(SEARCH_SPACE, n_iter=n_trials, random_state=42))

    best_score = -np.inf
    best_params = None
    best_run_id = None

    parent_run_name = f"training-run-{int(time.time())}"
    with mlflow.start_run(experiment_id=exp_id, run_name=parent_run_name) as parent_run:
        mlflow.set_tag("stage", "hyperparameter_search")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        for i, params in enumerate(sampler):
            with mlflow.start_run(experiment_id=exp_id, run_name=f"trial-{i}", nested=True):
                mlflow.log_params(params)
                pipe = _make_pipeline(params)
                pipe.fit(X_tr, y_tr)
                val_proba = pipe.predict_proba(X_val)[:, 1]
                result = evaluate(y_val, val_proba)
                mlflow.log_metrics(
                    {
                        "val_roc_auc": result.roc_auc,
                        "val_f1": result.f1,
                        "val_precision": result.precision,
                        "val_recall": result.recall,
                    }
                )
                if result.roc_auc > best_score:
                    best_score = result.roc_auc
                    best_params = params
                    best_run_id = mlflow.active_run().info.run_id

        # Refit best params on the FULL training set, evaluate on held-out test set
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        final_pipe = _make_pipeline(best_params)
        final_pipe.fit(X_train_full, y_train_full)
        test_proba = final_pipe.predict_proba(X_test)[:, 1]
        test_result = evaluate(y_test, test_proba)

        mlflow.log_metrics(
            {
                "test_roc_auc": test_result.roc_auc,
                "test_f1": test_result.f1,
                "test_precision": test_result.precision,
                "test_recall": test_result.recall,
                "test_accuracy": test_result.accuracy,
            }
        )
        mlflow.set_tag("gate_passed", str(test_result.passed_gate))
        mlflow.set_tag("gate_reason", test_result.gate_reason)

        signature = mlflow.models.infer_signature(X_train_full, final_pipe.predict_proba(X_train_full)[:, 1])
        mlflow.sklearn.log_model(
            final_pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name=config.MODEL_REGISTRY_NAME,
        )

        run_id = parent_run.info.run_id

        # Save local copy for the FastAPI fallback loader
        config.LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipe, config.LOCAL_MODEL_PATH)
        with open(config.LOCAL_METRICS_PATH, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "best_params": best_params,
                    **test_result.to_dict(),
                },
                f,
                indent=2,
                default=str,
            )

    print(f"Best validation ROC-AUC during search: {best_score:.4f}")
    print(f"Held-out test evaluation: {test_result.gate_reason}")

    if promote and test_result.passed_gate:
        _maybe_promote_to_production(run_id, test_result.roc_auc)

    return {
        "run_id": run_id,
        "best_params": best_params,
        "test_metrics": test_result.to_dict(),
    }


def _maybe_promote_to_production(run_id: str, new_roc_auc: float) -> None:
    """Promote the newly registered model version to Production if it beats
    (or there is no) current Production model. Uses MLflow's alias/stage API."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{config.MODEL_REGISTRY_NAME}'")
    new_version = None
    for v in versions:
        if v.run_id == run_id:
            new_version = v
            break
    if new_version is None:
        print("Could not locate newly registered model version; skipping promotion.")
        return

    # find current production version, if any
    current_prod = None
    for v in versions:
        if v.current_stage == "Production":
            current_prod = v
            break

    should_promote = True
    if current_prod is not None:
        prod_run = client.get_run(current_prod.run_id)
        prod_auc = prod_run.data.metrics.get("test_roc_auc", 0.0)
        should_promote = new_roc_auc >= prod_auc
        print(f"Current production ROC-AUC: {prod_auc:.4f}; candidate: {new_roc_auc:.4f}")

    if should_promote:
        client.transition_model_version_stage(
            name=config.MODEL_REGISTRY_NAME,
            version=new_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Promoted model version {new_version.version} to Production.")
    else:
        client.transition_model_version_stage(
            name=config.MODEL_REGISTRY_NAME,
            version=new_version.version,
            stage="Staging",
        )
        print(f"New model did not beat production; moved version {new_version.version} to Staging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    train(n_trials=args.n_trials, promote=args.promote)
