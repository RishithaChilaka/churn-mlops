# Customer Churn Prediction — Production-Grade MLOps Project

An end-to-end, demo-ready ML system covering the full lifecycle: data →
feature engineering → training → experiment tracking → model registry →
automated evaluation → REST API deployment → monitoring → data-drift
detection → automated retraining.

Stack: **Python, XGBoost, MLflow, FastAPI, Docker, AWS (ECS/ECR/S3 via
Terraform), GitHub Actions, Evidently.**

## Architecture

```mermaid
flowchart LR
    subgraph Data
        A[Synthetic churn data generator] --> B[Reference train/test]
        A --> C[Current / incoming batch\n(drift injected)]
    end

    subgraph Training["Training Pipeline (src/models/train.py)"]
        B --> D[Feature engineering\n+ preprocessing pipeline]
        D --> E[XGBoost + hyperparameter search]
        E --> F[MLflow Tracking\n(params, metrics, artifacts)]
        F --> G[Automated evaluation gate\n(ROC-AUC, F1 thresholds)]
        G -->|pass| H[MLflow Model Registry\nStaging → Production]
    end

    subgraph Serving["Serving (src/api)"]
        H --> I[FastAPI /predict]
        I --> J[Prediction log]
    end

    subgraph Monitoring["Monitoring (src/monitoring)"]
        C --> K[Evidently drift report]
        J -.optional.-> K
        K -->|drift ≥ threshold| L[Retraining trigger]
    end

    L --> M[Retraining Pipeline\nmerge new data → retrain → evaluate → promote]
    M --> Training

    subgraph CI/CD["GitHub Actions"]
        N[ci.yml: lint, test, quality gate]
        O[cd.yml: build image → push ECR → deploy ECS]
        P[retrain.yml: scheduled drift check + retrain]
    end

    subgraph AWS["AWS (Terraform, infra/)"]
        Q[ECR] --> R[ECS Fargate + ALB]
        S[S3: MLflow artifacts]
    end

    I --> Q
```

## Project layout

```
churn-mlops/
├── src/
│   ├── data/generate_data.py       synthetic dataset generator (reference + drifted "current" batch)
│   ├── features/build_features.py  feature engineering + sklearn preprocessing pipeline
│   ├── models/train.py             XGBoost training, MLflow tracking, model registry + promotion
│   ├── models/evaluate.py          automated evaluation / quality gate (used by CI + retraining)
│   ├── api/main.py                 FastAPI serving app (/predict, /health, /model-info, /metrics)
│   ├── monitoring/drift_report.py  Evidently data-drift report + retraining trigger
│   ├── pipelines/retrain_pipeline.py  orchestrates drift check → retrain → evaluate → promote
│   └── config.py                   central config (paths, thresholds, MLflow URIs)
├── tests/                          pytest unit + API tests
├── docker/                         Dockerfiles for API, training job, MLflow server
├── docker-compose.yml              runs MLflow + API + on-demand retraining job together
├── infra/                          Terraform for AWS ECR/S3/ECS/ALB deployment
├── .github/workflows/
│   ├── ci.yml                      lint, unit tests, model quality gate on every PR
│   ├── cd.yml                      build & push Docker image, deploy to ECS
│   └── retrain.yml                 scheduled (weekly) + on-demand drift check & retraining
└── run_demo.sh                     one-command live demo of the entire lifecycle
```

## Quickstart — live demo (fully local, no AWS account needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_demo.sh
```

This single script:
1. Generates a synthetic telecom churn dataset — a "reference" training set and
   a "current" production batch with **intentional data drift** injected
   (rising support tickets, fiber-optic price increases) to make monitoring
   demoable.
2. Trains an XGBoost model with a randomized hyperparameter search, logging
   every trial to MLflow and registering the best model in the MLflow Model
   Registry, promoting it to `Production` if it clears the quality gate
   (ROC-AUC ≥ 0.75, F1 ≥ 0.55 — configurable in `src/config.py`).
3. Starts the FastAPI service and sends a live `/predict` request.
4. Runs an Evidently drift report comparing reference vs. current data —
   writes an HTML report to `reports/drift/`.
5. Runs the automated retraining pipeline: because drift was injected, it
   merges the new batch into training data, retrains, re-evaluates, and
   promotes the new model if it's at least as good as the current one.

Interactive API docs: `http://localhost:8000/docs`
MLflow UI (if using Docker Compose / a real tracking server): `http://localhost:5000`

## Running with Docker Compose (MLflow server + API together)

```bash
docker compose up --build           # starts mlflow (5000) + api (8000)
python -m src.data.generate_data
docker compose --profile retrain run retrain   # on-demand retraining job
```

## Running tests / CI locally

```bash
pip install -r requirements-dev.txt
ruff check src tests
pytest tests -v
```

## The full ML lifecycle, explained

**Data & feature engineering.** `src/data/generate_data.py` produces a
realistic telecom-churn dataset (tenure, contract type, charges, support
tickets, etc.) with a logistic churn-probability model. `src/features/build_features.py`
adds derived features (tenure buckets, charges-per-tenure, high-price-fiber
flag) and wraps preprocessing (imputation, scaling, one-hot encoding) in a
single sklearn `Pipeline` so training and serving apply *identical* transforms.

**Training & experiment tracking.** `src/models/train.py` runs a randomized
hyperparameter search over XGBoost, logging every trial (params + val
metrics) as a nested MLflow run, then refits the best configuration on the
full training set and evaluates on a held-out test set.

**Model registry & promotion.** The final pipeline is logged as an MLflow
model artifact and registered under `churn-xgboost`. If it passes the
quality gate and beats (or there is no) current `Production` model, it's
promoted automatically; otherwise it's parked in `Staging`.

**Automated evaluation.** `src/models/evaluate.py` computes ROC-AUC, F1,
precision, recall, and a confusion matrix, and enforces a pass/fail gate.
This same gate runs in CI (`ci.yml`) on every PR and in the retraining
pipeline — a model that regresses quality is never promoted.

**Serving.** `src/api/main.py` is a FastAPI app that loads the `Production`
model from the MLflow registry (falling back to a local joblib artifact so
the demo works without a running tracking server), exposes `/predict` and
`/predict/batch`, and logs every prediction for monitoring.

**Monitoring & drift detection.** `src/monitoring/drift_report.py` uses
Evidently to compare the reference training distribution against the
current production batch, producing an HTML report and a drift-share metric.
If the share of drifted columns crosses `DRIFT_SHARE_THRESHOLD` (default
30%), retraining is recommended.

**Automated retraining.** `src/pipelines/retrain_pipeline.py` ties it all
together: check drift → if triggered, merge new data into the training set →
retrain → evaluate against the quality gate → promote if it clears the bar.
`retrain.yml` runs this on a weekly schedule (and on demand) in GitHub Actions.

**CI/CD.** `ci.yml` lints, runs unit/API tests, and enforces the model
quality gate on every PR. `cd.yml` builds the API Docker image, pushes to
ECR, and rolls out a new ECS Fargate deployment. `infra/` has the Terraform
for the underlying AWS resources (ECR, S3, ECS, ALB, IAM).

## Configuration

All thresholds and paths live in `src/config.py` and can be overridden via
environment variables: `MLFLOW_TRACKING_URI`, `MIN_ROC_AUC`, `MIN_F1`,
`DRIFT_SHARE_THRESHOLD`.

## Notes on realism vs. demoability

This project is built to actually run end-to-end on a laptop with no cloud
account (`run_demo.sh`), while including the real production pieces
(Terraform, ECS task definitions, CI/CD deploy workflow) so it demonstrates
what shipping it to AWS looks like. Swap the SQLite-backed MLflow server for
Postgres + S3 and point `MLFLOW_TRACKING_URI` at it to go fully production.
