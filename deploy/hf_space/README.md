---
title: Churn Prediction Demo
emoji: 📉
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Customer Churn Prediction — Live Demo

Live FastAPI service serving an XGBoost churn model (trained + tracked with
MLflow, evaluated against an automated quality gate) as part of a full
production-style MLOps pipeline.

- Interactive demo UI: this page
- API docs: `/docs`
- Model metadata: `/model-info`
- Health check: `/health`

Full source (training, feature engineering, MLflow tracking/registry,
Evidently drift monitoring, automated retraining, Docker, Terraform for AWS,
GitHub Actions CI/CD) lives in the parent project this Space was built from.
