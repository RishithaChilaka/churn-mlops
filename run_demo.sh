#!/usr/bin/env bash
# End-to-end local demo of the full ML lifecycle:
#   data -> feature engineering -> training + MLflow tracking -> model registry
#   -> REST API serving -> live predictions -> drift monitoring -> automated retraining
#
# Run from the project root:  bash run_demo.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
API_PORT=${API_PORT:-8000}
API_PID=""

cleanup() {
  if [[ -n "$API_PID" ]]; then
    echo -e "\n[demo] Stopping API server (pid $API_PID)..."
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

step() { echo -e "\n\033[1;36m==> $1\033[0m"; }

step "1/7  Generating synthetic churn dataset (train/test/current-with-drift)"
$PYTHON -m src.data.generate_data

step "2/7  Training XGBoost model with hyperparameter search + MLflow tracking"
$PYTHON -m src.models.train --n-trials 8 --promote

step "3/7  Inspecting the model registry / local metrics"
cat models/latest_metrics.json

step "4/7  Starting the FastAPI serving layer on :$API_PORT"
uvicorn src.api.main:app --port "$API_PORT" --host 0.0.0.0 &
API_PID=$!
sleep 3
curl -s "http://localhost:$API_PORT/health" | python -m json.tool

step "5/7  Sending a live prediction request"
curl -s -X POST "http://localhost:$API_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "customer_id": "CUST-DEMO-001",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure_months": 3,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 110.0,
        "total_charges": 330.0,
        "num_support_tickets": 4,
        "avg_monthly_usage_gb": 240.0
      }' | python -m json.tool

step "6/7  Running Evidently data-drift report (reference vs. incoming production data)"
$PYTHON -m src.monitoring.drift_report

step "7/7  Running the automated retraining pipeline (triggers because drift was injected)"
$PYTHON -m src.pipelines.retrain_pipeline

echo -e "\n\033[1;32m[demo] Complete. Open reports/drift/*.html for the drift report,"
echo -e "        models/latest_metrics.json for the latest model metrics, and"
echo -e "        http://localhost:$API_PORT/docs for the interactive API docs.\033[0m"
echo -e "[demo] Press Ctrl+C to stop the API server."
wait "$API_PID"
