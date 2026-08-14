"""
API tests. Trains a tiny local model first (fast, no MLflow server needed)
so the FastAPI app can start with model_loader's local-artifact fallback.
"""
import pytest
from fastapi.testclient import TestClient

from src.models.train import train


SAMPLE_CUSTOMER = {
    "customer_id": "CUST-TEST-001",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 5,
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
    "monthly_charges": 105.5,
    "total_charges": 527.5,
    "num_support_tickets": 3,
    "avg_monthly_usage_gb": 210.0,
}


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_trained():
    train(n_trials=2)


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_tier"] in ("low", "medium", "high")


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert "metrics" in resp.json()


def test_explain_degrades_gracefully_without_api_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/explain", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_configured"] is False
    assert "explanation" in body and len(body["explanation"]) > 0
    # still returns the real prediction alongside the fallback message
    assert 0.0 <= body["churn_probability"] <= 1.0
