from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., examples=["CUST-000123"])
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure_months: int = Field(..., ge=0, le=100)
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_tickets: int = Field(..., ge=0)
    avg_monthly_usage_gb: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_source: str
    metrics: dict


class ExplainResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    model_version: str
    explanation: str
    llm_configured: bool
