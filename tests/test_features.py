import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features, split_X_y


def _sample_row(**overrides):
    row = {
        "customer_id": "CUST-000001",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
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
        "monthly_charges": 95.0,
        "total_charges": 1140.0,
        "num_support_tickets": 2,
        "avg_monthly_usage_gb": 200.0,
    }
    row.update(overrides)
    return row


def test_engineer_features_adds_expected_columns():
    df = pd.DataFrame([_sample_row()])
    out = engineer_features(df)
    assert "tenure_bucket" in out.columns
    assert "charges_per_tenure" in out.columns
    assert "support_tickets_per_tenure" in out.columns
    assert "is_fiber_high_price" in out.columns
    assert out.loc[0, "is_fiber_high_price"] == 1


def test_engineer_features_handles_zero_tenure_without_division_error():
    df = pd.DataFrame([_sample_row(tenure_months=0, total_charges=0.0)])
    out = engineer_features(df)
    assert out.loc[0, "charges_per_tenure"] == 0.0


def test_split_X_y_with_target():
    df = pd.DataFrame([_sample_row()])
    df["churn"] = 1
    X, y = split_X_y(df)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(X.columns)
    assert y.iloc[0] == 1


def test_split_X_y_without_target():
    df = pd.DataFrame([_sample_row()])
    X, y = split_X_y(df)
    assert y is None
    assert "churn" not in X.columns
