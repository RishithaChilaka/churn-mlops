"""
Feature engineering + preprocessing pipeline for the churn model.

Exposes `build_preprocessor()` which returns an sklearn ColumnTransformer,
and `engineer_features()` which adds derived columns before the
ColumnTransformer runs. Kept as a single sklearn Pipeline so the exact same
transform is used at train time and inference time (no train/serve skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "customer_id"
TARGET_COL = "churn"

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_bucket",
]

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_monthly_usage_gb",
    "charges_per_tenure",
    "support_tickets_per_tenure",
    "is_fiber_high_price",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived / engineered columns. Pure function, no fitting required."""
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"],
    ).astype(str)

    safe_tenure = df["tenure_months"].replace(0, 1)
    df["charges_per_tenure"] = (df["total_charges"] / safe_tenure).round(2)
    df["support_tickets_per_tenure"] = (df["num_support_tickets"] / safe_tenure).round(4)

    df["is_fiber_high_price"] = (
        (df["internet_service"] == "Fiber optic") & (df["monthly_charges"] > 90)
    ).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    df = engineer_features(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y
