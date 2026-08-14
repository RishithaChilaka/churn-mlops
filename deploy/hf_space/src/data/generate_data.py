"""
Synthetic telecom customer-churn data generator.

Produces three datasets so the whole MLOps lifecycle can be demoed without
needing a real data source:

  * data/reference/train.csv   - the "current production" training data
  * data/reference/test.csv    - held-out test split (same distribution)
  * data/current/incoming.csv  - a batch of "new" production traffic that has
                                  intentional data drift injected into it, used
                                  to demo Evidently drift detection + the
                                  automated retraining trigger.

Run:
    python -m src.data.generate_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
ROOT = Path(__file__).resolve().parents[2]


def _base_population(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tenure_months = rng.gamma(shape=2.0, scale=18, size=n).clip(0, 72).round().astype(int)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(
        ["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.20]
    )
    monthly_charges = np.clip(
        rng.normal(70, 30, size=n) + np.where(internet == "Fiber optic", 25, 0), 18, 200
    )
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.85, 1.05, size=n)).round(2)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "senior_citizen": rng.choice([0, 1], size=n, p=[0.84, 0.16]),
            "partner": rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52]),
            "dependents": rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
            "tenure_months": tenure_months,
            "phone_service": rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10]),
            "multiple_lines": rng.choice(["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10]),
            "internet_service": internet,
            "online_security": rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.30, 0.50, 0.20]),
            "tech_support": rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.51, 0.20]),
            "streaming_tv": rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.38, 0.42, 0.20]),
            "streaming_movies": rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.38, 0.42, 0.20]),
            "contract": contract,
            "paperless_billing": rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41]),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                size=n, p=[0.34, 0.23, 0.22, 0.21],
            ),
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges,
            "num_support_tickets": rng.poisson(1.2, size=n),
            "avg_monthly_usage_gb": np.clip(rng.normal(180, 90, size=n), 0, None).round(1),
        }
    )
    return df


def _assign_churn(df: pd.DataFrame, rng: np.random.Generator, drift: bool = False) -> pd.Series:
    """Logistic-style churn probability driven by realistic risk factors."""
    logit = (
        -1.4
        + 1.35 * (df["contract"] == "Month-to-month")
        + 0.55 * (df["internet_service"] == "Fiber optic")
        - 0.9 * (df["contract"] == "Two year")
        - 0.02 * df["tenure_months"]
        + 0.012 * (df["monthly_charges"] - 70)
        + 0.30 * (df["online_security"] == "No")
        + 0.28 * (df["tech_support"] == "No")
        + 0.20 * df["num_support_tickets"].clip(upper=6)
        + 0.15 * (df["paperless_billing"] == "Yes")
        + 0.10 * (df["payment_method"] == "Electronic check")
        - 0.20 * (df["partner"] == "Yes")
    )
    if drift:
        # Simulate a real-world shift: support-ticket volume up, price sensitivity up,
        # and fiber customers churning faster (e.g. a competitor promo hit the market).
        logit = logit + 0.55 * (df["internet_service"] == "Fiber optic") + 0.35 * df["num_support_tickets"].clip(upper=6)

    prob = 1 / (1 + np.exp(-logit))
    return (rng.uniform(size=len(df)) < prob).astype(int)


def generate(n_train: int = 6000, n_test: int = 1500, n_current: int = 2000) -> None:
    rng = np.random.default_rng(RNG_SEED)

    train_df = _base_population(n_train, rng)
    train_df["churn"] = _assign_churn(train_df, rng)

    test_df = _base_population(n_test, rng)
    test_df["churn"] = _assign_churn(test_df, rng)

    # "Current" incoming production batch, several weeks later, with drift injected
    # (higher support tickets, more fiber adoption, higher prices) to demo monitoring.
    current_df = _base_population(n_current, rng)
    current_df["internet_service"] = rng.choice(
        ["Fiber optic", "DSL", "No"], size=n_current, p=[0.62, 0.28, 0.10]
    )
    current_df["num_support_tickets"] = rng.poisson(2.1, size=n_current)
    current_df["monthly_charges"] = np.clip(
        current_df["monthly_charges"] * rng.uniform(1.08, 1.22, size=n_current), 18, 240
    ).round(2)
    current_df["churn"] = _assign_churn(current_df, rng, drift=True)

    ref_dir = ROOT / "data" / "reference"
    cur_dir = ROOT / "data" / "current"
    ref_dir.mkdir(parents=True, exist_ok=True)
    cur_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(ref_dir / "train.csv", index=False)
    test_df.to_csv(ref_dir / "test.csv", index=False)
    current_df.to_csv(cur_dir / "incoming.csv", index=False)

    print(f"Wrote {len(train_df)} rows -> {ref_dir/'train.csv'}")
    print(f"Wrote {len(test_df)} rows -> {ref_dir/'test.csv'}")
    print(f"Wrote {len(current_df)} rows -> {cur_dir/'incoming.csv'} (drifted, for monitoring demo)")
    print(f"Train churn rate: {train_df['churn'].mean():.3f}")
    print(f"Current churn rate: {current_df['churn'].mean():.3f}")


if __name__ == "__main__":
    generate()
