"""
Data drift + model performance monitoring using Evidently.

Compares the reference training distribution against a "current" production
batch (data/current/incoming.csv, or the live prediction log if present) and:

  * writes an HTML drift report to reports/drift/ for humans to view
  * writes a JSON summary
  * returns a simple bool `retraining_recommended` based on the share of
    drifted columns, which the retraining pipeline uses as its trigger

Run:
    python -m src.monitoring.drift_report
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

REPORT_DIR = config.ROOT / "reports" / "drift"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    return df[cols]


def run_drift_report(reference_path=None, current_path=None) -> dict:
    reference_path = reference_path or config.TRAIN_PATH
    current_path = current_path or config.CURRENT_PATH

    ref_df = _prepare(pd.read_csv(reference_path))
    cur_df = _prepare(pd.read_csv(current_path))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)

        html_path = REPORT_DIR / f"drift_report_{timestamp}.html"
        report.save_html(str(html_path))

        result = report.as_dict()
        drift_metric = next(
            m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
        )
        n_drifted = drift_metric["result"]["number_of_drifted_columns"]
        n_columns = drift_metric["result"]["number_of_columns"]
        dataset_drift = drift_metric["result"]["dataset_drift"]
        share_drifted = n_drifted / n_columns if n_columns else 0.0

    except Exception as e:  # noqa: BLE001
        # Fallback: lightweight statistical drift check (PSI-ish) so the demo
        # still works even if the evidently version/API differs.
        html_path = None
        n_drifted, n_columns, share_drifted, dataset_drift = _fallback_drift(ref_df, cur_df)
        result = {"fallback_reason": str(e)}

    retraining_recommended = share_drifted >= config.DRIFT_SHARE_THRESHOLD

    summary = {
        "timestamp": timestamp,
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "n_drifted_columns": n_drifted,
        "n_columns": n_columns,
        "share_drifted_columns": round(share_drifted, 4),
        "dataset_drift_detected": bool(dataset_drift),
        "drift_threshold": config.DRIFT_SHARE_THRESHOLD,
        "retraining_recommended": retraining_recommended,
        "html_report": str(html_path) if html_path else None,
    }

    summary_path = REPORT_DIR / f"drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    # also write a stable "latest" pointer for the demo/UI to read
    with open(REPORT_DIR / "latest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def _fallback_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> tuple[int, int, float, bool]:
    """Simple mean-shift / distribution-shift heuristic per column, used only
    if the evidently import/API fails (keeps the demo resilient to version
    differences)."""
    drifted = 0
    total = 0
    for col in ref_df.columns:
        total += 1
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            ref_mean, ref_std = ref_df[col].mean(), ref_df[col].std() or 1e-6
            cur_mean = cur_df[col].mean()
            z = abs(cur_mean - ref_mean) / ref_std
            if z > 0.5:
                drifted += 1
        else:
            ref_dist = ref_df[col].value_counts(normalize=True)
            cur_dist = cur_df[col].value_counts(normalize=True)
            tvd = 0.5 * sum(
                abs(ref_dist.get(k, 0) - cur_dist.get(k, 0))
                for k in set(ref_dist.index) | set(cur_dist.index)
            )
            if tvd > 0.15:
                drifted += 1
    share = drifted / total if total else 0.0
    return drifted, total, share, share >= config.DRIFT_SHARE_THRESHOLD


if __name__ == "__main__":
    run_drift_report()
