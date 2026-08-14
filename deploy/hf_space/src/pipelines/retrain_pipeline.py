"""
Automated retraining orchestration.

Ties together monitoring -> conditional retraining -> evaluation -> promotion,
the same logic that would be triggered by:
  * a schedule (e.g. nightly / weekly cron via GitHub Actions)
  * a monitoring alert (drift share exceeds threshold)
  * a manual trigger from an ops engineer

Steps:
  1. Run the Evidently drift report comparing reference vs current data
  2. If drift is recommended (or --force is passed), merge the current batch
     into the training set (simulating "new labeled data arrived") and retrain
  3. Evaluate the new model against the quality gate
  4. If it passes and is at least as good as current Production, promote it

Run:
    python -m src.pipelines.retrain_pipeline
    python -m src.pipelines.retrain_pipeline --force
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config
from src.models.train import train
from src.monitoring.drift_report import run_drift_report


def run(force: bool = False, n_trials: int = 6) -> dict:
    print("Step 1/3: Checking for data drift...")
    drift_summary = run_drift_report()

    should_retrain = force or drift_summary["retraining_recommended"]
    result = {"drift_summary": drift_summary, "retrained": False}

    if not should_retrain:
        print("No significant drift detected. Skipping retraining.")
        return result

    print("Step 2/3: Drift detected (or --force) -> merging new data and retraining...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    current_df = pd.read_csv(config.CURRENT_PATH)
    merged_df = pd.concat([train_df, current_df], ignore_index=True)

    merged_path = config.ROOT / "data" / "reference" / "train_merged.csv"
    merged_df.to_csv(merged_path, index=False)

    print(f"Step 3/3: Retraining on {len(merged_df)} rows ({len(current_df)} new)...")
    train_result = train(n_trials=n_trials, promote=True, train_path=merged_path, test_path=config.TEST_PATH)

    result["retrained"] = True
    result["train_result"] = train_result

    summary_path = config.ROOT / "reports" / "drift" / "retrain_result.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("Retraining pipeline complete.")
    print(json.dumps(train_result["test_metrics"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if no drift detected")
    parser.add_argument("--n-trials", type=int, default=6)
    args = parser.parse_args()
    run(force=args.force, n_trials=args.n_trials)
