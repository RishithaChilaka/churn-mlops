"""
Automated model evaluation used by:
  * training (to report metrics)
  * CI (as a quality gate before merge)
  * the retraining pipeline (to decide whether to promote a new model)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


@dataclass
class EvalResult:
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int
    passed_gate: bool
    gate_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(y_true, y_pred_proba, threshold: float = 0.5) -> EvalResult:
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    f1 = float(f1_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    passed = roc_auc >= config.MIN_ROC_AUC and f1 >= config.MIN_F1
    if passed:
        reason = f"PASS: roc_auc={roc_auc:.4f} >= {config.MIN_ROC_AUC}, f1={f1:.4f} >= {config.MIN_F1}"
    else:
        reason = (
            f"FAIL: roc_auc={roc_auc:.4f} (min {config.MIN_ROC_AUC}), "
            f"f1={f1:.4f} (min {config.MIN_F1})"
        )

    return EvalResult(
        roc_auc=roc_auc,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        passed_gate=passed,
        gate_reason=reason,
    )
