import numpy as np

from src.models.evaluate import evaluate
from src import config


def test_evaluate_perfect_predictions_pass_gate():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.01, 0.02, 0.99, 0.95, 0.03, 0.88, 0.92, 0.05])
    result = evaluate(y_true, y_pred_proba)
    assert result.roc_auc == 1.0
    assert result.passed_gate is True


def test_evaluate_random_predictions_fail_gate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred_proba = rng.uniform(size=200)  # pure noise -> ~0.5 AUC
    result = evaluate(y_true, y_pred_proba)
    assert result.roc_auc < config.MIN_ROC_AUC
    assert result.passed_gate is False


def test_evaluate_confusion_matrix_counts_sum_to_total():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.3, 0.4, 0.9])
    result = evaluate(y_true, y_pred_proba)
    assert result.tn + result.fp + result.fn + result.tp == len(y_true)
