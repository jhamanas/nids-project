"""
tests/test_metrics.py

Unit tests for evaluation/metrics.py:
- Confusion matrix and rates (precision, recall, f1, fpr)
- Edge cases: zero division, perfect classification, all-benign
- ROC and PR curves with AUC values
- Non-finite (inf/NaN) score handling
"""

import numpy as np
import pytest

from evaluation.metrics import compute_binary_metrics, compute_roc_pr_curves


def test_compute_binary_metrics_standard():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    # TP = 2 (indices 0, 4)
    # FN = 1 (index 1)
    # FP = 1 (index 3)
    # TN = 2 (indices 2, 5)

    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics["total_samples"] == 6
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["false_negatives"] == 1
    assert pytest.approx(metrics["precision"], 1e-4) == 2 / 3
    assert pytest.approx(metrics["recall"], 1e-4) == 2 / 3
    assert pytest.approx(metrics["false_positive_rate"], 1e-4) == 1 / 3


def test_compute_binary_metrics_zero_division():
    # No positive predictions
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_compute_roc_pr_curves_perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    curves = compute_roc_pr_curves(y_true, scores)
    assert pytest.approx(curves["roc_auc"], 1e-4) == 1.0
    assert pytest.approx(curves["pr_auc"], 1e-4) == 1.0
    assert len(curves["fpr"]) > 0
    assert len(curves["precision_curve"]) > 0


def test_compute_roc_pr_curves_handles_inf_nan():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, np.nan, 0.8, np.inf])

    curves = compute_roc_pr_curves(y_true, scores)
    assert not np.isnan(curves["roc_auc"])
    assert not np.isnan(curves["pr_auc"])
    assert 0.0 <= curves["roc_auc"] <= 1.0
    assert 0.0 <= curves["pr_auc"] <= 1.0
