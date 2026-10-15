"""
evaluation/metrics.py

Functions for evaluating NIDS anomaly detection models:
- Confusion Matrix (TP, FP, TN, FN)
- Precision, Recall, F1-Score, False Positive Rate (FPR)
- ROC Curve & ROC-AUC score
- Precision-Recall Curve & PR-AUC score
"""

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard binary classification metrics from ground truth and predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / max(fp + tn, 1))

    return {
        "total_samples": int(len(y_true)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
    }


def compute_roc_pr_curves(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compute ROC and PR curve coordinate arrays and AUC areas.
    Handles non-finite/NaN scores gracefully.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    # Replace inf or nan with finite maximums
    valid_mask = np.isfinite(scores)
    max_val = np.max(scores[valid_mask]) if np.any(valid_mask) else 1.0
    scores[~valid_mask] = max_val * 2.0

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = float(auc(recall, precision))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision,
        "recall_curve": recall,
        "pr_thresholds": pr_thresholds,
    }
