"""
models/flow_baseline.py

MachineLearning / Flow-based intrusion detection baseline (Random Forest)
trained on statistical flow features (packet counts, durations, IATs, flags)
from CICIDS2017 flow-level CSVs.

Provides:
- Robust feature cleaning (handling inf, nan, leading whitespace in column names).
- Fast training with class weighting and multi-core parallelism.
- Feature importance extraction.
- Pickling/unpickling model weights.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Columns to ignore as predictive features (identifiers, timestamps, labels)
EXCLUDE_COLS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "label",
    "is_attack",
    "__source_file",
}


def clean_flow_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Strip leading/trailing whitespace from column names, drop non-feature columns,
    and replace inf / -inf / nan with valid finite values.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    labels = None
    if "Label" in df.columns:
        labels = df["Label"].astype(str).str.strip()

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].copy()

    # Convert all columns to numeric, replacing errors with NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace inf and -inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs with 0.0 or column median
    X.fillna(0.0, inplace=True)

    return X, labels


class FlowRandomForestBaseline:
    """Random Forest classifier on flow-level metadata."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced_subsample",
        )
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the random forest on cleaned features."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary anomaly class (0 = Benign, 1 = Attack)."""
        # Ensure column alignment
        X_aligned = X[self.feature_names].values
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict attack probability."""
        X_aligned = X[self.feature_names].values
        probs = self.model.predict_proba(X_aligned)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            # Only single class seen during fit
            return np.zeros(len(X), dtype=np.float32)
        return probs[:, -1]

    def get_feature_importances(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return top N most important flow features."""
        if not self.feature_names or not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        return [(self.feature_names[i], float(importances[i])) for i in sorted_idx[:top_n]]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FlowRandomForestBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)
