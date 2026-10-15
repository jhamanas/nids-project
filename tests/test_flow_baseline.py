"""
tests/test_flow_baseline.py

Unit tests for models/flow_baseline.py:
- DataFrame cleaning (whitespace stripping, column filtering, inf/nan replacement).
- Random Forest model training, prediction, and probability outputs.
- Model serialization (save and load).
- Feature importance extraction.
"""

import numpy as np
import pandas as pd
import pytest

from models.flow_baseline import FlowRandomForestBaseline, clean_flow_dataframe


def test_clean_flow_dataframe():
    data = {
        " Flow ID ": ["1-2-3-4", "5-6-7-8"],
        " Destination Port ": [80, 443],
        " Flow Duration ": [100.0, np.inf],
        " Total Fwd Packets ": [5, np.nan],
        " Label ": ["BENIGN", "DDoS"],
    }
    df = pd.DataFrame(data)

    X, labels = clean_flow_dataframe(df)

    assert "Flow ID" not in X.columns
    assert "Label" not in X.columns
    assert "Destination Port" in X.columns
    assert "Flow Duration" in X.columns
    assert np.isfinite(X.values).all()
    assert list(labels) == ["BENIGN", "DDoS"]


def test_flow_random_forest_fit_predict(tmp_path):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Flow Duration": np.random.uniform(10, 1000, n),
        "Total Fwd Packets": np.random.randint(1, 50, n),
        "Flow Bytes/s": np.random.exponential(100, n),
    })
    # Target: 1 if Fwd Packets > 25, else 0
    y = (df["Total Fwd Packets"] > 25).astype(int).values

    rf = FlowRandomForestBaseline(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(df, y)

    preds = rf.predict(df)
    probs = rf.predict_proba(df)

    assert len(preds) == n
    assert len(probs) == n
    assert (preds == y).mean() > 0.90
    assert 0.0 <= probs.min() and probs.max() <= 1.0

    # Top features
    importances = rf.get_feature_importances(top_n=3)
    assert len(importances) == 3
    assert importances[0][0] == "Total Fwd Packets"

    # Save and reload
    save_path = tmp_path / "rf_test.pkl"
    rf.save(save_path)
    loaded = FlowRandomForestBaseline.load(save_path)
    loaded_preds = loaded.predict(df)
    assert np.array_equal(preds, loaded_preds)
