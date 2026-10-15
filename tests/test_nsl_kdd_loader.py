"""
tests/test_nsl_kdd_loader.py

Unit tests for preprocessing/nsl_kdd_loader.py:
- Loading raw NSL-KDD lines with correct column mapping.
- Attack category classification.
- Joint one-hot feature encoding and standard scaling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from preprocessing.nsl_kdd_columns import NSL_KDD_COLUMNS
from preprocessing.nsl_kdd_loader import load_nsl_kdd_raw, prepare_nsl_kdd_datasets


def _create_sample_kdd_file(path: Path, rows: list):
    df = pd.DataFrame(rows, columns=NSL_KDD_COLUMNS)
    df.to_csv(path, header=False, index=False)
    return path


@pytest.fixture
def sample_kdd_data(tmp_path):
    base_row = [
        0, "tcp", "http", "SF", 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0, 0.0, 0.11, 0.0,
        0.0, 0.0, 0.0, 0.0, "normal", 21
    ]
    attack_row = list(base_row)
    attack_row[41] = "neptune"
    attack_row[42] = 19

    train_path = _create_sample_kdd_file(tmp_path / "train.txt", [base_row, attack_row])
    test_path = _create_sample_kdd_file(tmp_path / "test.txt", [base_row, attack_row])
    return train_path, test_path


def test_load_nsl_kdd_raw(sample_kdd_data):
    train_path, _ = sample_kdd_data
    df = load_nsl_kdd_raw(train_path)

    assert len(df) == 2
    assert "category" in df.columns
    assert "is_attack" in df.columns
    assert list(df["category"]) == ["Normal", "DoS"]
    assert list(df["is_attack"]) == [0, 1]


def test_prepare_nsl_kdd_datasets(sample_kdd_data):
    train_path, test_path = sample_kdd_data
    X_train, y_train, X_test, y_test, feat_names, train_cats, test_cats = prepare_nsl_kdd_datasets(
        train_path, test_path
    )

    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert list(y_train) == [0, 1]
    assert list(y_test) == [0, 1]
