"""
preprocessing/nsl_kdd_loader.py

Loader and feature encoder for the NSL-KDD dataset.
- Loads KDDTrain+.txt and KDDTest+.txt with correct headers from nsl_kdd_columns.py.
- Encodes categorical features (protocol_type, service, flag) with consistent one-hot alignment.
- Normalizes numeric attributes using standard scaling.
- Generates binary targets (Normal = 0, Attack = 1) and multi-class categories (Normal, DoS, Probe, R2L, U2R).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.nsl_kdd_columns import NSL_KDD_COLUMNS, label_to_category


CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
EXCLUDE_COLS = ["label", "difficulty_level"]


def load_nsl_kdd_raw(file_path: Path) -> pd.DataFrame:
    """Load raw NSL-KDD text file into a DataFrame with standardized column names."""
    df = pd.read_csv(file_path, header=None, names=NSL_KDD_COLUMNS)
    df["category"] = df["label"].map(label_to_category)
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def prepare_nsl_kdd_datasets(
    train_path: Path,
    test_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.Series, pd.Series]:
    """Load train and test NSL-KDD datasets, apply unified one-hot encoding,
    and scale numerical attributes.

    Returns:
        X_train, y_train (binary), X_test, y_test (binary),
        feature_names, train_categories, test_categories
    """
    df_train = load_nsl_kdd_raw(train_path)
    df_test = load_nsl_kdd_raw(test_path)

    train_cats = df_train["category"]
    test_cats = df_test["category"]

    y_train = df_train["is_attack"].values
    y_test = df_test["is_attack"].values

    # Separate feature sets
    feat_train = df_train.drop(columns=EXCLUDE_COLS + ["category", "is_attack"])
    feat_test = df_test.drop(columns=EXCLUDE_COLS + ["category", "is_attack"])

    # One-hot encode categorical features across joint categories to ensure exact column alignment
    combined = pd.concat([feat_train, feat_test], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=CATEGORICAL_COLS, drop_first=False)

    feature_names = list(combined_encoded.columns)

    n_train = len(df_train)
    X_train_df = combined_encoded.iloc[:n_train].copy()
    X_test_df = combined_encoded.iloc[n_train:].copy()

    # Identify numeric columns for scaling
    num_cols = [c for c in feat_train.columns if c not in CATEGORICAL_COLS]

    scaler = StandardScaler()
    X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols].values)
    X_test_df[num_cols] = scaler.transform(X_test_df[num_cols].values)

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    return X_train, y_train, X_test, y_test, feature_names, train_cats, test_cats
