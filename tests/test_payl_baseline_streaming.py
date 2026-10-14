"""
Regression tests for the streaming variants in models/payl_baseline.py
(build_profiles_streaming, score_features_streaming, top_n_rows).

The critical property being tested here isn't just "doesn't crash on large
data" (that's covered by manual stress testing) -- it's that the streaming,
batched computation produces NUMERICALLY IDENTICAL results to the original
in-memory build_profiles()/score_features() on the same data. If these ever
diverge, the streaming versions have a bug, however fast or memory-safe they
are.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.payl_baseline import (
    build_profiles, score_features,
    build_profiles_streaming, score_features_streaming, top_n_rows,
)
from preprocessing.config_loader import load_config
from tests.fixtures.synthetic_features import build_synthetic_features_df


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def synthetic_df(config):
    # Bigger and split across two services so grouping/batching logic is
    # actually exercised across multiple batches, not just one.
    http_df = build_synthetic_features_df(config, n_normal=150, n_attack=10, service="http", seed=1)
    ftp_df = build_synthetic_features_df(config, n_normal=150, n_attack=10, service="ftp", seed=2)
    return pd.concat([http_df, ftp_df], ignore_index=True)


def test_streaming_profiles_match_in_memory_profiles(synthetic_df, config, tmp_path):
    normal_only = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"]).reset_index(drop=True)

    # In-memory reference
    in_memory_profiles = build_profiles(normal_only, config)

    # Streaming version, deliberately using a small batch size (50) so the
    # 300 normal rows span 6+ batches -- this is what actually exercises the
    # cross-batch accumulation logic, not just "small enough to fit in one batch".
    input_path = tmp_path / "normal_features.parquet"
    normal_only.to_parquet(input_path, index=False)
    streaming_profiles = build_profiles_streaming(input_path, config, batch_size=50, verbose=False)

    assert set(in_memory_profiles.keys()) == set(streaming_profiles.keys())

    for key in in_memory_profiles:
        a, b = in_memory_profiles[key], streaming_profiles[key]
        assert a["status"] == b["status"]
        if a["status"] != "ok":
            continue
        np.testing.assert_allclose(a["mean"], b["mean"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a["var"], b["var"], rtol=1e-6, atol=1e-9)
        # Threshold is a percentile over per-sample distances -- allow slightly
        # looser tolerance since it's derived from derived values (mean/var),
        # compounding floating-point differences a bit, but should still be
        # very close.
        assert a["threshold"] == pytest.approx(b["threshold"], rel=1e-4)


def test_streaming_scoring_matches_in_memory_scoring(synthetic_df, config, tmp_path):
    normal_only = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"]).reset_index(drop=True)
    attack_only = synthetic_df[synthetic_df["is_attack"]].drop(columns=["is_attack"]).reset_index(drop=True)

    profiles = build_profiles(normal_only, config)

    # Score the attack rows both ways and compare
    in_memory_scored = score_features(attack_only, profiles)

    input_path = tmp_path / "attack_features.parquet"
    output_path = tmp_path / "attack_scored.parquet"
    attack_only.to_parquet(input_path, index=False)
    score_features_streaming(input_path, output_path, profiles, batch_size=7, verbose=False)
    streaming_scored = pd.read_parquet(output_path)

    # Row order may differ across batches vs. one-shot grouping -- sort both
    # by a stable key before comparing.
    sort_cols = ["service", "payload_len"]
    a = in_memory_scored.sort_values(sort_cols).reset_index(drop=True)
    b = streaming_scored.sort_values(sort_cols).reset_index(drop=True)

    np.testing.assert_allclose(
        a["payl_score"].replace([np.inf], 1e18),
        b["payl_score"].replace([np.inf], 1e18),
        rtol=1e-6,
    )
    assert (a["payl_anomaly"] == b["payl_anomaly"]).all()


def test_top_n_rows_matches_full_sort(synthetic_df, tmp_path):
    df = synthetic_df.copy()
    df["fake_score"] = np.arange(len(df))  # deterministic, easy to verify top-N
    path = tmp_path / "for_topn.parquet"
    df.to_parquet(path, index=False)

    result = top_n_rows(path, n=5, score_col="fake_score", columns=["fake_score"], batch_size=13)
    expected_top5 = sorted(df["fake_score"].tolist(), reverse=True)[:5]
    assert sorted(result["fake_score"].tolist(), reverse=True) == expected_top5


def test_streaming_handles_multiple_batches_correctly(config, tmp_path):
    # Specifically construct a case where a single (service, length_bucket)
    # group's rows are split across several batches, to catch any bug where
    # accumulation resets per-batch instead of persisting across them.
    df = build_synthetic_features_df(config, n_normal=100, n_attack=0, service="http", seed=99)
    df = df.drop(columns=["is_attack"])
    path = tmp_path / "multi_batch.parquet"
    df.to_parquet(path, index=False)

    # batch_size=10 forces at least 10 batches for 100 rows
    profiles = build_profiles_streaming(path, config, batch_size=10, verbose=False)
    ok_profiles = [p for p in profiles.values() if p["status"] == "ok"]
    assert len(ok_profiles) >= 1
    # n_samples should reflect ALL 100 rows accumulated across batches, not
    # just whatever was in the last batch seen
    total_n = sum(p["n_samples"] for p in profiles.values())
    assert total_n == 100
