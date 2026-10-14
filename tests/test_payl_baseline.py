"""
Tests for models/payl_baseline.py.

Uses tests/fixtures/synthetic_features.py rather than the tiny Phase 2 PCAP
fixture, since profile-building needs enough samples to have genuine
mean/variance spread and a meaningful percentile threshold -- 6 payloads
(the PCAP fixture's size) isn't enough for that.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.payl_baseline import build_profiles, score_features, save_profiles, load_profiles
from preprocessing.config_loader import load_config
from tests.fixtures.synthetic_features import build_synthetic_features_df


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def synthetic_df(config):
    return build_synthetic_features_df(config, n_normal=60, n_attack=5, seed=42)


@pytest.fixture(scope="module")
def profiles(synthetic_df, config):
    # Train only on the normal rows -- exactly like a real deployment, where
    # you'd build profiles from known-benign traffic (e.g. CICIDS2017 Monday).
    normal_only = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"])
    return build_profiles(normal_only, config)


# ---------- build_profiles ----------

def test_profile_built_for_populated_group(profiles):
    # All 60 synthetic normal payloads are http; they'll land in a small
    # number of length buckets depending on the random path/host lengths.
    ok_profiles = [p for p in profiles.values() if p["status"] == "ok"]
    assert len(ok_profiles) >= 1


def test_profile_has_expected_fields(profiles):
    ok_profiles = [p for p in profiles.values() if p["status"] == "ok"]
    profile = ok_profiles[0]
    assert profile["mean"].shape == (256,)
    assert profile["var"].shape == (256,)
    assert profile["threshold"] > 0
    assert profile["n_samples"] >= 1


def test_insufficient_data_marks_small_groups(config):
    # A group with fewer than min_samples_per_profile should be marked
    # insufficient rather than given an unreliable profile.
    tiny_df = build_synthetic_features_df(config, n_normal=3, n_attack=0, seed=1)
    tiny_profiles = build_profiles(tiny_df.drop(columns=["is_attack"]), config)
    statuses = {p["status"] for p in tiny_profiles.values()}
    assert "insufficient_data" in statuses
    for p in tiny_profiles.values():
        if p["status"] == "insufficient_data":
            assert "mean" not in p  # no unreliable profile was built


def test_variance_never_zero(profiles):
    # variance_epsilon in config must prevent divide-by-zero at scoring time
    # for byte positions that never vary (e.g. bytes that never appear in
    # HTTP text) -- this is the whole reason epsilon exists.
    for p in profiles.values():
        if p["status"] == "ok":
            assert np.all(p["var"] > 0)


# ---------- score_features ----------

def test_normal_payloads_mostly_not_flagged(synthetic_df, profiles):
    normal_only = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"])
    scored = score_features(normal_only, profiles)
    # The threshold is the alert_percentile (default 99th) of the TRAINING
    # distances themselves, so ~1% of normal training data scoring above it
    # is expected and correct -- not a bug. The overwhelming majority should
    # still score below it.
    false_positive_rate = scored["payl_anomaly"].mean()
    assert false_positive_rate < 0.10, (
        f"Too many normal payloads flagged: {false_positive_rate:.1%} "
        f"(expected roughly ~1-10% given a 99th-percentile threshold)"
    )


def test_attack_payloads_flagged_more_than_normal(synthetic_df, profiles):
    # This is the core premise of the whole PAYL approach: structurally
    # distinct payloads should score as more anomalous than normal ones,
    # even though the detector never saw an "attack" label during training.
    normal_only = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"])
    attack_only = synthetic_df[synthetic_df["is_attack"]].drop(columns=["is_attack"])

    scored_normal = score_features(normal_only, profiles)
    scored_attack = score_features(attack_only, profiles)

    normal_flag_rate = scored_normal["payl_anomaly"].mean()
    attack_flag_rate = scored_attack["payl_anomaly"].mean()

    assert attack_flag_rate > normal_flag_rate
    # With such structurally different payloads (all-printable text vs.
    # mostly-nonprintable repeated bytes), we should catch essentially all of them.
    assert attack_flag_rate >= 0.8


def test_unseen_service_scores_as_maximally_anomalous(synthetic_df, profiles):
    # A (service, length_bucket) combination never seen in training should be
    # treated as maximally anomalous, not silently passed through -- this is
    # the documented design choice in payl_baseline.py's module docstring.
    unseen = synthetic_df[~synthetic_df["is_attack"]].drop(columns=["is_attack"]).iloc[[0]].copy()
    unseen["service"] = "totally_unseen_service"
    scored = score_features(unseen, profiles)
    assert scored["payl_anomaly"].iloc[0] == True  # noqa: E712
    assert scored["payl_score"].iloc[0] == np.inf
    assert scored["payl_profile_available"].iloc[0] == False  # noqa: E712


# ---------- save/load ----------

def test_save_and_load_profiles_roundtrip(profiles, config, tmp_path):
    save_path = tmp_path / "test_payl_profile.pkl"
    save_profiles(profiles, config, save_path)
    assert save_path.exists()

    loaded = load_profiles(save_path)
    assert "profiles" in loaded
    assert "payl_config" in loaded

    # Scoring with the reloaded profiles should give identical results to
    # scoring with the original in-memory profiles.
    reloaded_profiles = loaded["profiles"]
    for key in profiles:
        if profiles[key]["status"] == "ok":
            assert np.array_equal(profiles[key]["mean"], reloaded_profiles[key]["mean"])
            assert np.array_equal(profiles[key]["var"], reloaded_profiles[key]["var"])
