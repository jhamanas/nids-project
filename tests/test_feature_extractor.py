"""
Tests for preprocessing/feature_extractor.py, using hand-constructed payloads
with known properties (not the pcap fixture -- these need exact byte control).
"""

import math

import numpy as np
import pytest

from preprocessing.feature_extractor import (
    byte_histogram, ngram_histogram, shannon_entropy, printable_ratio,
    length_bucket, extract_features,
)
from preprocessing.config_loader import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


# ---------- byte_histogram ----------

def test_byte_histogram_shape_and_normalization():
    payload = b"AAAABBBBCCCC"
    hist = byte_histogram(payload)
    assert hist.shape == (256,)
    assert math.isclose(hist.sum(), 1.0, rel_tol=1e-9)


def test_byte_histogram_uniform_payload():
    # All-same-byte payload -> a single spike at that byte value, entropy 0
    payload = bytes([0x41]) * 100  # 'A' * 100
    hist = byte_histogram(payload)
    assert hist[0x41] == 1.0
    assert hist.sum() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 1


def test_byte_histogram_empty_payload():
    hist = byte_histogram(b"")
    assert hist.shape == (256,)
    assert hist.sum() == 0.0


# ---------- ngram_histogram ----------

def test_ngram_histogram_shape(config):
    payload = b"GET /index.html HTTP/1.1\r\n"
    hist = ngram_histogram(payload, ngram_size=3, num_buckets=256)
    assert hist.shape == (256,)
    assert math.isclose(hist.sum(), 1.0, rel_tol=1e-9)


def test_ngram_histogram_short_payload_returns_zeros():
    # payload shorter than ngram_size -> no n-grams possible
    hist = ngram_histogram(b"AB", ngram_size=3, num_buckets=64)
    assert hist.sum() == 0.0


def test_ngram_histogram_deterministic_across_calls():
    # CRC32-based hashing must be reproducible (unlike Python's built-in hash(),
    # which is randomized per-process) -- this is the whole reason CRC32 was used.
    payload = b"repeated pattern repeated pattern"
    hist1 = ngram_histogram(payload, ngram_size=3, num_buckets=128)
    hist2 = ngram_histogram(payload, ngram_size=3, num_buckets=128)
    assert np.array_equal(hist1, hist2)


# ---------- shannon_entropy ----------

def test_entropy_zero_for_uniform_byte():
    payload = bytes([0x00]) * 50
    assert shannon_entropy(payload) == 0.0


def test_entropy_near_max_for_high_diversity_payload():
    # All 256 byte values, each exactly once -> maximum entropy = 8.0 bits
    payload = bytes(range(256))
    entropy = shannon_entropy(payload)
    assert entropy == pytest.approx(8.0, abs=1e-9)


def test_entropy_ordering_nopsled_vs_text():
    # A NOP-sled-style repeated pattern should have LOWER entropy than
    # natural-language-like HTTP text, since it's far more repetitive.
    nopsled = b"\x90" * 40 + b"\xde\xad\xbe\xef" * 10
    http_text = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: test\r\n\r\n"
    assert shannon_entropy(nopsled) < shannon_entropy(http_text)


def test_entropy_empty_payload():
    assert shannon_entropy(b"") == 0.0


# ---------- printable_ratio ----------

def test_printable_ratio_all_printable():
    payload = b"GET /index.html HTTP/1.1"
    assert printable_ratio(payload) == 1.0


def test_printable_ratio_all_nonprintable():
    payload = bytes([0x90, 0xDE, 0xAD, 0xBE, 0xEF])
    assert printable_ratio(payload) == 0.0


def test_printable_ratio_mixed():
    payload = b"AB" + bytes([0x00, 0x01])  # 2 printable, 2 non-printable
    assert printable_ratio(payload) == 0.5


def test_printable_ratio_empty_payload():
    assert printable_ratio(b"") == 0.0


# ---------- length_bucket ----------

def test_length_bucket_boundaries():
    assert length_bucket(0, bucket_size=256) == 0
    assert length_bucket(255, bucket_size=256) == 0
    assert length_bucket(256, bucket_size=256) == 1
    assert length_bucket(1500, bucket_size=256) == 5


# ---------- extract_features (integration of all of the above) ----------

def test_extract_features_returns_all_expected_keys(config):
    payload = b"GET /index.html HTTP/1.1\r\n\r\n"
    features = extract_features(payload, config)
    expected_keys = {"byte_hist", "ngram_hist", "entropy", "printable_ratio",
                      "length_bucket", "payload_len"}
    assert set(features.keys()) == expected_keys
    assert features["payload_len"] == len(payload)
    assert features["byte_hist"].shape == (256,)


def test_extract_features_distinguishes_anomalous_from_normal(config):
    normal = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
    anomalous = b"\x90" * 40 + b"\xde\xad\xbe\xef" * 10

    normal_feats = extract_features(normal, config)
    anomalous_feats = extract_features(anomalous, config)

    # This is the whole premise of PAYL-style detection: structurally
    # distinct payloads should show up as distinct in these basic features.
    assert normal_feats["printable_ratio"] > anomalous_feats["printable_ratio"]
    assert normal_feats["entropy"] != anomalous_feats["entropy"]
    assert not np.array_equal(normal_feats["byte_hist"], anomalous_feats["byte_hist"])
