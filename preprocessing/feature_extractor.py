"""
Phase 3 — Payload feature engineering.

Turns a raw payload (bytes) into the numeric features the PAYL baseline
(Phase 4) and autoencoder (Phase 5) consume:

  - byte_histogram      256-dim normalized byte-value frequency vector
                        (the core PAYL feature -- Wang & Stolfo 2004)
  - ngram_histogram      fixed-size normalized n-gram frequency vector, via a
                        hashing trick (raw n-gram space is 256^n, intractable
                        to represent directly -- Anagram-style extension)
  - entropy              Shannon entropy of the payload bytes, in bits
  - printable_ratio      fraction of bytes in the printable ASCII range
  - length_bucket        which PAYL length-bucket this payload falls into

Each function operates on a single payload; scripts/extract_features.py
applies these across an entire Parquet file of extracted payloads.
"""

import math
import zlib
from collections import Counter

import numpy as np

PRINTABLE_MIN, PRINTABLE_MAX = 0x20, 0x7E  # space through '~'


def byte_histogram(payload: bytes) -> np.ndarray:
    """256-dim normalized byte-value frequency vector."""
    if not payload:
        return np.zeros(256, dtype=np.float64)
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    return counts.astype(np.float64) / len(payload)


def ngram_histogram(payload: bytes, ngram_size: int, num_buckets: int) -> np.ndarray:
    """
    Fixed-size normalized n-gram frequency vector.

    Uses CRC32 (not Python's built-in hash(), which is randomized per-process
    for strings/bytes as a security measure and would make features
    non-reproducible across runs) to hash each n-gram into one of
    `num_buckets` buckets, then counts bucket frequency. This is the
    standard "hashing trick" used to keep n-gram feature vectors a fixed,
    tractable size regardless of the (astronomically large) raw n-gram space.
    """
    vec = np.zeros(num_buckets, dtype=np.float64)
    if len(payload) < ngram_size:
        return vec

    n_ngrams = len(payload) - ngram_size + 1
    for i in range(n_ngrams):
        chunk = payload[i:i + ngram_size]
        bucket = zlib.crc32(chunk) % num_buckets
        vec[bucket] += 1

    return vec / n_ngrams


def shannon_entropy(payload: bytes) -> float:
    """Shannon entropy in bits (0 = all one byte value, 8 = uniform over 256 values)."""
    if not payload:
        return 0.0
    counts = Counter(payload)
    length = len(payload)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def printable_ratio(payload: bytes) -> float:
    """Fraction of bytes in the printable ASCII range [0x20, 0x7E]."""
    if not payload:
        return 0.0
    printable = sum(1 for b in payload if PRINTABLE_MIN <= b <= PRINTABLE_MAX)
    return printable / len(payload)


def length_bucket(payload_len: int, bucket_size: int) -> int:
    """Which length-bucket this payload falls into (PAYL segments profiles this way)."""
    return payload_len // bucket_size


def extract_features(payload: bytes, config: dict) -> dict:
    """
    Extract all Phase 3 features for a single payload.

    Returns a flat dict: {"byte_hist": np.ndarray(256,), "ngram_hist": np.ndarray(N,),
    "entropy": float, "printable_ratio": float, "length_bucket": int,
    "payload_len": int}
    """
    feat_cfg = config["features"]
    return {
        "byte_hist": byte_histogram(payload),
        "ngram_hist": ngram_histogram(payload, feat_cfg["ngram_size"], feat_cfg["ngram_buckets"]),
        "entropy": shannon_entropy(payload),
        "printable_ratio": printable_ratio(payload),
        "length_bucket": length_bucket(len(payload), feat_cfg["length_bucket_size"]),
        "payload_len": len(payload),
    }
