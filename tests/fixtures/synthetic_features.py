"""
Generates a synthetic features DataFrame (bypassing PCAP parsing entirely) for
testing the PAYL baseline. The Phase 2 fixture (tests/fixtures/sample.pcap)
only has 6 payloads -- nowhere near enough to build a meaningful mean/variance
profile or a percentile threshold. This generates dozens of "normal" HTTP-like
payloads with small random variation (so the profile has genuine spread to
learn from) plus a few clearly different "attack-like" payloads for testing
that the detector actually discriminates.
"""

import numpy as np
import pandas as pd

from preprocessing.feature_extractor import extract_features


def _random_http_like_payload(rng: np.random.Generator) -> bytes:
    """A GET request with a random path/host, so payloads vary slightly
    (like real traffic) rather than being byte-identical."""
    path_len = rng.integers(3, 12)
    path = "".join(rng.choice(list("abcdefghijklmnop"), size=path_len))
    host_len = rng.integers(5, 10)
    host = "".join(rng.choice(list("abcdefghijklmnop"), size=host_len))
    return f"GET /{path} HTTP/1.1\r\nHost: {host}.com\r\n\r\n".encode()


def _attack_like_payload(rng: np.random.Generator) -> bytes:
    """Structurally distinct from HTTP text -- a NOP-sled-style repeated
    binary pattern, the same kind used in the Phase 2/3 fixtures."""
    sled_len = int(rng.integers(30, 60))
    return b"\x90" * sled_len + b"\xde\xad\xbe\xef" * 8


def build_synthetic_features_df(
    config: dict,
    n_normal: int = 60,
    n_attack: int = 5,
    service: str = "http",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a features DataFrame in the same shape build_feature_dataframe()
    (scripts/extract_features.py) produces: service, length_bucket, entropy,
    printable_ratio, payload_len, byte_hist_000..255, ngram_hist_000..N,
    plus an `is_attack` column (ground truth, NOT something the real pipeline
    has -- only present here for testing).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_normal):
        payload = _random_http_like_payload(rng)
        feats = extract_features(payload, config)
        rows.append({
            "service": service,
            "length_bucket": feats["length_bucket"],
            "entropy": feats["entropy"],
            "printable_ratio": feats["printable_ratio"],
            "payload_len": feats["payload_len"],
            "is_attack": False,
            **{f"byte_hist_{i:03d}": v for i, v in enumerate(feats["byte_hist"])},
            **{f"ngram_hist_{i:03d}": v for i, v in enumerate(feats["ngram_hist"])},
        })

    for _ in range(n_attack):
        payload = _attack_like_payload(rng)
        feats = extract_features(payload, config)
        rows.append({
            "service": service,
            "length_bucket": feats["length_bucket"],
            "entropy": feats["entropy"],
            "printable_ratio": feats["printable_ratio"],
            "payload_len": feats["payload_len"],
            "is_attack": True,
            **{f"byte_hist_{i:03d}": v for i, v in enumerate(feats["byte_hist"])},
            **{f"ngram_hist_{i:03d}": v for i, v in enumerate(feats["ngram_hist"])},
        })

    return pd.DataFrame(rows)
