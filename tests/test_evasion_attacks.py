"""
tests/test_evasion_attacks.py

Unit tests for models/evasion_attacks.py:
- Payload padding attack logic and byte ratios.
- Payload fragmentation and chunking.
- Feature extraction consistency.
"""

import numpy as np
import pytest

from models.evasion_attacks import (
    apply_padding_attack,
    apply_fragmentation_attack,
    extract_features_from_payload,
)


def test_apply_padding_attack():
    attack = b"\x90\x90\x90\x90"  # 4 bytes
    benign_pool = b"GET /index.html HTTP/1.1\r\n"

    # 0% padding
    res_0 = apply_padding_attack(attack, benign_pool, padding_ratio=0.0)
    assert res_0 == attack

    # 50% padding (should roughly double length)
    res_50 = apply_padding_attack(attack, benign_pool, padding_ratio=0.5)
    assert len(res_50) >= 8
    assert res_50.startswith(attack)


def test_apply_fragmentation_attack():
    payload = b"A" * 64
    fragments = apply_fragmentation_attack(payload, fragment_size=16)

    assert len(fragments) == 4
    assert all(len(f) == 16 for f in fragments)
    assert b"".join(fragments) == payload


def test_extract_features_from_payload():
    payload = b"Hello, World!"
    feats = extract_features_from_payload(payload)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (258,)
    # Sum of byte histogram frequencies should be 1.0
    assert pytest.approx(np.sum(feats[:256]), 1e-4) == 1.0
    # Entropy and printable ratio should be valid
    assert 0.0 <= feats[256] <= 8.0
    assert 0.0 <= feats[257] <= 1.0
