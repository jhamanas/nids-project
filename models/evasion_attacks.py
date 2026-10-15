"""
models/evasion_attacks.py

Adversarial Evasion and Mimicry Generators for Network Intrusion Detection (Phase 7):
1. Payload Padding Attack (D1):
   - Injects benign padding bytes into an attack payload to pull its byte distribution
     toward the target service's normal profile (e.g. blending shellcode with HTTP keywords).
   - Generates varying padding ratios (10%, 25%, 50%, 75%, 90% benign padding).
2. Fragmentation / Split Payload Attack (D2):
   - Splits a payload across multiple small fragments (e.g. MTU 16, 32, 64 bytes)
     to alter length buckets and disrupt multi-byte feature distributions.
3. Low-Rate Flood Evasion (D3):
   - Simulates lowering packet rate and flow duration to emulate human interaction,
     measuring how Flow Random Forest detection degrades compared to payload detection.
"""

from collections import Counter
import math
import numpy as np

from preprocessing.feature_extractor import byte_histogram, shannon_entropy, printable_ratio


def apply_padding_attack(
    attack_payload: bytes,
    benign_padding_pool: bytes,
    padding_ratio: float,
) -> bytes:
    """Pad an attack payload with benign bytes to mimic the target distribution.

    Args:
        attack_payload: Raw malicious payload bytes (e.g. shellcode, exploit request).
        benign_padding_pool: Sample benign bytes representing target normal profile.
        padding_ratio: Fraction of the final payload that consists of padding [0.0, 1.0).
                       e.g. 0.8 means 80% padding bytes, 20% attack bytes.
    """
    if padding_ratio <= 0.0 or not benign_padding_pool:
        return attack_payload

    padding_ratio = min(0.99, float(padding_ratio))
    n_attack = len(attack_payload)
    # n_total = n_attack / (1 - padding_ratio) => n_pad = n_total - n_attack
    n_pad = int(math.ceil(n_attack * padding_ratio / (1.0 - padding_ratio)))

    # Sample padding bytes cyclically from benign pool
    pool_len = len(benign_padding_pool)
    padding_bytes = bytes([benign_padding_pool[i % pool_len] for i in range(n_pad)])

    # Append padding to the attack payload (e.g. HTTP comment or URL parameter padding)
    return attack_payload + padding_bytes


def apply_fragmentation_attack(
    payload: bytes,
    fragment_size: int,
) -> list[bytes]:
    """Split a contiguous payload into discrete sub-fragments.

    Simulates IP/TCP-level fragmentation where an IDS inspecting individual
    fragments without session reassembly sees fragmented byte representations.
    """
    if fragment_size <= 0 or fragment_size >= len(payload):
        return [payload]
    return [payload[i : i + fragment_size] for i in range(0, len(payload), fragment_size)]


def extract_features_from_payload(payload: bytes) -> np.ndarray:
    """Extract standard 258-dimensional feature vector from raw bytes:
    [byte_hist_000 ... byte_hist_255, entropy, printable_ratio]
    """
    hist = byte_histogram(payload)
    ent = shannon_entropy(payload)
    pr = printable_ratio(payload)
    return np.concatenate([hist, [ent, pr]]).astype(np.float32)
