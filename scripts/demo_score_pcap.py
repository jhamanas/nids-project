"""
scripts/demo_score_pcap.py

Phase 8 Interactive Demo (Task E4):
Scores an arbitrary input PCAP file end-to-end using both trained NIDS detectors:
1. Statistical PAYL Baseline (diagonal Mahalanobis distance).
2. Deep PyTorch Autoencoder (reconstruction error MSE).

Workflow:
- Ingests raw PCAP packets with Scapy.
- Extracts streaming payload features (byte histograms, Shannon entropy, printable ratio).
- Evaluates anomaly scores against learned normal baseline profiles and thresholds.
- Outputs a terminal dashboard highlighting detected malicious packets, source/destination
  endpoints, anomaly scores, and inspects the raw payload hex & ASCII bytes.

Usage:
    python scripts/demo_score_pcap.py --pcap tests/fixtures/sample.pcap
"""

import argparse
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.autoencoder import load_autoencoder, compute_reconstruction_error
from preprocessing.feature_extractor import byte_histogram, shannon_entropy, printable_ratio, length_bucket
from preprocessing.pcap_parser import extract_packet_payloads


def format_payload_preview(payload: bytes, max_len: int = 48) -> str:
    """Format payload into human-readable ASCII representation with non-printable dots."""
    preview = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in payload[:max_len])
    if len(payload) > max_len:
        preview += f"... (+{len(payload) - max_len} bytes)"
    return preview


def demo_score_pcap(
    pcap_path: Path,
    payl_profiles_path: Path,
    ae_model_path: Path,
    show_all: bool = False,
):
    print("=" * 80)
    print(f"  NIDS DEMO: End-to-End Real-Time PCAP Payload Inspection (Phase 8)")
    print(f"  Target PCAP: {pcap_path.name}")
    print("=" * 80)

    # 1. Load Models
    print(f"[1/3] Loading trained models...")
    with open(payl_profiles_path, "rb") as f:
        loaded = pickle.load(f)
        profiles = loaded.get("profiles", loaded) if isinstance(loaded, dict) else loaded
    print(f"      - PAYL Baseline: Loaded {len(profiles)} normal traffic profiles.")

    ae_model, ae_threshold, _ = load_autoencoder(ae_model_path)
    print(f"      - PyTorch Autoencoder: Loaded model (Anomaly Threshold: {ae_threshold:.6f}).")

    # 2. Parse PCAP
    print(f"\n[2/3] Parsing packets from {pcap_path}...")
    cfg = {
        "services": {"http": 80, "https": 443, "dns": 53, "ftp": 21, "ssh": 22, "smtp": 25},
        "features": {"max_payload_bytes": 1460, "length_bucket_size": 64},
    }
    extracted = extract_packet_payloads(pcap_path, config=cfg)
    records = []
    raw_payloads = []

    for pkt in extracted:
        payload = pkt["payload"]
        raw_payloads.append(payload)

        # Feature extraction
        hist = byte_histogram(payload)
        ent = shannon_entropy(payload)
        pr = printable_ratio(payload)
        n_len = pkt["payload_len"]
        l_bucket = length_bucket(n_len, bucket_size=64)

        row = {
            "timestamp": pkt.get("timestamp"),
            "src_ip": pkt.get("src_ip"),
            "dst_ip": pkt.get("dst_ip"),
            "src_port": pkt.get("src_port"),
            "dst_port": pkt.get("dst_port"),
            "protocol": pkt.get("protocol"),
            "service": pkt.get("service", "other"),
            "payload_len": n_len,
            "length_bucket": l_bucket,
            "entropy": ent,
            "printable_ratio": pr,
        }
        for i, val in enumerate(hist):
            row[f"byte_hist_{i:03d}"] = val
        records.append(row)

    if not records:
        print(f"[!] No packets with raw payload found in {pcap_path}.")
        return

    df = pd.DataFrame(records)
    print(f"      - Extracted {len(df):,} payloads with valid application data.")

    # 3. Score with PAYL
    print(f"\n[3/3] Scoring payloads against models...")
    payl_scores = []
    payl_anomalies = []
    feat_cols = [f"byte_hist_{i:03d}" for i in range(256)]

    for _, row in df.iterrows():
        srv = row["service"]
        b_idx = row["length_bucket"]
        prof = profiles.get((srv, b_idx))
        if not prof or prof.get("status") != "ok":
            # fallback
            srv_profs = [p for k, p in profiles.items() if k[0] == srv and p.get("status") == "ok"]
            prof = srv_profs[0] if srv_profs else None

        if prof:
            x = row[feat_cols].to_numpy(dtype=np.float64)
            diff = x - prof["mean"]
            dist = float(np.sqrt(np.sum((diff ** 2) / prof["var"])))
            flag = dist > prof["threshold"]
            payl_scores.append(dist)
            payl_anomalies.append(flag)
        else:
            payl_scores.append(float("inf"))
            payl_anomalies.append(True)

    df["payl_score"] = payl_scores
    df["payl_anomaly"] = payl_anomalies

    # Score with Autoencoder
    byte_cols = [f"byte_hist_{i:03d}" for i in range(256)] + ["entropy", "printable_ratio"]
    feat_matrix = df[byte_cols].to_numpy(dtype=np.float32)
    ae_scores = compute_reconstruction_error(ae_model, feat_matrix)
    df["ae_score"] = ae_scores
    df["ae_anomaly"] = ae_scores > ae_threshold

    # Summary Display
    print("\n" + "=" * 80)
    print("  INSPECTION RESULTS & ALERTS")
    print("=" * 80)

    anomalous_df = df[df["payl_anomaly"] | df["ae_anomaly"]]
    print(f"Total Packets Inspected:   {len(df):,}")
    print(f"PAYL Anomaly Alerts:       {int(df['payl_anomaly'].sum()):,} ({100 * df['payl_anomaly'].mean():.1f}%)")
    print(f"Autoencoder Alerts:        {int(df['ae_anomaly'].sum()):,} ({100 * df['ae_anomaly'].mean():.1f}%)")
    print(f"Jointly Flagged (Consensus): {int((df['payl_anomaly'] & df['ae_anomaly']).sum()):,}")
    print("-" * 80)

    rows_to_display = df if show_all else (anomalous_df if len(anomalous_df) > 0 else df.head(5))

    for idx, row in rows_to_display.iterrows():
        p_flag = "[ALERT]" if row["payl_anomaly"] else "[NORMAL]"
        a_flag = "[ALERT]" if row["ae_anomaly"] else "[NORMAL]"
        payload_bytes = raw_payloads[idx]

        print(f"Packet #{idx + 1:03d} | {row['src_ip']}:{row['src_port']} -> {row['dst_ip']}:{row['dst_port']} ({row['service'].upper()})")
        print(f"  Length: {row['payload_len']}B | Entropy: {row['entropy']:.2f} | Printable: {row['printable_ratio']*100:.1f}%")
        print(f"  PAYL Distance:        {row['payl_score']:8.2f}  {p_flag}")
        print(f"  Autoencoder MSE:      {row['ae_score']:8.6f}  {a_flag} (Threshold: {ae_threshold:.6f})")
        print(f"  Payload ASCII:        {format_payload_preview(payload_bytes)}")
        print(f"  Payload Hex:          {payload_bytes[:24].hex()}...")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="End-to-end interactive demo of NIDS PCAP scoring")
    parser.add_argument("--pcap", type=Path, default=ROOT / "tests" / "fixtures" / "sample.pcap", help="Path to input PCAP")
    parser.add_argument("--payl-profiles", type=Path, default=ROOT / "models" / "payl_baseline.pkl")
    parser.add_argument("--ae-model", type=Path, default=ROOT / "models" / "autoencoder.pt")
    parser.add_argument("--show-all", action="store_true", help="Display all packets, not just flagged anomalies")
    args = parser.parse_args()

    demo_score_pcap(
        pcap_path=args.pcap,
        payl_profiles_path=args.payl_profiles,
        ae_model_path=args.ae_model,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()
