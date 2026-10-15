"""
scripts/run_pipeline.py

Phase 8 Master Pipeline Runner (Task E1):
Executes the entire NIDS pipeline end-to-end on a specified dataset or sample PCAP:
1. PCAP Ingestion & Raw Payload Extraction (`preprocessing/pcap_parser.py`).
2. Feature Engineering: 258-dim feature extraction (`preprocessing/feature_extractor.py`).
3. Statistical PAYL Baseline Scoring (`models/payl_baseline.py`).
4. Deep PyTorch Autoencoder Scoring (`models/autoencoder.py`).
5. Performance Evaluation & Alert Summary (`evaluation/metrics.py`).

Usage:
    python scripts/run_pipeline.py --pcap tests/fixtures/sample.pcap --output-dir data/processed/sample_run
"""

import argparse
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.autoencoder import load_autoencoder, compute_reconstruction_error
from preprocessing.feature_extractor import byte_histogram, shannon_entropy, printable_ratio, length_bucket
from preprocessing.pcap_parser import extract_packet_payloads


def run_pipeline(
    pcap_path: Path,
    payl_profiles_path: Path,
    ae_model_path: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 80)
    print("  NIDS PIPELINE RUNNER (Phase 8 Master Orchestrator)")
    print(f"  Target PCAP:    {pcap_path}")
    print(f"  Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Parse PCAP
    print("\n[Step 1/4] Ingesting and extracting payloads from PCAP...")
    cfg = {
        "services": {"http": 80, "https": 443, "dns": 53, "ftp": 21, "ssh": 22, "smtp": 25},
        "features": {"max_payload_bytes": 1460, "length_bucket_size": 64},
    }
    extracted = extract_packet_payloads(pcap_path, config=cfg)
    print(f"           Extracted {len(extracted):,} application payload frames.")

    if not extracted:
        print("[!] No payload frames to process. Exiting.")
        return

    # 2. Extract Features
    print("\n[Step 2/4] Generating 258-dim feature representations...")
    records = []
    for pkt in extracted:
        p_bytes = pkt["payload"]
        p_len = pkt["payload_len"]
        hist = byte_histogram(p_bytes)
        row = {
            "pcap_file": pcap_path.name,
            "timestamp": pkt.get("timestamp"),
            "src_ip": pkt.get("src_ip"),
            "dst_ip": pkt.get("dst_ip"),
            "src_port": pkt.get("src_port"),
            "dst_port": pkt.get("dst_port"),
            "protocol": pkt.get("protocol"),
            "service": pkt.get("service", "other"),
            "payload_len": p_len,
            "length_bucket": length_bucket(p_len, bucket_size=64),
            "entropy": shannon_entropy(p_bytes),
            "printable_ratio": printable_ratio(p_bytes),
        }
        for i, val in enumerate(hist):
            row[f"byte_hist_{i:03d}"] = val
        records.append(row)

    df = pd.DataFrame(records)
    feat_parquet = output_dir / "extracted_features.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), feat_parquet)
    print(f"           Saved extracted features to {feat_parquet}")

    # 3. Model Scoring
    print("\n[Step 3/4] Evaluating payloads with PAYL & Autoencoder detectors...")
    with open(payl_profiles_path, "rb") as f:
        loaded = pickle.load(f)
        profiles = loaded.get("profiles", loaded) if isinstance(loaded, dict) else loaded

    ae_model, ae_threshold, _ = load_autoencoder(ae_model_path)

    # PAYL Scoring
    payl_scores = []
    payl_anomalies = []
    feat_cols = [f"byte_hist_{i:03d}" for i in range(256)]

    for _, row in df.iterrows():
        srv = row["service"]
        b_idx = row["length_bucket"]
        prof = profiles.get((srv, b_idx))
        if not prof or prof.get("status") != "ok":
            srv_profs = [p for k, p in profiles.items() if k[0] == srv and p.get("status") == "ok"]
            prof = srv_profs[0] if srv_profs else None

        if prof:
            x = row[feat_cols].to_numpy(dtype=np.float64)
            diff = x - prof["mean"]
            dist = float(np.sqrt(np.sum((diff ** 2) / prof["var"])))
            payl_scores.append(dist)
            payl_anomalies.append(dist > prof["threshold"])
        else:
            payl_scores.append(float("inf"))
            payl_anomalies.append(True)

    df["payl_score"] = payl_scores
    df["payl_anomaly"] = payl_anomalies

    # Autoencoder Scoring
    ae_feat_cols = [f"byte_hist_{i:03d}" for i in range(256)] + ["entropy", "printable_ratio"]
    ae_scores = compute_reconstruction_error(ae_model, df[ae_feat_cols].to_numpy(dtype=np.float32))
    df["autoencoder_score"] = ae_scores
    df["autoencoder_anomaly"] = ae_scores > ae_threshold

    scored_parquet = output_dir / "scored_packets.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), scored_parquet)
    print(f"           Saved scored results to {scored_parquet}")

    # 4. Summary Output
    print("\n[Step 4/4] Pipeline Execution Summary:")
    print(f"  - Total Packets Processed: {len(df):,}")
    print(f"  - PAYL Flagged:            {int(df['payl_anomaly'].sum()):,} ({100*df['payl_anomaly'].mean():.1f}%)")
    print(f"  - Autoencoder Flagged:     {int(df['autoencoder_anomaly'].sum()):,} ({100*df['autoencoder_anomaly'].mean():.1f}%)")
    print(f"  - Joint Consensus Alerts:  {int((df['payl_anomaly'] & df['autoencoder_anomaly']).sum()):,}")
    print(f"  - Total Elapsed Time:      {time.time() - t0:.2f}s")
    print("=" * 80)
    print("[ OK ] Pipeline completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run NIDS detection pipeline end-to-end")
    parser.add_argument("--pcap", type=Path, default=ROOT / "tests" / "fixtures" / "sample.pcap")
    parser.add_argument("--payl-profiles", type=Path, default=ROOT / "models" / "payl_baseline.pkl")
    parser.add_argument("--ae-model", type=Path, default=ROOT / "models" / "autoencoder.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "sample_run")
    args = parser.parse_args()

    run_pipeline(
        pcap_path=args.pcap,
        payl_profiles_path=args.payl_profiles,
        ae_model_path=args.ae_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
