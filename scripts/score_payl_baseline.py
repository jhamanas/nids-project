#!/usr/bin/env python3
"""
Phase 4 — Score payload features against a trained PAYL baseline.

Usage:
    python scripts/score_payl_baseline.py \\
        --model models/payl_baseline.pkl \\
        --input data/processed/features_packet_friday_ddos.parquet \\
        --output data/processed/scored_friday_ddos.parquet

This is purely unsupervised scoring -- it does not require (or use) attack
labels. Every payload gets a payl_score (Mahalanobis distance) and a
payl_anomaly flag (score > the profile's learned threshold). Matching these
scores against CICIDS2017's actual attack labels for a proper precision/
recall evaluation is Phase 6 (that requires joining payloads back to the
flow-level CSV labels by 5-tuple + timestamp, which is a separate, non-trivial
step -- not yet built).

Processes the input in batches and streams results to the output file --
never loads the full features file (or the full scored output) into memory
at once, so this works the same whether the input is 6 rows or 6 million.
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.payl_baseline import load_profiles, score_features_streaming, top_n_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Score payloads against a trained PAYL baseline.")
    parser.add_argument("--model", required=True, help="Trained profile .pkl from scripts/train_payl_baseline.py")
    parser.add_argument("--input", required=True, help="Features Parquet to score (scripts/extract_features.py output)")
    parser.add_argument("--output", required=True, help="Output Parquet path with scores appended")
    parser.add_argument("--batch-size", type=int, default=200_000,
                         help="Rows processed per batch (default: 200000)")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"[FAIL] {model_path} not found. Run scripts/train_payl_baseline.py first.")
        sys.exit(1)
    if not input_path.exists():
        print(f"[FAIL] {input_path} not found. Run scripts/extract_features.py first.")
        sys.exit(1)

    print(f"Loading trained profile from {model_path} ...")
    loaded = load_profiles(model_path)
    profiles = loaded["profiles"]
    n_ok = sum(1 for p in profiles.values() if p["status"] == "ok")
    print(f"Loaded {len(profiles)} (service, length_bucket) profiles ({n_ok} usable).")

    total_rows = pq.ParquetFile(input_path).metadata.num_rows
    print(f"Input has {total_rows:,} rows to score. Processing in batches of {args.batch_size:,}.")

    if total_rows == 0:
        print("[WARN] Input file has 0 rows -- nothing to score. Exiting.")
        sys.exit(0)

    summary = score_features_streaming(input_path, output_path, profiles, batch_size=args.batch_size)
    print(f"\n[ OK ] Saved scored output -> {output_path}")

    n = summary["total_scored"]
    n_flagged = summary["n_flagged"]
    n_no_profile = summary["n_no_profile"]

    print(f"\n-- Summary --")
    print(f"  Total scored:         {n:,}")
    print(f"  Flagged anomalous:    {n_flagged:,} ({100 * n_flagged / n:.2f}%)")
    print(f"  No profile available: {n_no_profile:,} ({100 * n_no_profile / n:.2f}%) "
          f"-- these count as anomalous above, and mean this exact (service, length_bucket) "
          f"combination wasn't in the training data")

    if summary["service_total"]:
        print(f"\n-- Flagged rate by service --")
        for service in sorted(summary["service_total"]):
            rate = summary["service_flagged"][service] / summary["service_total"][service]
            print(f"  {service:<10} {rate:.2%}  (n={summary['service_total'][service]:,})")

    print(f"\n-- Top 10 highest-scoring payloads --")
    slim_cols = [c for c in ["service", "src_ip", "dst_ip", "dst_port", "payload_len", "payl_score"]]
    # payl_score only exists in the OUTPUT file (score_payl_baseline just wrote it),
    # so read the top-N back from there, not from the input.
    top10 = top_n_rows(output_path, n=10, score_col="payl_score", columns=slim_cols)
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
