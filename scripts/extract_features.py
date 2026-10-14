#!/usr/bin/env python3
"""
Phase 3 — Extract features from the payload records produced by
scripts/parse_pcaps.py (Phase 2).

Usage:
    # From packet-level payloads:
    python scripts/extract_features.py --input data/processed/payloads_packet.parquet \\
        --output data/processed/features_packet.parquet

    # From session-level (TCP-reassembled) payloads:
    python scripts/extract_features.py --input data/processed/payloads_session.parquet \\
        --output data/processed/features_session.parquet

    # Try it right now against the Phase 2 fixture output:
    python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode both
    python scripts/extract_features.py --input data/processed/payloads_packet.parquet \\
        --output data/processed/features_packet.parquet

    # On a very large file (millions of rows), lower --batch-size if you hit
    # memory pressure, or raise it (fewer, larger batches) if you have RAM to
    # spare and want it to run faster:
    python scripts/extract_features.py --input data/processed/payloads_packet_monday.parquet \\
        --output data/processed/features_packet_monday.parquet --batch-size 100000

Output columns:
    All metadata columns from the input (service, ports, timestamps, etc.) are
    carried through unchanged, plus:
        entropy, printable_ratio, length_bucket, payload_len   (scalars)
        byte_hist_000 .. byte_hist_255                            (256 columns)
        ngram_hist_000 .. ngram_hist_{N-1}                        (N columns, N = config ngram_buckets)

    Byte-histogram and n-gram columns are expanded (not left as array columns)
    so Phase 4/5 can consume them directly as a flat numeric feature matrix
    (e.g. `df.filter(like="byte_hist_").values`).

IMPORTANT (memory): this script processes the input in fixed-size batches and
writes each batch's features to the output Parquet file as it goes -- it
never holds the full feature matrix in memory at once. This matters a lot in
practice: a real CICIDS2017 day (e.g. Monday's ~5.7 million packets) would
need ~23 GB just to hold one dense in-memory copy of the full feature matrix
(5.7M rows x 524 columns x 8 bytes) -- more RAM than most machines have.
Batching keeps peak memory to roughly (batch_size x 524 x 8 bytes), a few
hundred MB at the default batch size, regardless of how large the input file is.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.feature_extractor import extract_features  # noqa: E402
from preprocessing.config_loader import load_config  # noqa: E402


def build_feature_dataframe(payloads_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract features for one batch of payloads. Called once per batch by
    main() below -- never on the full dataset at once for large files.
    """
    ngram_buckets = config["features"]["ngram_buckets"]

    byte_hist_cols = [f"byte_hist_{i:03d}" for i in range(256)]
    ngram_hist_cols = [f"ngram_hist_{i:03d}" for i in range(ngram_buckets)]

    scalar_rows = []
    byte_hist_rows = np.zeros((len(payloads_df), 256), dtype=np.float64)
    ngram_hist_rows = np.zeros((len(payloads_df), ngram_buckets), dtype=np.float64)

    for i, payload in enumerate(payloads_df["payload"]):
        feats = extract_features(payload, config)
        byte_hist_rows[i] = feats["byte_hist"]
        ngram_hist_rows[i] = feats["ngram_hist"]
        scalar_rows.append({
            "entropy": feats["entropy"],
            "printable_ratio": feats["printable_ratio"],
            "length_bucket": feats["length_bucket"],
            "payload_len": feats["payload_len"],
        })

    scalar_df = pd.DataFrame(scalar_rows)
    byte_hist_df = pd.DataFrame(byte_hist_rows, columns=byte_hist_cols)
    ngram_hist_df = pd.DataFrame(ngram_hist_rows, columns=ngram_hist_cols)

    # Carry through all original metadata columns except the raw payload bytes
    # (superseded by the extracted features) and payload_len (recomputed above
    # from the same source as a sanity cross-check -- keeping both would be a
    # duplicate column, not new information).
    metadata_df = payloads_df.drop(columns=["payload", "payload_len"], errors="ignore").reset_index(drop=True)

    return pd.concat([metadata_df, scalar_df, byte_hist_df, ngram_hist_df], axis=1)


class RunningStats:
    """Tracks min/max/mean (overall and per-service) across batches without
    ever holding the full column of values in memory."""

    def __init__(self):
        self.n = 0
        self.entropy_min = float("inf")
        self.entropy_max = float("-inf")
        self.entropy_sum = 0.0
        self.printable_min = float("inf")
        self.printable_max = float("-inf")
        self.printable_sum = 0.0
        self.service_entropy_sum: dict[str, float] = {}
        self.service_count: dict[str, int] = {}

    def update(self, features_df: pd.DataFrame):
        self.n += len(features_df)
        self.entropy_min = min(self.entropy_min, features_df["entropy"].min())
        self.entropy_max = max(self.entropy_max, features_df["entropy"].max())
        self.entropy_sum += features_df["entropy"].sum()
        self.printable_min = min(self.printable_min, features_df["printable_ratio"].min())
        self.printable_max = max(self.printable_max, features_df["printable_ratio"].max())
        self.printable_sum += features_df["printable_ratio"].sum()

        if "service" in features_df.columns:
            grouped = features_df.groupby("service")["entropy"].agg(["sum", "count"])
            for service, row in grouped.iterrows():
                self.service_entropy_sum[service] = self.service_entropy_sum.get(service, 0.0) + row["sum"]
                self.service_count[service] = self.service_count.get(service, 0) + int(row["count"])

    def print_summary(self):
        if self.n == 0:
            return
        print("\n-- Quick sanity summary --")
        print(f"  Entropy:          min={self.entropy_min:.2f}, "
              f"max={self.entropy_max:.2f}, mean={self.entropy_sum / self.n:.2f}")
        print(f"  Printable ratio:  min={self.printable_min:.2f}, "
              f"max={self.printable_max:.2f}, mean={self.printable_sum / self.n:.2f}")
        if self.service_count:
            print("\n  By service (mean entropy):")
            for service in sorted(self.service_count):
                mean_entropy = self.service_entropy_sum[service] / self.service_count[service]
                print(f"    {service:<10} {mean_entropy:.3f}  (n={self.service_count[service]:,})")


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 3 features from payload Parquet.")
    parser.add_argument("--input", required=True, help="Input Parquet file from scripts/parse_pcaps.py")
    parser.add_argument("--output", required=True, help="Output Parquet path for extracted features")
    parser.add_argument("--batch-size", type=int, default=200_000,
                         help="Rows processed per batch (default: 200000). Lower this if you hit "
                              "memory errors; raise it if you have RAM to spare and want more speed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[FAIL] {input_path} not found. Run scripts/parse_pcaps.py first (Phase 2).")
        sys.exit(1)

    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Input has {total_rows:,} payload records. Processing in batches of {args.batch_size:,}.")

    if total_rows == 0:
        print("[WARN] Input file has 0 rows -- nothing to extract. Exiting.")
        sys.exit(0)

    config = load_config()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = RunningStats()
    writer = None
    rows_done = 0
    t0 = time.time()

    try:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            payloads_df = batch.to_pandas()
            features_df = build_feature_dataframe(payloads_df, config)

            table = pa.Table.from_pandas(features_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)

            stats.update(features_df)
            rows_done += len(payloads_df)

            elapsed = time.time() - t0
            rate = rows_done / elapsed if elapsed > 0 else 0
            eta = (total_rows - rows_done) / rate if rate > 0 else float("nan")
            print(f"  ... {rows_done:,}/{total_rows:,} rows "
                  f"({100 * rows_done / total_rows:.1f}%, {rate:,.0f} rows/s, "
                  f"ETA {eta:,.0f}s)")
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    print(f"\n[ OK ] Extracted features for {rows_done:,} payloads in {elapsed:.1f}s")
    print(f"[ OK ] Saved -> {output_path}")

    stats.print_summary()


if __name__ == "__main__":
    main()
