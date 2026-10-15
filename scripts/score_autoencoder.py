#!/usr/bin/env python3
"""
Phase 5 — Score Payloads Using Trained Autoencoder.

Usage:
    python scripts/score_autoencoder.py \\
        --model models/autoencoder.pt \\
        --input data/processed/features_packet_friday.parquet \\
        --output data/processed/scored_friday_autoencoder.parquet

Streams through the input file, computes reconstruction error (MSE) per row,
compares it against the learned percentile threshold, and writes out:
  - autoencoder_score (MSE error)
  - autoencoder_anomaly (bool)
along with standard identification columns.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.autoencoder import (
    compute_reconstruction_error,
    get_feature_columns,
    load_autoencoder,
)


def score_streaming(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    chunk_size: int = 200_000,
    batch_size: int = 2048,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained autoencoder from {model_path} on {device}...")
    model, threshold, meta = load_autoencoder(model_path, device=device)
    print(f"Loaded model (input_dim={model.input_dim}, bottleneck={model.bottleneck_dim}, threshold={threshold:.6f})")

    pf = pq.ParquetFile(input_path)
    feat_cols = get_feature_columns(pf.schema_arrow)

    # Columns to pass through into scored file
    id_cols = [
        c
        for c in [
            "pcap_file",
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "service",
            "entropy",
            "printable_ratio",
            "length_bucket",
            "payload_len",
        ]
        if c in pf.schema_arrow.names
    ]

    cols_to_read = list(set(id_cols + feat_cols))
    total_rows = pf.metadata.num_rows
    print(f"Scoring {total_rows:,} rows from {input_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    rows_scored = 0
    flagged_count = 0
    start_time = time.time()

    for batch in pf.iter_batches(batch_size=chunk_size, columns=cols_to_read):
        chunk_df = batch.to_pandas()
        feat_arr = chunk_df[feat_cols].to_numpy(dtype=np.float32)

        errors = compute_reconstruction_error(model, feat_arr, batch_size=batch_size, device=device)
        is_anomaly = errors > threshold

        flagged_count += int(is_anomaly.sum())
        rows_scored += len(chunk_df)

        out_df = chunk_df[id_cols].copy()
        out_df["autoencoder_score"] = errors
        out_df["autoencoder_anomaly"] = is_anomaly
        # Alias for spot_check_labels.py compatibility
        out_df["payl_score"] = errors
        out_df["payl_anomaly"] = is_anomaly

        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

        pct = 100 * rows_scored / total_rows
        print(f"  {rows_scored:,}/{total_rows:,} rows scored ({pct:.1f}%)")

    if writer is not None:
        writer.close()

    elapsed = time.time() - start_time
    print(f"\n[ OK ] Saved scored autoencoder results to {output_path}")
    print(f"Total Scored:      {rows_scored:,}")
    print(f"Flagged Anomalous: {flagged_count:,} ({100 * flagged_count / max(rows_scored, 1):.2f}%)")
    print(f"Elapsed Time:      {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Path to trained autoencoder .pt")
    parser.add_argument("--input", required=True, type=Path, help="Input parquet with extracted features")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path for scored features")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    score_streaming(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
