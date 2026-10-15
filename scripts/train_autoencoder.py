#!/usr/bin/env python3
"""
Phase 5 — Train Autoencoder Anomaly Detector on Normal Traffic.

Usage:
    python scripts/train_autoencoder.py \\
        --input data/processed/features_packet.parquet \\
        --output models/autoencoder.pt \\
        --epochs 3 --batch-size 2048 --percentile 99

Trains in streaming chunks over Parquet data so multi-million-row datasets
never exhaust system memory.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.autoencoder import (
    PayloadAutoencoder,
    get_feature_columns,
    save_autoencoder,
)


def train_streaming(
    input_path: Path,
    output_path: Path,
    epochs: int = 3,
    chunk_size: int = 200_000,
    train_batch_size: int = 2048,
    bottleneck_dim: int = 16,
    lr: float = 0.001,
    percentile: float = 99.0,
    max_train_rows: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pf = pq.ParquetFile(input_path)
    feat_cols = get_feature_columns(pf.schema_arrow)
    input_dim = len(feat_cols)
    print(f"Features: {input_dim} columns ({feat_cols[0]} ... {feat_cols[-1]})")

    model = PayloadAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_rows = pf.metadata.num_rows
    if max_train_rows:
        total_rows = min(total_rows, max_train_rows)
    print(f"Training on up to {total_rows:,} rows across {epochs} epochs...")

    start_time = time.time()

    # Pass 1: Training
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches_count = 0
        rows_seen = 0

        for batch in pf.iter_batches(batch_size=chunk_size, columns=feat_cols):
            chunk_df = batch.to_pandas()
            if max_train_rows and (rows_seen + len(chunk_df) > max_train_rows):
                chunk_df = chunk_df.iloc[: max_train_rows - rows_seen]

            arr = chunk_df.to_numpy(dtype=np.float32)
            n_chunk = len(arr)
            rows_seen += n_chunk

            model.train()
            # Mini-batch gradient descent within chunk
            for i in range(0, n_chunk, train_batch_size):
                sub = arr[i : i + train_batch_size]
                if len(sub) < 2:
                    continue
                x_t = torch.tensor(sub, device=device)
                optimizer.zero_grad()
                pred = model(x_t)
                loss = criterion(pred, x_t)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(sub)
                batches_count += len(sub)

            if max_train_rows and rows_seen >= max_train_rows:
                break

        avg_loss = epoch_loss / max(batches_count, 1)
        print(f"  Epoch {epoch}/{epochs} - Avg MSE Loss: {avg_loss:.6f} ({rows_seen:,} rows)")

    # Pass 2: Calibrate anomaly threshold on training data (percentile of MSE errors)
    print(f"\nCalibrating {percentile}th-percentile anomaly threshold on training data...")
    errors_sample = []
    # Collect sample errors from training chunks to estimate threshold reliably
    rows_seen = 0
    model.eval()
    with torch.no_grad():
        for batch in pf.iter_batches(batch_size=chunk_size, columns=feat_cols):
            chunk_df = batch.to_pandas()
            if max_train_rows and (rows_seen + len(chunk_df) > max_train_rows):
                chunk_df = chunk_df.iloc[: max_train_rows - rows_seen]
            arr = chunk_df.to_numpy(dtype=np.float32)
            rows_seen += len(arr)

            # Subsample 10% from each chunk to fit comfortably in memory for percentile
            subsample = arr[::10]
            for i in range(0, len(subsample), train_batch_size):
                x_t = torch.tensor(subsample[i : i + train_batch_size], device=device)
                rec = model(x_t)
                mse = torch.mean((x_t - rec) ** 2, dim=1).cpu().numpy()
                errors_sample.extend(mse)

            if max_train_rows and rows_seen >= max_train_rows:
                break

    errors_arr = np.array(errors_sample, dtype=np.float32)
    threshold = float(np.percentile(errors_arr, percentile))
    elapsed = time.time() - start_time

    print(f"Learned threshold at {percentile}th percentile: {threshold:.6f}")
    print(f"Sample MSE error min/median/max: {np.min(errors_arr):.6f} / {np.median(errors_arr):.6f} / {np.max(errors_arr):.6f}")

    meta = {
        "epochs": epochs,
        "lr": lr,
        "bottleneck_dim": bottleneck_dim,
        "percentile": percentile,
        "rows_trained": rows_seen,
        "training_time_s": round(elapsed, 2),
    }

    save_autoencoder(model, threshold, output_path, extra_meta=meta)
    print(f"[ OK ] Saved trained autoencoder to {output_path} (time: {elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Input parquet with feature columns (e.g. Monday features)")
    parser.add_argument("--output", required=True, type=Path, help="Path to save trained autoencoder .pt model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size for Adam updates")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Streaming Parquet chunk size")
    parser.add_argument("--bottleneck", type=int, default=16, help="Bottleneck latent dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--percentile", type=float, default=99.0, help="Anomaly alert threshold percentile")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap training rows (optional)")
    args = parser.parse_args()

    train_streaming(
        input_path=args.input,
        output_path=args.output,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        train_batch_size=args.batch_size,
        bottleneck_dim=args.bottleneck,
        lr=args.lr,
        percentile=args.percentile,
        max_train_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
