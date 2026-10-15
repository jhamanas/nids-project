"""
Phase 5 — Autoencoder-based Anomaly Detector in PyTorch.

Architecture:
    Encoder: Input (258 dims: 256 byte_hist_* + entropy + printable_ratio)
             -> Dense(128) -> ReLU -> BatchNorm
             -> Dense(64)  -> ReLU -> BatchNorm
             -> Bottleneck(16) -> ReLU
    Decoder: Bottleneck(16)
             -> Dense(64)  -> ReLU -> BatchNorm
             -> Dense(128) -> ReLU -> BatchNorm
             -> Output(258) -> Sigmoid / Linear

Reconstruction Error:
    MSE(x, reconstructed(x)) computed per-sample.
    Trained on normal (benign Monday) traffic.
    Samples exceeding a learned percentile threshold of normal reconstruction
    error are flagged as anomalous.
"""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def get_feature_columns(schema_or_df) -> list[str]:
    """Returns 258 feature columns: 256 byte_hist_* + entropy + printable_ratio."""
    if hasattr(schema_or_df, "columns"):
        names = list(schema_or_df.columns)
    elif hasattr(schema_or_df, "names"):
        names = list(schema_or_df.names)
    elif hasattr(schema_or_df, "schema_arrow"):
        names = list(schema_or_df.schema_arrow.names)
    else:
        names = list(schema_or_df)

    byte_cols = sorted([c for c in names if c.startswith("byte_hist_")])
    if len(byte_cols) != 256:
        raise ValueError(f"Expected 256 byte_hist_* columns, found {len(byte_cols)}")

    extra_cols = [c for c in ["entropy", "printable_ratio"] if c in names]
    return byte_cols + extra_cols


class PayloadAutoencoder(nn.Module):
    """Deep autoencoder for compressed representation of payload feature vectors."""

    def __init__(self, input_dim: int = 258, bottleneck_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_errors(self, x: torch.Tensor) -> np.ndarray:
        """Compute MSE per sample."""
        self.eval()
        with torch.no_grad():
            reconstructed = self(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse.cpu().numpy()


def compute_reconstruction_error(
    model: PayloadAutoencoder,
    features_array: np.ndarray,
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute per-row reconstruction error in batches."""
    model.eval()
    model.to(device)
    errors = []
    n = len(features_array)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = features_array[start : start + batch_size]
            x_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            err = model.reconstruction_errors(x_tensor)
            errors.append(err)
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


def save_autoencoder(model: PayloadAutoencoder, threshold: float, path: Path, extra_meta: dict = None):
    """Save PyTorch model weights and learned anomaly threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "bottleneck_dim": model.bottleneck_dim,
        "threshold": float(threshold),
        "extra_meta": extra_meta or {},
    }
    torch.save(payload, path)


def load_autoencoder(path: Path, device: torch.device = torch.device("cpu")) -> tuple[PayloadAutoencoder, float, dict]:
    """Load model, threshold, and metadata from disk."""
    payload = torch.load(path, map_location=device)
    model = PayloadAutoencoder(
        input_dim=payload["input_dim"],
        bottleneck_dim=payload["bottleneck_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload["threshold"], payload.get("extra_meta", {})
