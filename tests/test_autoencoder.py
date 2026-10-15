"""
Unit tests for models/autoencoder.py and Autoencoder training/scoring pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from models.autoencoder import (
    PayloadAutoencoder,
    compute_reconstruction_error,
    get_feature_columns,
    load_autoencoder,
    save_autoencoder,
)


def _synthetic_dataframe(n_rows: int = 100):
    data = {}
    for i in range(256):
        data[f"byte_hist_{i:03d}"] = np.random.uniform(0, 1, n_rows).astype(np.float32)
    data["entropy"] = np.random.uniform(0, 8, n_rows).astype(np.float32)
    data["printable_ratio"] = np.random.uniform(0, 1, n_rows).astype(np.float32)
    data["src_ip"] = ["10.0.0.1"] * n_rows
    data["dst_ip"] = ["10.0.0.2"] * n_rows
    data["dst_port"] = [80] * n_rows
    return pd.DataFrame(data)


def test_get_feature_columns_returns_258():
    df = _synthetic_dataframe(10)
    cols = get_feature_columns(df)
    assert len(cols) == 258
    assert cols[0] == "byte_hist_000"
    assert cols[255] == "byte_hist_255"
    assert "entropy" in cols
    assert "printable_ratio" in cols


def test_autoencoder_forward_shape():
    model = PayloadAutoencoder(input_dim=258, bottleneck_dim=16)
    x = torch.randn(8, 258)
    out = model(x)
    assert out.shape == (8, 258)


def test_autoencoder_reconstruction_errors_vector():
    model = PayloadAutoencoder(input_dim=258, bottleneck_dim=16)
    x = torch.randn(10, 258)
    err = model.reconstruction_errors(x)
    assert len(err) == 10
    assert (err >= 0).all()


def test_compute_reconstruction_error_numpy():
    model = PayloadAutoencoder(input_dim=258, bottleneck_dim=16)
    arr = np.random.randn(25, 258).astype(np.float32)
    err = compute_reconstruction_error(model, arr, batch_size=10)
    assert len(err) == 25
    assert np.all(err >= 0)


def test_save_and_load_autoencoder_roundtrip(tmp_path):
    model = PayloadAutoencoder(input_dim=258, bottleneck_dim=16)
    out_file = tmp_path / "test_ae.pt"
    save_autoencoder(model, threshold=0.1234, path=out_file, extra_meta={"test": 1})

    loaded_model, threshold, meta = load_autoencoder(out_file)
    assert abs(threshold - 0.1234) < 1e-6
    assert meta["test"] == 1

    x = torch.randn(5, 258)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(x).numpy(),
            loaded_model(x).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_autoencoder_overfits_constant_pattern():
    """An autoencoder trained on a constant vector should have near-zero reconstruction error on that vector."""
    torch.manual_seed(42)
    model = PayloadAutoencoder(input_dim=258, bottleneck_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    target = torch.ones(16, 258)
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(target)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_err = model.reconstruction_errors(target)
        # Error on training target should be very small
        assert np.mean(final_err) < 0.1
