"""
tests/test_pipeline_e2e.py

Phase 8 End-to-End Tests (Tasks E1, E4):
Verifies that:
1. scripts/run_pipeline.py executes without error and generates valid Parquet files.
2. scripts/demo_score_pcap.py runs without error and outputs alert details.
"""

from pathlib import Path
import pytest
import pyarrow.parquet as pq

from scripts.run_pipeline import run_pipeline
from scripts.demo_score_pcap import demo_score_pcap

ROOT = Path(__file__).resolve().parents[1]


def test_run_pipeline_e2e(tmp_path):
    pcap = ROOT / "tests" / "fixtures" / "sample.pcap"
    payl_profiles = ROOT / "models" / "payl_baseline.pkl"
    ae_model = ROOT / "models" / "autoencoder.pt"
    out_dir = tmp_path / "pipeline_run"

    run_pipeline(
        pcap_path=pcap,
        payl_profiles_path=payl_profiles,
        ae_model_path=ae_model,
        output_dir=out_dir,
    )

    feat_parquet = out_dir / "extracted_features.parquet"
    scored_parquet = out_dir / "scored_packets.parquet"

    assert feat_parquet.exists()
    assert scored_parquet.exists()

    df_scored = pq.read_table(scored_parquet).to_pandas()
    assert len(df_scored) == 6
    assert "payl_anomaly" in df_scored.columns
    assert "autoencoder_anomaly" in df_scored.columns


def test_demo_score_pcap_e2e(capsys):
    pcap = ROOT / "tests" / "fixtures" / "sample.pcap"
    payl_profiles = ROOT / "models" / "payl_baseline.pkl"
    ae_model = ROOT / "models" / "autoencoder.pt"

    demo_score_pcap(
        pcap_path=pcap,
        payl_profiles_path=payl_profiles,
        ae_model_path=ae_model,
        show_all=True,
    )

    captured = capsys.readouterr()
    assert "INSPECTION RESULTS & ALERTS" in captured.out
    assert "Packet #001" in captured.out
