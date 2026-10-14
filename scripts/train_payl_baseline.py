#!/usr/bin/env python3
"""
Phase 4 — Train the PAYL baseline from normal (benign-only) traffic features.

IMPORTANT: the input features file must come from traffic known to be
normal/benign only. For CICIDS2017, that's Monday's PCAP (the entire day
is benign -- see data/raw/cicids2017/README.md). Training on a mixed day
would bake attack traffic into the "normal" profile, defeating the whole
point of anomaly detection.

Usage:
    python scripts/train_payl_baseline.py \\
        --input data/processed/features_packet_monday.parquet \\
        --output models/payl_baseline.pkl

    # Try it now against the fixture output (Phase 2/3's tiny sample --
    # not enough data for a real profile, but exercises the pipeline):
    python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode packet
    python scripts/extract_features.py --input data/processed/payloads_packet.parquet \\
        --output data/processed/features_packet.parquet
    python scripts/train_payl_baseline.py \\
        --input data/processed/features_packet.parquet \\
        --output models/payl_baseline_test.pkl
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.payl_baseline import build_profiles_streaming, save_profiles  # noqa: E402
from preprocessing.config_loader import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the PAYL baseline profile.")
    parser.add_argument("--input", required=True, help="Features Parquet from scripts/extract_features.py")
    parser.add_argument("--output", required=True, help="Output path for the trained profile (.pkl)")
    parser.add_argument("--batch-size", type=int, default=200_000,
                         help="Rows processed per batch (default: 200000). The file is read twice "
                              "(two passes) regardless of size, but never held fully in memory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[FAIL] {input_path} not found. Run scripts/extract_features.py first (Phase 3).")
        sys.exit(1)

    total_rows = pq.ParquetFile(input_path).metadata.num_rows
    print(f"Input has {total_rows:,} feature records. Training in two streaming passes "
          f"(batch size {args.batch_size:,}) -- this file is never fully loaded into memory.")

    if total_rows == 0:
        print("[WARN] Input file has 0 rows -- nothing to train on. Exiting.")
        sys.exit(0)

    print("\n[NOTE] This assumes every row here is NORMAL/BENIGN traffic.")
    print("       If this file mixes in attack traffic, the profile will be")
    print("       contaminated. See the script's docstring for details.\n")

    config = load_config()
    profiles = build_profiles_streaming(input_path, config, batch_size=args.batch_size)

    n_ok = sum(1 for p in profiles.values() if p["status"] == "ok")
    n_insufficient = sum(1 for p in profiles.values() if p["status"] == "insufficient_data")

    print(f"Built {len(profiles)} (service, length_bucket) groups:")
    print(f"  {n_ok} with usable profiles")
    print(f"  {n_insufficient} marked insufficient_data (fewer than "
          f"{config['payl_baseline']['min_samples_per_profile']} samples)")

    print("\n-- Per-service breakdown --")
    by_service = {}
    for (service, length_bucket), profile in profiles.items():
        by_service.setdefault(service, {"ok": 0, "insufficient": 0, "total_samples": 0})
        if profile["status"] == "ok":
            by_service[service]["ok"] += 1
        else:
            by_service[service]["insufficient"] += 1
        by_service[service]["total_samples"] += profile["n_samples"]

    for service, stats in sorted(by_service.items()):
        print(f"  {service:<10} {stats['ok']:>3} usable profiles, "
              f"{stats['insufficient']:>3} insufficient, "
              f"{stats['total_samples']:>8,} total samples")

    save_profiles(profiles, config, output_path)
    print(f"\n[ OK ] Saved trained profile -> {output_path}")

    if n_ok == 0:
        print("\n[WARN] No usable profiles were built at all -- either you have very")
        print("       little data, or it's spread across too many (service, length_bucket)")
        print("       groups. Consider lowering min_samples_per_profile in config.yaml")
        print("       or feeding in more training data.")


if __name__ == "__main__":
    main()
