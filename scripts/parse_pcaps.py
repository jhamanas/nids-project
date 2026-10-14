#!/usr/bin/env python3
"""
Phase 2 — Run payload extraction over a directory of PCAP files.

Usage:
    # Extract from CICIDS2017's PCAPs, packet-level, all files:
    python scripts/parse_pcaps.py --input data/raw/cicids2017/PCAPs --mode packet

    # Session-level (TCP reassembly), single file, cap at 50k packets (quick test):
    python scripts/parse_pcaps.py --input data/raw/cicids2017/PCAPs/Monday-WorkingHours.pcap \\
        --mode session --limit 50000

    # Try it right now against the synthetic fixture, no real PCAPs needed:
    python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode both

Output:
    data/processed/payloads_packet.parquet   (if --mode packet or both)
    data/processed/payloads_session.parquet  (if --mode session or both)

Each row's `payload` column holds raw bytes (Parquet's native binary type —
no base64 encoding needed, and Phase 3 feature extraction can read it back
as bytes directly via pandas/pyarrow).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pcap_parser import extract_packet_payloads, extract_session_payloads  # noqa: E402
from preprocessing.config_loader import load_config  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def find_pcap_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.pcap")) + sorted(input_path.glob("*.pcapng"))
        return files
    return []


def run_extraction(pcap_files: list[Path], config: dict, mode: str, limit: int,
                    progress_every: int = 500_000):
    packet_records, session_records = [], []

    for pcap_path in pcap_files:
        print(f"[....] Processing {pcap_path.name} ...")
        t0 = time.time()

        if mode in ("packet", "both"):
            recs = extract_packet_payloads(pcap_path, config, limit=limit, progress_every=progress_every)
            packet_records.extend(recs)
            print(f"       packet-level: {len(recs):,} payload records")

        if mode in ("session", "both"):
            recs = extract_session_payloads(pcap_path, config, limit=limit, progress_every=progress_every)
            session_records.extend(recs)
            print(f"       session-level: {len(recs):,} payload records")

        print(f"[ OK ] {pcap_path.name} done in {time.time() - t0:.1f}s")

    return packet_records, session_records


def save_records(records: list[dict], out_path: Path, label: str):
    if not records:
        print(f"[WARN] No {label} records extracted — nothing to save.")
        return
    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[ OK ] Saved {len(df):,} {label} records -> {out_path}")

    print(f"\n-- {label} summary --")
    print(f"  Total records: {len(df):,}")
    print(f"  Service breakdown:\n{df['service'].value_counts().to_string()}")
    print(f"  Payload length: min={df['payload_len'].min()}, "
          f"max={df['payload_len'].max()}, mean={df['payload_len'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Extract payloads from PCAP files.")
    parser.add_argument("--input", required=True, help="PCAP file or directory of PCAP files")
    parser.add_argument("--mode", choices=["packet", "session", "both"], default="packet")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max packets to read per PCAP file (for quick testing on large files)")
    parser.add_argument("--progress-every", type=int, default=500_000,
                         help="Print a status line every N packets read (0 to disable). Default: 500000")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[FAIL] {input_path} does not exist.")
        sys.exit(1)

    pcap_files = find_pcap_files(input_path)
    if not pcap_files:
        print(f"[FAIL] No .pcap/.pcapng files found at {input_path}.")
        sys.exit(1)

    print(f"Found {len(pcap_files)} PCAP file(s) to process.\n")

    config = load_config()
    packet_records, session_records = run_extraction(
        pcap_files, config, args.mode, args.limit, progress_every=args.progress_every or None
    )

    print()
    if args.mode in ("packet", "both"):
        save_records(packet_records, PROCESSED_DIR / "payloads_packet.parquet", "packet-level")
    if args.mode in ("session", "both"):
        save_records(session_records, PROCESSED_DIR / "payloads_session.parquet", "session-level")


if __name__ == "__main__":
    main()
