#!/usr/bin/env python3
"""
Phase 1 — Verify CICIDS2017 CSV files are present and readable.

CICIDS2017 can't be downloaded automatically (registration-gated) — see
data/raw/cicids2017/README.md for manual download steps. Run this script
after placing the CSVs to confirm everything is in order before moving on
to exploration/Phase 2.

Usage:
    python scripts/verify_cicids2017.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"

# Loose match patterns — CICIDS2017 zip mirrors vary slightly in exact
# filenames/capitalization, so we match by keyword rather than exact name.
EXPECTED_DAYS = {
    "Monday": ["monday"],
    "Tuesday": ["tuesday"],
    "Wednesday": ["wednesday"],
    "Thursday (Web Attacks)": ["thursday", "webattack"],
    "Thursday (Infiltration)": ["thursday", "infilteration"],
    "Friday (Morning)": ["friday", "morning"],
    "Friday (PortScan)": ["friday", "portscan"],
    "Friday (DDoS)": ["friday", "ddos"],
}


def find_csvs():
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


def main():
    print("=" * 60)
    print("CICIDS2017 — Verification")
    print(f"Looking in: {DATA_DIR}")
    print("=" * 60)
    print()

    csvs = find_csvs()
    if not csvs:
        print("[FAIL] No CSV files found.")
        print(f"       Place CICIDS2017 CSVs in: {DATA_DIR}")
        print("       See data/raw/cicids2017/README.md for download steps.")
        sys.exit(1)

    print(f"Found {len(csvs)} CSV file(s):\n")

    try:
        import pandas as pd
    except ImportError:
        print("[FAIL] pandas is not installed — run: pip install -r requirements.txt")
        sys.exit(1)

    total_rows = 0
    all_readable = True
    for csv_path in csvs:
        try:
            # Read only first chunk for a fast row-count + column sanity check
            df = pd.read_csv(csv_path, low_memory=False)
            n_rows, n_cols = df.shape
            total_rows += n_rows

            # CICIDS2017's label column is usually literally " Label" (leading space)
            label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
            if label_col:
                n_classes = df[label_col].nunique()
                print(f"[ OK ] {csv_path.name:<55} {n_rows:>8,} rows, {n_cols} cols, "
                      f"{n_classes} label(s)")
            else:
                print(f"[WARN] {csv_path.name:<55} {n_rows:>8,} rows, {n_cols} cols, "
                      f"NO 'Label' column found — check this file")
        except Exception as e:
            print(f"[FAIL] {csv_path.name}: could not read — {e}")
            all_readable = False

    print()
    print("-" * 60)

    # Check which expected days are covered, by loose filename matching
    found_days = []
    filenames_lower = " ".join(p.name.lower() for p in csvs)
    for day, keywords in EXPECTED_DAYS.items():
        if all(kw in filenames_lower for kw in keywords):
            found_days.append(day)

    missing_days = [d for d in EXPECTED_DAYS if d not in found_days]
    print(f"Days covered: {len(found_days)}/{len(EXPECTED_DAYS)}")
    if missing_days:
        print("Missing (by filename heuristic — double check manually if this "
              "looks wrong, mirrors name files inconsistently):")
        for d in missing_days:
            print(f"   - {d}")

    print()
    print(f"Total rows across all files: {total_rows:,}")

    pcap_dir = DATA_DIR / "PCAPs"
    if pcap_dir.exists() and any(pcap_dir.glob("*.pcap")):
        n_pcaps = len(list(pcap_dir.glob("*.pcap")))
        print(f"[ OK ] PCAPs/ folder found with {n_pcaps} .pcap file(s) — ready for Phase 2")
    else:
        print("[WARN] No PCAPs/ folder with .pcap files found. The CSVs are enough for "
              "Phase 1 exploration, but Phase 2 (payload extraction) needs the raw PCAPs.")

    print()
    if all_readable and csvs:
        print("CICIDS2017 CSVs verified. Next: python scripts/explore_dataset.py --dataset cicids2017")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
