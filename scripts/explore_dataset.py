#!/usr/bin/env python3
"""
Phase 1 — Data exploration.

Loads NSL-KDD and/or CICIDS2017, and reports the things you need to know
before designing Phase 2-3 (parsing/feature extraction):
  - shape, label/attack-category distribution, class imbalance
  - missing values / infinities (CICIDS2017's CSVs are known to have some)
  - basic numeric-feature statistics
  - saves a couple of summary plots to data/processed/

Usage:
    python scripts/explore_dataset.py --dataset nsl-kdd
    python scripts/explore_dataset.py --dataset cicids2017
    python scripts/explore_dataset.py --dataset both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.nsl_kdd_columns import NSL_KDD_COLUMNS, label_to_category  # noqa: E402

NSL_KDD_DIR = PROJECT_ROOT / "data" / "raw" / "nsl-kdd"
CICIDS_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def explore_nsl_kdd():
    _section("NSL-KDD — Exploration")

    train_path = NSL_KDD_DIR / "KDDTrain+.txt"
    test_path = NSL_KDD_DIR / "KDDTest+.txt"

    if not train_path.exists():
        print(f"[FAIL] {train_path} not found. Run: python scripts/download_nsl_kdd.py")
        return

    train = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test = pd.read_csv(test_path, names=NSL_KDD_COLUMNS) if test_path.exists() else None

    print(f"\nTrain set: {train.shape[0]:,} rows, {train.shape[1]} columns")
    if test is not None:
        print(f"Test set:  {test.shape[0]:,} rows, {test.shape[1]} columns")

    # Attack category breakdown (Normal / DoS / Probe / R2L / U2R)
    train["category"] = train["label"].apply(label_to_category)
    print("\n-- Attack category distribution (train) --")
    cat_counts = train["category"].value_counts()
    for cat, count in cat_counts.items():
        pct = 100 * count / len(train)
        print(f"  {cat:<10} {count:>7,}  ({pct:5.2f}%)")

    print("\n-- Top 10 specific attack labels (train) --")
    label_counts = train["label"].value_counts().head(10)
    for label, count in label_counts.items():
        print(f"  {label:<20} {count:>7,}")

    # Missing values
    n_missing = train.isnull().sum().sum()
    print(f"\nMissing values in train set: {n_missing}")

    # Protocol/service breakdown — relevant later for per-service profiling (Phase 4)
    print("\n-- protocol_type distribution (train) --")
    print(train["protocol_type"].value_counts().to_string())

    # Save a plot
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        cat_counts.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_title("NSL-KDD (KDDTrain+) — Attack Category Distribution")
        ax.set_ylabel("Record count")
        ax.set_xlabel("Category")
        plt.xticks(rotation=0)
        plt.tight_layout()
        out_path = PROCESSED_DIR / "nsl_kdd_category_distribution.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n[ OK ] Saved plot: {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save plot: {e}")


def explore_cicids2017():
    _section("CICIDS2017 — Exploration")

    if not CICIDS_DIR.exists():
        print(f"[FAIL] {CICIDS_DIR} not found.")
        return

    csvs = sorted(CICIDS_DIR.glob("*.csv"))
    if not csvs:
        print(f"[FAIL] No CSVs found in {CICIDS_DIR}.")
        print("       See data/raw/cicids2017/README.md for manual download steps,")
        print("       then run: python scripts/verify_cicids2017.py")
        return

    all_label_counts = {}
    total_rows = 0

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"[FAIL] Could not read {csv_path.name}: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]  # CICIDS2017 headers have stray spaces
        total_rows += len(df)

        if "Label" not in df.columns:
            print(f"[WARN] {csv_path.name}: no 'Label' column found, skipping label breakdown")
            continue

        counts = df["Label"].value_counts()
        for label, count in counts.items():
            all_label_counts[label] = all_label_counts.get(label, 0) + count

        # Check for inf/NaN — a known CICIDS2017 CSV quirk (some flow features
        # divide by zero for very short flows)
        numeric_df = df.select_dtypes(include=[np.number])
        n_inf = np.isinf(numeric_df).sum().sum()
        n_nan = df.isnull().sum().sum()
        print(f"[ OK ] {csv_path.name:<55} {len(df):>8,} rows  "
              f"(missing: {n_nan}, inf: {n_inf})")

    print(f"\nTotal rows across all CICIDS2017 CSVs: {total_rows:,}")

    print("\n-- Combined label distribution (all days) --")
    combined = pd.Series(all_label_counts).sort_values(ascending=False)
    for label, count in combined.items():
        pct = 100 * count / total_rows if total_rows else 0
        print(f"  {label:<25} {count:>9,}  ({pct:5.2f}%)")

    # Save a plot
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        combined.plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title("CICIDS2017 — Label Distribution (All Days Combined)")
        ax.set_xlabel("Record count")
        plt.tight_layout()
        out_path = PROCESSED_DIR / "cicids2017_label_distribution.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n[ OK ] Saved plot: {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save plot: {e}")

    pcap_dir = CICIDS_DIR / "PCAPs"
    if not (pcap_dir.exists() and any(pcap_dir.glob("*.pcap"))):
        print("\n[NOTE] No PCAPs found yet. The CSVs give you flow-level exploration now,"
              "\n       but Phase 2 (payload byte extraction) needs the raw PCAP files —"
              "\n       see data/raw/cicids2017/README.md.")


def main():
    parser = argparse.ArgumentParser(description="Explore NSL-KDD and/or CICIDS2017.")
    parser.add_argument(
        "--dataset", choices=["nsl-kdd", "cicids2017", "both"], default="both",
        help="Which dataset to explore (default: both)"
    )
    args = parser.parse_args()

    if args.dataset in ("nsl-kdd", "both"):
        explore_nsl_kdd()
    if args.dataset in ("cicids2017", "both"):
        explore_cicids2017()

    print()
    print("Done. Plots (if generated) are in data/processed/.")


if __name__ == "__main__":
    main()
