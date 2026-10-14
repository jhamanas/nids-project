#!/usr/bin/env python3
"""
Phase 1 — Download the NSL-KDD dataset.

NSL-KDD's original host (unb.ca) has taken the direct download links down; the
files are only recoverable via the Wayback Machine. This script pulls them from
a widely-used, stable GitHub mirror instead:
    https://github.com/jmnwong/NSL-KDD-Dataset

Downloads (into data/raw/nsl-kdd/):
    KDDTrain+.txt              - full training set (labels + difficulty level)
    KDDTrain+_20Percent.txt    - 20% subset of the training set (faster iteration)
    KDDTest+.txt                - full test set
    KDDTest-21.txt               - harder test subset (excludes easiest records)

Usage:
    python scripts/download_nsl_kdd.py
    python scripts/download_nsl_kdd.py --force     # re-download even if files exist
"""

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nsl-kdd"

BASE_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"

FILES = [
    "KDDTrain+.txt",
    "KDDTrain+_20Percent.txt",
    "KDDTest+.txt",
    "KDDTest-21.txt",
]

# Rough expected byte-size floors, just to catch a truncated/HTML-error download
# (e.g. GitHub returning a 404 page instead of the file). Not exact — just a
# sanity floor well below the real file sizes.
MIN_EXPECTED_BYTES = {
    "KDDTrain+.txt": 10_000_000,
    "KDDTrain+_20Percent.txt": 2_000_000,
    "KDDTest+.txt": 2_000_000,
    "KDDTest-21.txt": 1_500_000,
}


def download_file(filename: str, force: bool) -> bool:
    url = BASE_URL + filename.replace("+", "%2B")
    dest = OUTPUT_DIR / filename

    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"[SKIP] {filename} already exists ({size:,} bytes). Use --force to re-download.")
        return True

    print(f"[....] Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {filename}: HTTP {e.code} — {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"[FAIL] {filename}: {e.reason}")
        print("       Check your internet connection / proxy settings.")
        return False

    size = dest.stat().st_size
    floor = MIN_EXPECTED_BYTES.get(filename, 1000)
    if size < floor:
        print(f"[FAIL] {filename} downloaded but is only {size:,} bytes "
              f"(expected at least {floor:,}). It may be an error page, not the dataset.")
        print(f"       Inspect it manually: {dest}")
        return False

    print(f"[ OK ] {filename} — {size:,} bytes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download the NSL-KDD dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading NSL-KDD dataset")
    print(f"Source: {BASE_URL}")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    results = [download_file(f, args.force) for f in FILES]

    print()
    print("=" * 60)
    if all(results):
        print("All NSL-KDD files downloaded successfully.")
        print("Next: python scripts/explore_dataset.py --dataset nsl-kdd")
        sys.exit(0)
    else:
        print("Some files failed to download. If the mirror above is unreachable,")
        print("download manually from the Wayback Machine snapshot and place the")
        print(f"files directly in {OUTPUT_DIR}:")
        print("  https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/")
        sys.exit(1)


if __name__ == "__main__":
    main()
