#!/usr/bin/env python3
"""
Phase 0 — Environment verification script.

Run this after `pip install -r requirements.txt` to confirm:
  1. Python version is adequate
  2. All required packages import correctly (with version numbers)
  3. System-level tools (tshark) are reachable on PATH
  4. The expected project folder structure exists (creates missing folders)
  5. config/config.yaml loads correctly

Usage:
    python scripts/verify_setup.py
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 10)

REQUIRED_PACKAGES = [
    "scapy",
    "pyshark",
    "numpy",
    "pandas",
    "yaml",       # pyyaml imports as `yaml`
    "tqdm",
    "sklearn",    # scikit-learn imports as `sklearn`
    "scipy",
    "torch",
    "matplotlib",
    "seaborn",
]

REQUIRED_DIRS = [
    "config",
    "data/raw/cicids2017",
    "data/raw/nsl-kdd",
    "data/processed",
    "preprocessing",
    "models",
    "evaluation",
    "notebooks",
    "scripts",
    "logs",
]

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check_python_version() -> bool:
    print("== Python version ==")
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    status = PASS if ok else FAIL
    print(f"{status} Python {sys.version.split()[0]} "
          f"(requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    print()
    return ok


def check_packages() -> bool:
    print("== Required packages ==")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "unknown version")
            print(f"{PASS} {pkg:<12} {version}")
        except ImportError:
            print(f"{FAIL} {pkg:<12} not installed — run: pip install -r requirements.txt")
            all_ok = False
    print()
    return all_ok


def check_system_tools() -> bool:
    print("== System tools ==")
    all_ok = True

    tshark_path = shutil.which("tshark")
    if tshark_path:
        try:
            result = subprocess.run(
                ["tshark", "--version"], capture_output=True, text=True, timeout=5
            )
            first_line = result.stdout.splitlines()[0] if result.stdout else "installed"
            print(f"{PASS} tshark found at {tshark_path} ({first_line})")
        except Exception:
            print(f"{WARN} tshark found at {tshark_path} but version check failed")
    else:
        print(f"{WARN} tshark not found on PATH — install Wireshark/tshark "
              f"(needed for pyshark and manual PCAP inspection in Phase 2)")
        # Not a hard failure — scapy alone can get you through early phases.

    zeek_path = shutil.which("zeek")
    if zeek_path:
        print(f"{PASS} zeek found at {zeek_path} (optional, used for protocol-aware logs)")
    else:
        print(f"{WARN} zeek not found on PATH — optional, skip unless you plan to use it in Phase 2")

    print()
    return all_ok


def check_folder_structure() -> bool:
    print("== Project folder structure ==")
    all_ok = True
    for rel_dir in REQUIRED_DIRS:
        dir_path = PROJECT_ROOT / rel_dir
        if dir_path.exists():
            print(f"{PASS} {rel_dir}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"{WARN} {rel_dir} was missing — created it")
    print()
    return all_ok


def check_config() -> bool:
    print("== Config file ==")
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        print(f"{FAIL} config/config.yaml not found")
        print()
        return False
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        required_keys = ["paths", "services", "features", "payl_baseline", "autoencoder"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            print(f"{FAIL} config.yaml is missing keys: {missing}")
            print()
            return False
        print(f"{PASS} config.yaml loaded successfully ({len(cfg)} top-level sections)")
        print()
        return True
    except Exception as e:
        print(f"{FAIL} config.yaml failed to parse: {e}")
        print()
        return False


def main():
    print("=" * 60)
    print("NIDS Project — Phase 0 Environment Verification")
    print("=" * 60)
    print()

    results = {
        "Python version": check_python_version(),
        "Packages": check_packages(),
        "System tools": check_system_tools(),
        "Folder structure": check_folder_structure(),
        "Config": check_config(),
    }

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        print(f"{PASS if ok else FAIL} {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("Environment ready. You can proceed to Phase 1 (data acquisition).")
        sys.exit(0)
    else:
        print("Fix the FAIL items above before proceeding to Phase 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
