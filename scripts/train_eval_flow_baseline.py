"""
scripts/train_eval_flow_baseline.py

Train and evaluate the flow-based Random Forest baseline on CICIDS2017:
- Training: Balanced split using benign flows + known training attack distribution.
- Evaluation: Test on Friday attack flows (DDoS, PortScan, Botnet, Benign).
- Metrics: Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, and feature importances.
- Comparison: Outputs structured comparison metrics against PAYL and Autoencoder.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_binary_metrics, compute_roc_pr_curves
from models.flow_baseline import FlowRandomForestBaseline, clean_flow_dataframe


BENIGN = "BENIGN"


def train_eval_flow_baseline(
    train_csvs: list[Path],
    test_csvs: list[Path],
    output_model_path: Path,
    output_report_path: Path,
    sample_train_size: int = 150_000,
    sample_test_size: int = 150_000,
) -> dict:
    # Check if train_csvs and test_csvs are disjoint or if we do a held-out split
    if train_csvs == test_csvs:
        print("[INFO] Evaluating on held-out split of test CSVs...")
        dfs = []
        for p in test_csvs:
            print(f"  Reading {p.name}...")
            dfs.append(pd.read_csv(p, low_memory=False))
        full_df = pd.concat(dfs, ignore_index=True)
        X_all, y_all_labels = clean_flow_dataframe(full_df)
        y_all = (y_all_labels != BENIGN).astype(int).values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
            X_all, y_all, y_all_labels, test_size=0.3, stratify=y_all, random_state=42
        )
    else:
        print(f"[INFO] Loading training flows from {len(train_csvs)} files...")
        train_dfs = [pd.read_csv(p, low_memory=False) for p in train_csvs]
        train_full = pd.concat(train_dfs, ignore_index=True)
        X_train, y_train_labels = clean_flow_dataframe(train_full)
        y_train = (y_train_labels != BENIGN).astype(int).values

        print(f"[INFO] Loading test flows from {len(test_csvs)} files...")
        test_dfs = [pd.read_csv(p, low_memory=False) for p in test_csvs]
        test_full = pd.concat(test_dfs, ignore_index=True)
        X_test, y_test_labels = clean_flow_dataframe(test_full)
        y_test = (y_test_labels != BENIGN).astype(int).values

    # Cap training and test sizes to maintain speed and bounded memory
    if len(X_train) > sample_train_size:
        print(f"[INFO] Downsampling training set to {sample_train_size:,}...")
        att_idx = np.where(y_train == 1)[0]
        ben_idx = np.where(y_train == 0)[0]
        n_att = min(len(att_idx), sample_train_size // 2)
        n_ben = sample_train_size - n_att
        ch_att = np.random.choice(att_idx, size=n_att, replace=False)
        ch_ben = np.random.choice(ben_idx, size=n_ben, replace=False)
        sel = np.sort(np.concatenate([ch_att, ch_ben]))
        X_train = X_train.iloc[sel].copy()
        y_train = y_train[sel]

    if len(X_test) > sample_test_size:
        print(f"[INFO] Downsampling test set to {sample_test_size:,}...")
        att_idx = np.where(y_test == 1)[0]
        ben_idx = np.where(y_test == 0)[0]
        n_att = min(len(att_idx), sample_test_size // 2)
        n_ben = sample_test_size - n_att
        ch_att = np.random.choice(att_idx, size=n_att, replace=False)
        ch_ben = np.random.choice(ben_idx, size=n_ben, replace=False)
        sel = np.sort(np.concatenate([ch_att, ch_ben]))
        X_test = X_test.iloc[sel].copy()
        y_test = y_test[sel]
        y_test_labels = y_test_labels.iloc[sel]

    print(f"[INFO] Training set composition: {len(X_train):,} total ({int((y_train == 1).sum()):,} attacks, {int((y_train == 0).sum()):,} benign)")

    print("[INFO] Fitting FlowRandomForestBaseline...")
    t0 = time.time()
    rf = FlowRandomForestBaseline(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"[INFO] Fitted Random Forest in {t_train:.1f}s.")

    rf.save(output_model_path)
    print(f"[INFO] Saved model to {output_model_path}")

    # Top Features
    top_features = rf.get_feature_importances(top_n=10)
    print("\n[INFO] Top 10 Flow Features by Gini Importance:")
    for feat, imp in top_features:
        print(f"  - {feat:30s}: {imp:.4f}")

    print(f"\n[INFO] Evaluating on {len(X_test):,} flows...")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    binary_metrics = compute_binary_metrics(y_test, y_pred)
    curves = compute_roc_pr_curves(y_test, y_prob)

    binary_metrics["roc_auc"] = curves["roc_auc"]
    binary_metrics["pr_auc"] = curves["pr_auc"]

    # Per-Class Breakdown
    per_class = {}
    for label in y_test_labels.unique():
        mask = (y_test_labels == label).values
        n_tot = int(mask.sum())
        n_flagged = int(y_pred[mask].sum())
        rec = float(n_flagged / n_tot) if n_tot else 0.0
        per_class[label] = {
            "total_flows": n_tot,
            "flagged_flows": n_flagged,
            "recall": rec,
        }

    summary = {
        "model": "FlowRandomForestBaseline",
        "training_time_sec": round(t_train, 2),
        "top_features": top_features,
        "metrics": binary_metrics,
        "per_class": per_class,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    _write_markdown_comparison(summary, output_report_path)
    print(f"[INFO] Saved flow evaluation report to {output_report_path}")

    return summary


def _write_markdown_comparison(flow_summary: dict, path: Path):
    m = flow_summary["metrics"]
    pc = flow_summary["per_class"]
    top_f = flow_summary["top_features"]

    lines = [
        "# Section C3: Three-Detector Intrusion Detection Benchmark",
        "",
        "Comparison of three distinct architectural paradigms on CICIDS2017:",
        "1. **PAYL Statistical Baseline:** 1D byte-frequency Mahalanobis distance per service bucket.",
        "2. **PyTorch Autoencoder:** 258-dim payload representation with 16-dim latent bottleneck reconstruction error.",
        "3. **Flow Random Forest Baseline:** 78 statistical network flow features (rates, durations, packet counts, flags).",
        "",
        "## 1. Multi-Paradigm Binary Detection Performance",
        "",
        "| Metric | PAYL (Payload 1D) | Autoencoder (Payload Deep) | Flow Baseline (Random Forest) | Best Detector |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **ROC-AUC** | 0.9813 | 0.9894 | **{m['roc_auc']:.4f}** | {'Flow Baseline' if m['roc_auc'] >= 0.9894 else 'Autoencoder'} |",
        f"| **PR-AUC** | 0.4018 | 0.4626 | **{m['pr_auc']:.4f}** | {'Flow Baseline' if m['pr_auc'] >= 0.4626 else 'Autoencoder'} |",
        f"| **Precision** | 1.98% | 73.77% | **{m['precision']*100:.2f}%** | {'Flow Baseline' if m['precision'] >= 0.7377 else 'Autoencoder'} |",
        f"| **Recall** | 0.70% | 97.49% | **{m['recall']*100:.2f}%** | {'Flow Baseline' if m['recall'] >= 0.9749 else 'Autoencoder'} |",
        f"| **F1-Score** | 0.0104 | 0.8399 | **{m['f1_score']:.4f}** | {'Flow Baseline' if m['f1_score'] >= 0.8399 else 'Autoencoder'} |",
        f"| **False Positive Rate** | 1.90% | 1.89% | **{m['false_positive_rate']*100:.2f}%** | {'Flow Baseline' if m['false_positive_rate'] <= 0.0189 else 'Autoencoder'} |",
        "",
        "## 2. Granular Flow Recall Breakdown",
        "",
        "| Flow Class | Total Test Flows | Flagged Flows | Detection Rate (Recall) |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for label, stats in sorted(pc.items()):
        rec_pct = stats["recall"] * 100
        lines.append(f"| **{label}** | {stats['total_flows']:,} | {stats['flagged_flows']:,} | **{rec_pct:.2f}%** |")

    lines.extend([
        "",
        "## 3. Top Predictive Flow Features (Feature Importance)",
        "",
        "| Rank | Feature Name | Gini Importance | Network Semantic |",
        "| :---: | :--- | :---: | :--- |",
    ])

    for i, (f_name, imp) in enumerate(top_f, start=1):
        lines.append(f"| {i} | `{f_name}` | {imp:.4f} | Statistical flow characteristic |")

    lines.extend([
        "",
        "## 4. Key Comparative Architectural Findings",
        "",
        "- **Payload vs. Flow Trade-off:**",
        "  - Flow features excel at detecting rate anomalies, volume floods (DDoS), and rapid connection cycling (PortScan) without needing packet payloads.",
        "  - However, flow metadata is **completely blind** to zero-day application exploits (SQL injection, XSS, shellcode in valid single-packet requests) where traffic rates and packet counts match benign flows exactly.",
        "  - Deep payload inspection (Autoencoder) inspects byte distributions directly inside single packets, catching structural payload anomalies independently of flow rates or session state.",
        "- **Complementary Hybrid Defense:** An optimal NIDS architecture should employ **Flow Filtering as a fast first stage** (to filter voluminous volumetric floods) and **Autoencoder Payload Inspection as a second stage** (to inspect application content for exploitation and shellcode).",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate flow baseline")
    parser.add_argument("--train-csv", type=Path, action="append", dest="train_csvs")
    parser.add_argument("--test-csv", type=Path, action="append", dest="test_csvs")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "flow_baseline.pkl")
    parser.add_argument("--report-out", type=Path, default=ROOT / "evaluation" / "flow_baseline_comparison.md")
    args = parser.parse_args()

    test_csvs = args.test_csvs
    if not test_csvs:
        test_csvs = [
            ROOT / "data" / "raw" / "cicids2017" / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            ROOT / "data" / "raw" / "cicids2017" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            ROOT / "data" / "raw" / "cicids2017" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ]

    train_csvs = args.train_csvs
    if not train_csvs:
        train_csvs = test_csvs

    train_eval_flow_baseline(
        train_csvs=train_csvs,
        test_csvs=test_csvs,
        output_model_path=args.model_out,
        output_report_path=args.report_out,
    )


if __name__ == "__main__":
    main()
