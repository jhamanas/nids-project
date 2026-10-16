# Evaluation & Benchmarking Suite (Phases 6 & 7)

This directory contains the benchmarking pipelines, evaluation metrics, adversarial evasion engines, and empirical reports comparing all three intrusion detection paradigms.

---

## 1. Core Modules

| Script / Module | Description |
| :--- | :--- |
| [`metrics.py`](metrics.py) | Mathematical implementations of binary classification metrics: Confusion Matrix (TP, FP, TN, FN), Precision, Recall, F1-Score, False Positive Rate (FPR), ROC curve, and Precision-Recall (PR) curve coordinate generation and AUC integration. |
| [`compare_models.py`](compare_models.py) | Memory-bounded, streaming lockstep evaluation pipeline comparing **PAYL** and **PyTorch Autoencoder** across 4.38M Friday packets. Uses directional ground-truth keys (`src->dst#port`) and reservoir sampling for exact ROC/PR curves. |
| [`eval_nsl_kdd.py`](eval_nsl_kdd.py) | Section C4 independent validation pipeline on the **NSL-KDD dataset** (`KDDTrain+.txt` and `KDDTest+.txt`), evaluating a Tabular Autoencoder and Flow Random Forest across DoS, Probe, R2L, and U2R attacks. |
| [`eval_evasion_mimicry.py`](eval_evasion_mimicry.py) | Phase 7 adversarial evasion benchmark assessing detector robustness under byte padding (D1), packet fragmentation (D2), and low-rate volumetric throttling (D3). |

---

## 2. Benchmark Reports & Visualizations

| Report / Artifact | Description |
| :--- | :--- |
| [`comprehensive_evaluation_report.md`](comprehensive_evaluation_report.md) | **Master Synthesis Report (Section C6):** Detailed architectural comparison, resolution of directional artifacts, empirical baseline shift verification, and production blueprint. |
| [`benchmark_report.md`](benchmark_report.md) & [`.json`](benchmark_report.json) | Binary classification metrics and per-class breakdown comparing PAYL vs. Autoencoder on Friday traffic. |
| [`flow_baseline_comparison.md`](flow_baseline_comparison.md) & [`.json`](flow_baseline_comparison.json) | **Section C3:** Three-detector comparison comparing PAYL (1D Payload), Autoencoder (Deep Payload), and Random Forest (Flow Metadata). |
| [`nsl_kdd_report.md`](nsl_kdd_report.md) & [`.json`](nsl_kdd_report.json) | **Section C4:** Independent evaluation results on NSL-KDD benchmark. |
| [`evasion_benchmark_report.md`](evasion_benchmark_report.md) & [`.json`](evasion_benchmark_report.json) | **Phase 7:** Evasion matrix summarizing detector degradation under adversarial padding and fragmentation. |
| [`dataset_scope_rationale.md`](dataset_scope_rationale.md) | **Section C5:** Formal computational and methodological rationale for scoping deep payload analysis to Monday and Friday. |
| [`roc_pr_comparison.png`](roc_pr_comparison.png) | High-resolution plot displaying side-by-side Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves. |

---

## 3. How to Run the Evaluations

```powershell
# 1. Run PAYL vs Autoencoder benchmark on Friday traffic:
python evaluation\compare_models.py

# 2. Run independent NSL-KDD benchmark:
python evaluation\eval_nsl_kdd.py

# 3. Run Phase 7 adversarial evasion benchmark:
python evaluation\eval_evasion_mimicry.py
```
