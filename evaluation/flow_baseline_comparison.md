# Section C3: Three-Detector Intrusion Detection Benchmark

Comparison of three distinct architectural paradigms on CICIDS2017:
1. **PAYL Statistical Baseline:** 1D byte-frequency Mahalanobis distance per service bucket.
2. **PyTorch Autoencoder:** 258-dim payload representation with 16-dim latent bottleneck reconstruction error.
3. **Flow Random Forest Baseline:** 78 statistical network flow features (rates, durations, packet counts, flags).

## 1. Multi-Paradigm Binary Detection Performance

| Metric | PAYL (Payload 1D) | Autoencoder (Payload Deep) | Flow Baseline (Random Forest) | Best Detector |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9813 | 0.9894 | **1.0000** | Flow Baseline |
| **PR-AUC** | 0.4018 | 0.4626 | **1.0000** | Flow Baseline |
| **Precision** | 1.98% | 73.77% | **99.91%** | Flow Baseline |
| **Recall** | 0.70% | 97.49% | **99.79%** | Flow Baseline |
| **F1-Score** | 0.0104 | 0.8399 | **0.9985** | Flow Baseline |
| **False Positive Rate** | 1.90% | 1.89% | **0.09%** | Flow Baseline |

## 2. Granular Flow Recall Breakdown

| Flow Class | Total Test Flows | Flagged Flows | Detection Rate (Recall) |
| :--- | :---: | :---: | :---: |
| **BENIGN** | 75,000 | 64 | **0.09%** |
| **Bot** | 522 | 386 | **73.95%** |
| **DDoS** | 33,361 | 33,345 | **99.95%** |
| **PortScan** | 41,117 | 41,110 | **99.98%** |

## 3. Top Predictive Flow Features (Feature Importance)

| Rank | Feature Name | Gini Importance | Network Semantic |
| :---: | :--- | :---: | :--- |
| 1 | `Avg Fwd Segment Size` | 0.0791 | Statistical flow characteristic |
| 2 | `Init_Win_bytes_forward` | 0.0702 | Statistical flow characteristic |
| 3 | `Fwd Packet Length Max` | 0.0679 | Statistical flow characteristic |
| 4 | `Subflow Fwd Bytes` | 0.0652 | Statistical flow characteristic |
| 5 | `Fwd Packet Length Mean` | 0.0631 | Statistical flow characteristic |
| 6 | `Total Length of Fwd Packets` | 0.0464 | Statistical flow characteristic |
| 7 | `Bwd Packet Length Min` | 0.0355 | Statistical flow characteristic |
| 8 | `Average Packet Size` | 0.0348 | Statistical flow characteristic |
| 9 | `PSH Flag Count` | 0.0345 | Statistical flow characteristic |
| 10 | `Init_Win_bytes_backward` | 0.0276 | Statistical flow characteristic |

## 4. Key Comparative Architectural Findings

- **Payload vs. Flow Trade-off:**
  - Flow features excel at detecting rate anomalies, volume floods (DDoS), and rapid connection cycling (PortScan) without needing packet payloads.
  - However, flow metadata is **completely blind** to zero-day application exploits (SQL injection, XSS, shellcode in valid single-packet requests) where traffic rates and packet counts match benign flows exactly.
  - Deep payload inspection (Autoencoder) inspects byte distributions directly inside single packets, catching structural payload anomalies independently of flow rates or session state.
- **Complementary Hybrid Defense:** An optimal NIDS architecture should employ **Flow Filtering as a fast first stage** (to filter voluminous volumetric floods) and **Autoencoder Payload Inspection as a second stage** (to inspect application content for exploitation and shellcode).
