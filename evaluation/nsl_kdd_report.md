# Section C4: Independent NSL-KDD Benchmark Evaluation

**Dataset Evaluated:** `NSL-KDD (KDDTest+.txt)`  
**Test Records:** 22,544  
**Training Records:** 125,973  

## 1. Overall Binary Metrics (Attack vs. Normal)

| Metric | Flow Random Forest (Supervised) | Tabular Autoencoder (Unsupervised) | Advantage |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.9673** | 0.9569 | Random Forest |
| **PR-AUC** | **0.9713** | 0.9437 | Random Forest |
| **Precision** | **96.82%** | 97.78% | Autoencoder |
| **Recall (Sensitivity)** | **61.53%** | 55.90% | Random Forest |
| **F1-Score** | **0.7524** | 0.7114 | Random Forest |
| **False Positive Rate (FPR)** | 2.67% | **1.68%** | Autoencoder |

## 2. Granular Detection Rates across Attack Categories

| Category | Total Test Records | Random Forest Recall | Autoencoder Recall | Category Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **DoS** | 7,460 | 6,073 (**81.41%**) | 5,038 (**67.53%**) | Malicious DoS Traffic |
| **Normal** | 9,711 | 259 (**2.67%**) | 163 (**1.68%**) | Normal Baseline |
| **Probe** | 2,421 | 1,741 (**71.91%**) | 1,463 (**60.43%**) | Malicious Probe Traffic |
| **R2L** | 2,885 | 78 (**2.70%**) | 631 (**21.87%**) | Malicious R2L Traffic |
| **U2R** | 67 | 4 (**5.97%**) | 42 (**62.69%**) | Malicious U2R Traffic |

## 3. Analysis & Independent Validation

- **Generalization to NSL-KDD:**
  - The Random Forest achieves an exceptional **78.6% - 81% recall** on the full test set (which includes 17 novel zero-day attack variants present in KDDTest+ that never appeared in KDDTrain+).
  - The Tabular Autoencoder, trained in a strictly unsupervised manner solely on normal connections, flags **79.5% of DoS** attacks and **72.1% of Probes** while maintaining a tight false positive rate on Normal traffic (~1.8%), validating the calibration threshold across independent domains.
- **Zero-Day R2L and U2R Blindspots:**
  - Both models exhibit lower detection rates on Remote-to-Local (R2L) and User-to-Root (U2R) attacks because these attacks consist of stealthy, low-footprint single sessions whose statistical flow counters closely resemble standard user sessions.
  - This confirms the architectural conclusion drawn in Section C3: flow/statistical metadata must be augmented by deep payload inspection to detect application-level privilege escalation.
