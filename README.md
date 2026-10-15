# Anomalous Payload-Based Network Intrusion Detection

A system that detects network intrusions by analyzing raw packet **payload content**
(byte-frequency / n-gram statistics + deep learning), rather than relying solely on
flow metadata or known attack signatures.

Baseline: PAYL-style byte-frequency + Mahalanobis distance
Modern model: Autoencoder-based reconstruction-error anomaly detector
Datasets: CICIDS2017 (primary, payload-rich), NSL-KDD (flow-level baseline comparison)

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Environment setup & repo scaffolding | ✅ done |
| 1 | Data acquisition & exploration | ✅ done |
| 2 | Packet parsing & payload extraction | ✅ done |
| 3 | Feature engineering | ✅ done |
| 4 | PAYL baseline (Mahalanobis distance) | ✅ done |
| 5 | Autoencoder detector | ⬜ not started |
| 6 | Evaluation & comparison | ⬜ not started |
| 7 | Evasion / mimicry attack testing | ⬜ not started |
| 8 | Packaging & demo | ⬜ not started |

---

## Folder Structure

```
nids-project/
├── README.md                  ← this file
├── requirements.txt            ← Python dependencies
├── environment.yml             ← optional conda environment spec
├── .gitignore
├── config/
│   └── config.yaml             ← central paths, ports, thresholds, hyperparameters
├── data/
│   ├── raw/
│   │   ├── cicids2017/         ← place downloaded CICIDS2017 CSVs/PCAPs here (Phase 1)
│   │   └── nsl-kdd/            ← place downloaded NSL-KDD .txt/.csv files here (Phase 1)
│   └── processed/              ← cleaned/feature-extracted data lands here (Phase 3+)
├── preprocessing/               ← packet parsing + feature extraction modules (Phase 2-3)
├── models/                      ← PAYL baseline + autoencoder model code & saved weights (Phase 4-5)
├── evaluation/                  ← metrics, ROC/plots, comparison scripts (Phase 6)
├── notebooks/                   ← exploratory Jupyter notebooks (Phase 1+)
├── scripts/                     ← standalone utility/CLI scripts (all phases)
└── logs/                        ← run logs, alert output (Phase 8)
```

Each phase folder currently holds a `.gitkeep` placeholder — files land there as each
phase is built out.

---

## Phase 0 — Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install packet-capture system tools (outside pip)

These are required by `scapy`/`pyshark` for reading PCAPs and, later, live capture:

- **Wireshark / tshark** — https://www.wireshark.org/download.html
  - Debian/Ubuntu: `sudo apt install wireshark tshark`
  - macOS: `brew install wireshark`
- **libpcap** (usually preinstalled on Linux/macOS; Windows needs Npcap: https://npcap.com/)
- **Zeek** (optional, recommended for protocol-aware payload/session logs) — https://zeek.org/get-zeek/

### 4. Verify the environment

```bash
python scripts/verify_setup.py
```

This checks Python version, confirms every required package imports correctly, checks
whether `tshark` is on PATH, and confirms the folder structure above exists (creating
any missing folders).

### 5. Review configuration

Open `config/config.yaml` and confirm the paths match your machine before moving to
Phase 1 (data acquisition).

---

## Phase 1 — Data Acquisition & Exploration

### NSL-KDD (scripted, no registration needed)

```bash
python scripts/download_nsl_kdd.py
```

Pulls `KDDTrain+.txt`, `KDDTrain+_20Percent.txt`, `KDDTest+.txt`, and
`KDDTest-21.txt` from a stable GitHub mirror into `data/raw/nsl-kdd/`
(the original unb.ca host no longer serves these directly). Re-run with
`--force` to re-download.

### CICIDS2017 (manual — registration required)

CIC gates this dataset behind a short access-request form; it can't be
scripted. Full steps are in `data/raw/cicids2017/README.md`. Once files are in
place, verify them with:

```bash
python scripts/verify_cicids2017.py
```

### Explore both

```bash
python scripts/explore_dataset.py --dataset both
```

Prints label/attack-category distributions, class imbalance, missing/inf
value checks, and saves summary plots to `data/processed/`. Or work through
`notebooks/01_data_exploration.ipynb` interactively.

### Key Phase 1 findings (NSL-KDD, already verified against real data)

- Train set: 125,973 rows / Test set: 22,544 rows, 43 columns (41 features + label + difficulty)
- Category split: Normal 53.5%, DoS 36.5%, Probe 9.3%, R2L 0.8%, U2R 0.04% — R2L/U2R are extremely rare, worth remembering for Phase 6 evaluation
- `preprocessing/nsl_kdd_columns.py` holds the column list and attack→category mapping — import it, don't retype it

---

## Phase 2 — Packet Parsing & Payload Extraction

`preprocessing/pcap_parser.py` turns PCAP files into payload records, two ways:

- **Packet-level** (`extract_packet_payloads`) — one record per packet with a
  payload. Simple, fast, matches PAYL's original per-packet approach.
- **Session-level** (`extract_session_payloads`) — reassembles each TCP stream
  (by 5-tuple + direction) into one concatenated payload, handling simple
  out-of-order/retransmit segments. See the module docstring for what TCP
  edge cases are deliberately out of scope for this project.

### Try it now (no real PCAPs needed)

```bash
python scripts/generate_test_pcap.py            # builds tests/fixtures/sample.pcap
python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode both
python -m pytest tests/ -v                        # 9 tests, all passing
```

### Progress reporting (important for real, multi-GB PCAPs)

`parse_pcaps.py` prints a status line every 500,000 packets by default
(`--progress-every N` to change it, `0` to disable). This matters in
practice: a full CICIDS2017 day can be several million packets, and at
roughly 5,000-15,000 packets/sec, that's tens of minutes with otherwise zero
output — indistinguishable from a hung process without progress lines.

### Run against real CICIDS2017 PCAPs (once downloaded — see Phase 1)

```bash
python scripts/parse_pcaps.py --input data/raw/cicids2017/PCAPs --mode packet
# or, for a quick test on one large file before committing to the full run:
python scripts/parse_pcaps.py --input data/raw/cicids2017/PCAPs/Monday-WorkingHours.pcap \
    --mode session --limit 50000
```

Output lands in `data/processed/payloads_packet.parquet` and/or
`payloads_session.parquet` — payload bytes are stored as Parquet's native
binary type (verified to round-trip exactly, no base64 needed) and read back
as plain Python `bytes`.

### Verified

- `tests/test_pcap_parser.py` — 9 tests, all passing: service classification,
  raw payload preservation, session reassembly correctness (including
  skipping a duplicated/retransmitted segment), per-direction packet counts,
  and a `--limit` flag for testing against huge files without reading them
  in full.
- Parquet round-trip confirmed: reassembled HTTP payload bytes come back
  identical after a save/load cycle.

---

## Phase 3 — Feature Engineering

`preprocessing/feature_extractor.py` turns each payload's raw bytes into the
numeric features PAYL (Phase 4) and the autoencoder (Phase 5) consume:

| Feature | What it captures |
|---|---|
| `byte_histogram` | 256-dim normalized byte-value frequency — the core PAYL feature (Wang & Stolfo 2004) |
| `ngram_histogram` | Fixed-size n-gram frequency vector via CRC32 hashing (Anagram-style; avoids the 256^n raw n-gram space) |
| `shannon_entropy` | Bits of entropy — low for repetitive/structured payloads (e.g. NOP-sleds), higher for natural text |
| `printable_ratio` | Fraction of printable-ASCII bytes — text protocols near 1.0, binary/shellcode-like payloads near 0 |
| `length_bucket` | PAYL-style length segmentation (profiles are built per service *and* length bucket) |

### Memory (important for real, multi-million-row PCAPs)

`extract_features.py` processes the input in fixed-size batches
(`--batch-size`, default 200,000) and writes each batch's output to Parquet
as it goes — it never holds the full feature matrix in memory at once. This
isn't optional at real scale: a full CICIDS2017 day (e.g. Monday's ~5.7
million packets) would need roughly 23 GB just to hold one dense in-memory
copy of the full feature matrix, which crashes on most machines. Verified by
generating a 2-million-row synthetic dataset and running the full extraction
successfully in a sandbox with only 3.9 GB total RAM — proof the batching
genuinely bounds memory rather than just working on a machine with lots of
headroom to spare.

### Try it now

```bash
python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode both
python scripts/extract_features.py --input data/processed/payloads_packet.parquet \
    --output data/processed/features_packet.parquet
python -m pytest tests/ -v    # 26 tests, all passing
```

### Run against real CICIDS2017 payloads (once Phase 2 has been run on real PCAPs)

```bash
python scripts/extract_features.py --input data/processed/payloads_packet.parquet \
    --output data/processed/features_packet.parquet
```

Output: one row per payload, with 256 `byte_hist_*` columns + N `ngram_hist_*`
columns (N = `config.yaml`'s `features.ngram_buckets`) + entropy/printable_ratio/
length_bucket/payload_len, plus all metadata carried through from Phase 2
(service, ports, timestamps, etc.) — ready to feed straight into a Mahalanobis
distance calculation (Phase 4) or an autoencoder (Phase 5) as a flat numeric
matrix, e.g. `df.filter(like="byte_hist_").values`.

### Verified

- `tests/test_feature_extractor.py` — 17 tests against hand-constructed
  payloads with known properties (all-same-byte → entropy 0; all-256-values →
  entropy 8.0 exactly; NOP-sled pattern shows lower entropy than HTTP text; etc.)
- Ran end-to-end against the Phase 2 fixture output — sensible real numbers:
  the synthetic anomalous payload's entropy (2.00 bits) came out clearly
  lower than HTTP traffic (4.4-4.7 bits), exactly matching the intuition
  PAYL-style detection depends on.
- Caught and fixed a real bug during testing: `payload_len` was present in
  both the carried-through metadata and the recomputed scalar features,
  causing a duplicate-column error on Parquet write.

---

## Phase 4 — PAYL Baseline (Mahalanobis Distance)

`models/payl_baseline.py` builds a per-(service, length_bucket) profile of
normal traffic (mean + diagonal variance over the 256 byte-histogram
features from Phase 3), then scores new payloads by their Mahalanobis
distance from the matching profile. Payloads above a percentile-based
threshold (learned from the training data itself) are flagged anomalous.

Key design decisions — see the module docstring for full reasoning:
- **Byte-histogram features only**, not n-grams — matches the original PAYL
  paper (Wang & Stolfo 2004) exactly.
- **Diagonal covariance**, not full — a full 256×256 covariance matrix over
  real traffic is very likely singular; this is a standard, literature-
  acknowledged simplification.
- **Unseen (service, length_bucket) combinations are scored as maximally
  anomalous** (not silently passed through) — an unseen traffic shape is
  itself suspicious information for an anomaly detector.

### Try it now

```bash
python scripts/parse_pcaps.py --input tests/fixtures/sample.pcap --mode packet
python scripts/extract_features.py --input data/processed/payloads_packet.parquet \
    --output data/processed/features_packet.parquet
python scripts/train_payl_baseline.py --input data/processed/features_packet.parquet \
    --output models/payl_baseline_test.pkl
python scripts/score_payl_baseline.py --model models/payl_baseline_test.pkl \
    --input data/processed/features_packet.parquet --output data/processed/scored_test.parquet
python -m pytest tests/ -v    # 34 tests, all passing
```

Note: the 6-payload fixture is too small to build real profiles (fewer than
`min_samples_per_profile`, default 10) — you'll see every group marked
`insufficient_data` and every payload flagged, with a clear warning
explaining why. That's the correct, documented fallback behavior, not a bug.

### Run against real CICIDS2017 data

```bash
# Train ONLY on Monday (the benign-only day):
python scripts/train_payl_baseline.py \
    --input data/processed/features_packet_monday.parquet \
    --output models/payl_baseline.pkl

# Score any other day (e.g. Friday's DDoS traffic):
python scripts/score_payl_baseline.py --model models/payl_baseline.pkl \
    --input data/processed/features_packet_friday_ddos.parquet \
    --output data/processed/scored_friday_ddos.parquet
```

### Verified

- `tests/test_payl_baseline.py` — 8 tests against a synthetic dataset built
  specifically for this (`tests/fixtures/synthetic_features.py`: 60 normal
  HTTP-like payloads with realistic small variation + 5 structurally
  distinct attack-like payloads) — the Phase 2 fixture's 6 payloads aren't
  enough to build a real statistical profile.
- Ran the actual CLI scripts (not just the Python API) end-to-end: trained
  on 60 synthetic-normal payloads, scored 5 synthetic-attack payloads —
  **all 5 flagged (100% detection)**, with normal-payload scores in the
  3–9 range vs. attack-payload scores in the 571–674 range. That gap is
  the core premise of PAYL-style detection working as intended.
- Confirmed profile save/load round-trips exactly (identical scores before
  and after a pickle save/reload cycle).

---

## Next: Phase 5

Phase 5 covers the autoencoder-based detector — a deep-learning alternative
to Phase 4's statistical baseline, trained on the same normal traffic,
scored by reconstruction error instead of Mahalanobis distance. Not started
yet.

## Hotfix: streaming training/scoring for large real datasets

`train_payl_baseline.py` and `score_payl_baseline.py` previously loaded the
entire features file into memory (`pd.read_parquet`), which works fine for
small/synthetic data but crashes on a real CICIDS2017 day's features file
(multi-GB, `ArrowMemoryError`/`Killed`). Both now stream through the file in
batches instead:

- `train_payl_baseline.py` uses `build_profiles_streaming()` — two passes
  over the file (pass 1: accumulate sum/sum-of-squares per group for
  mean/variance; pass 2: compute each training sample's own distance for the
  percentile threshold), never holding more than one batch in memory.
- `score_payl_baseline.py` uses `score_features_streaming()` — scores and
  writes each batch immediately, drops the unused `ngram_hist_*` columns on
  read (PAYL scoring only uses `byte_hist_*`), and avoids an unnecessary
  extra copy per batch.

**Verified, not just assumed fixed:** `tests/test_payl_baseline_streaming.py`
proves the streaming versions produce numerically identical results to the
original in-memory functions (same mean/variance/threshold, same scores,
same anomaly flags) on the same data — the important property isn't just
"doesn't crash," it's "computes the same thing." Beyond that, both scripts
were run end-to-end against a 2-million-row synthetic dataset (~740 MB
Parquet, the same scale that crashed before) in a sandbox with only 3.9 GB
total RAM, and completed successfully: training, then scoring with a
correct top-10 report — the same combination that previously failed with an
`ArrowMemoryError` and a `Killed` (OOM) process respectively.

Both scripts also gained `--batch-size` (default 200,000) if you ever need
to tune it further for a particularly memory-constrained machine.

---

## Phase 4.5 — Heuristic Spot-Check & Empirical Results

`scripts/spot_check_labels.py` evaluates the trained PAYL baseline against
CICIDS2017's labeled flows (`GeneratedLabelledFlows`).

### Methodology
- **Key Disambiguation via Majority Vote:** Maps each `(host_pair, port)` key to
  a single winning label using flow counts (`max(flow_count)`), eliminating
  double-counting across colliding attack/benign traffic.
- **BENIGN Control Baseline:** Evaluates benign-only host pairs using identical
  key logic to measure baseline false-alarm rates.
- **Top-N Precision:** Examines the highest-scoring anomalies to inspect payload
  characteristics and identify attack vs. benign outlier traffic.

### Empirical Results (Friday Attack Traffic — ~4.4M Scored Packets)

With directional attack keys (`src->dst#port`), victim reply stream isolation, and pure benign control host pairs:

| Label | Scored Rows | Flagged Rows | Flagged Rate | Best Score | Best Rank | Best Pct |
|---|---|---|---|---|---|---|
| **Bot** | 2,545 | 670 | **26.33%** | 206.660 | 140 | 0.003% |
| **DDoS (forward flood)** | 95,814 | 20 | **0.02%** | 57.735 | 16,704 | 0.38% |
| **PortScan (probes)** | 19 | 0 | **0.00%** | 26.143 | 422,911 | 9.66% |
| **Victim Reply (to attacker)** | 2 | 2 | **100.00%** | 29.164 | 407,629 | 9.31% |
| **BENIGN (pure control)** | 1,801,194 | 34,210 | **1.90%** | 354.418 | 13 | 0.0003% |

**Key Findings:**
1. **Benign Baseline Calibration Confirmed (1.00% on Monday, 1.90% on Friday):**
   - Running the PAYL baseline over Monday's full benign dataset (5.68M packets) produces **56,626 flagged rows (exactly 1.00%)**, matching the 99th-percentile calibration.
   - On Friday's pure benign host pairs (1.80M packets), the false-positive rate is **1.90%**, confirming that baseline noise remains low across days.
2. **Directional Separation of Flood vs. Replies:**
   - The earlier ~66% DDoS flag rate was an artifact of undirected pairing that matched large victim-to-attacker reply streams.
   - Forward DDoS flood packets (`172.16.0.1 -> 192.168.10.50:80`) consist of small, repeated ~20-byte payloads whose byte frequencies mimic standard HTTP requests, causing PAYL to miss them at the payload level (0.02% flagged). This illustrates a foundational limitation of payload-only statistical profiles against volumetric floods and provides direct motivation for **Phase 5 (Autoencoder)** and flow-level detection.
3. **Botnet C&C Detection:**
   - Bot traffic (`192.168.10.x <-> 205.174.165.73:8080`) is reliably flagged at **26.33%** and generates the highest-ranked attack anomalies in the file (`206.66`, ranking 140th in the whole dataset).

### Try it:

```powershell
python scripts\spot_check_labels.py `
    --scored-parquet data\processed\scored_friday.parquet `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Morning.pcap_ISCX.csv `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv `
    --top-n 50
```

### Known Limitations & Threat Model Boundaries (A4, A5)

1. **DNS Layer Extraction (A5):** Scapy dissects RFC 1035 DNS packets into dedicated protocol header attributes rather than an unparsed `Raw` byte layer. Because the payload extractor targets raw bytes, well-formed DNS traffic is largely transparent to the raw extractor (only malformed/unparsed frames survive, ~20 out of 5.68M packets). In production pipelines, an explicit layer-aware reassembler would decode application payloads across all protocol layers.
2. **Service Classification Fallback (A4):** When `dst_port` is not in the known service map, the parser currently inspects `src_port` as a heuristic fallback. Consequently, port-scan probes originated with `src_port=80` scanning arbitrary destination ports can be heuristically labeled as "http". In strict anomaly detection pipelines, traffic should be categorized solely by destination service port, leaving non-standard destinations classified as `"other"`.

---

## Phase 5 — Deep Learning Autoencoder Detector

`models/autoencoder.py` implements a deep neural network that compresses 258-dimensional payload vectors (256 byte-histogram bins + Shannon entropy + printable ratio) through a bottleneck latent space (16 dimensions) and measures anomaly level by **Reconstruction Error (Mean Squared Error)**.

- **Architecture:** `Linear(258, 128) -> LayerNorm -> ReLU -> Linear(128, 64) -> LayerNorm -> ReLU -> Bottleneck(16)` and symmetric decoder.
- **Training:** Trained on 5.68M benign Monday packets in streaming batches using PyTorch Adam optimizer.
- **Threshold Calibration:** Anomaly threshold calibrated at the 99.0th percentile of normal reconstruction loss (`MSE = 0.000167`).

### Comparative Detection Results: PAYL vs. Autoencoder (Friday Traffic)

| Traffic Class | Total Packets | PAYL Flagged | **PAYL Recall** | Autoencoder Flagged | **Autoencoder Recall** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DDoS (forward flood)** | 95,814 | 20 | 0.02% | **95,750** | **99.93%** 🚀 |
| **PortScan (probes)** | 19 | 0 | 0.00% | **19** | **100.00%** 🚀 |
| **Botnet C&C** | 2,545 | 670 | **26.33%** | 143 | 5.62% |
| **BENIGN (pure control)** | 1,801,194 | 34,210 | **1.90%** FPR | 34,109 | **1.89%** FPR |

### Central Empirical Discovery:
* **Resolving Flood Payload Blindness:** Where PAYL's diagonal Mahalanobis distance missed forward DDoS flood packets (0.02% recall) because their individual byte frequencies resembled normal HTTP text, the **Autoencoder achieved 99.93% recall**. By compressing features into a 16-dimensional bottleneck, the Autoencoder captures non-linear inter-byte correlations that deviate significantly from typical traffic manifolds.
* **Stable False Positive Rate:** Both models maintain an identical false positive rate of **~1.9%** on unseen pure benign traffic, demonstrating that the Autoencoder's dramatic lift on DDoS detection was achieved without blowing up false alarms.

### Run Autoencoder Pipeline:

```powershell
# 1. Train on Monday benign features
python scripts\train_autoencoder.py `
    --input data\processed\features_packet.parquet `
    --output models\autoencoder.pt `
    --epochs 2 --batch-size 2048 --percentile 99

# 2. Score Friday features
python scripts\score_autoencoder.py `
    --model models\autoencoder.pt `
    --input data\processed\features_packet_friday.parquet `
    --output data\processed\scored_friday_autoencoder.parquet

# 3. Evaluate recall & precision
python scripts\spot_check_labels.py `
    --scored-parquet data\processed\scored_friday_autoencoder.parquet `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Morning.pcap_ISCX.csv `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv `
    --flow-csv data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv `
    --top-n 50
```

---

## Phase 6 — Evaluation, Multi-Paradigm Benchmarks & Scope (C1-C5)

### Dataset Scope Decision (Section C5)
In accordance with **Task C5**, the primary payload-level inspection pipeline is scoped to **Monday (5.68M packets, Benign Training)** and **Friday (4.38M packets, Attack & Control Evaluation)**.
- **Computational Boundary:** Ingesting and extracting all 5 days (52.4 GB raw PCAPs, ~27M packets) single-threaded requires ~16+ hours of Scapy parsing. Monday and Friday yield 10,057,595 packets, providing statistically overwhelming representation of normal traffic, DDoS floods, port scans, and botnets.
- **Multi-Day Representation:** Tuesday through Thursday traffic (e.g. FTP/SSH Patator brute-force) is integrated through the **Flow-Based Random Forest Baseline** and the independent **NSL-KDD Benchmark**.
- Detailed architectural analysis available in [`evaluation/dataset_scope_rationale.md`](evaluation/dataset_scope_rationale.md).
- Comprehensive Phase 6 benchmark report available in [`evaluation/comprehensive_evaluation_report.md`](evaluation/comprehensive_evaluation_report.md).

---

## Phase 7 — Adversarial Evasion & Mimicry Attacks

Phase 7 evaluates detector robustness under active evasion techniques:
1. **Payload Padding (D1):** Blending malicious payloads with benign HTTP keywords.
2. **Fragmentation (D2):** Splitting exploits across 16B, 32B, 64B MTU segments.
3. **Low-Rate Throttling (D3):** Throttling volumetric flood rates to evade flow thresholding.

### Run Evasion Benchmark:
```powershell
python evaluation\eval_evasion_mimicry.py
```
Report: [`evaluation/evasion_benchmark_report.md`](evaluation/evasion_benchmark_report.md)

---

## Phase 8 — Packaging, Master Pipeline & Live Demo

### 1. Master Pipeline (One Command End-to-End)
Runs PCAP parsing, feature engineering, and dual-model anomaly scoring end-to-end:
```powershell
python scripts\run_pipeline.py --pcap tests\fixtures\sample.pcap --output-dir data\processed\sample_run
```

### 2. Interactive Terminal PCAP Scoring Demo
Inspects packet streams in real-time, displays PAYL distances vs. Autoencoder reconstruction errors, and renders payload hex and ASCII previews:
```powershell
python scripts\demo_score_pcap.py --pcap tests\fixtures\sample.pcap --show-all
```

### 3. Full Test Suite (77 Automated Tests)
```powershell
python -m pytest tests/ -v
```


