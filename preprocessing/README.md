# preprocessing/

- `config_loader.py` — loads `config/config.yaml` from anywhere in the project
- `nsl_kdd_columns.py` — NSL-KDD column names + attack→category mapping (Phase 1)
- `pcap_parser.py` — **Phase 2**: turns PCAP files into payload records
  (packet-level and TCP-session-reassembled)
- `feature_extractor.py` — **Phase 3**: turns a payload (bytes) into numeric
  features:
  - `byte_histogram()` — 256-dim normalized byte-value frequency vector (core PAYL feature)
  - `ngram_histogram()` — fixed-size normalized n-gram frequency vector via CRC32 hashing trick
  - `shannon_entropy()` — entropy in bits
  - `printable_ratio()` — fraction of printable-ASCII bytes
  - `length_bucket()` — PAYL-style length segmentation
  - `extract_features()` — runs all of the above, returns one dict per payload

PAYL baseline (Phase 4: Mahalanobis distance over these features) and the
autoencoder (Phase 5) are not yet built — they consume the Parquet output of
`scripts/extract_features.py`.

Tested against hand-constructed payloads with known properties
(`tests/test_feature_extractor.py`, 17 tests) and against
`tests/fixtures/sample.pcap` end-to-end via `scripts/extract_features.py`.
