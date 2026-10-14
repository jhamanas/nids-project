# models/

- `payl_baseline.py` — **Phase 4**: PAYL-style detector
  - `build_profiles()` — per-(service, length_bucket) mean/variance profile
    from normal-traffic byte histograms, plus a percentile-based alert threshold
  - `score_features()` — diagonal Mahalanobis distance scoring, vectorized per group
  - `save_profiles()` / `load_profiles()` — pickle persistence
  - See the module docstring for the documented scope decisions: byte-histogram
    features only (not n-grams), diagonal covariance (not full), and how
    unseen (service, length_bucket) combinations are handled (scored as
    maximally anomalous, not silently passed through)

Autoencoder (Phase 5) is not yet built.

Tested against a synthetic dataset (`tests/fixtures/synthetic_features.py`) with
enough samples to build a real profile — 8 tests, including a check that attack-like
payloads are flagged far more often than normal ones (5/5 vs ~1/60 in practice).

## Hotfix: streaming variants for large files

- `build_profiles_streaming()` / `score_features_streaming()` — batch-based
  equivalents of `build_profiles()`/`score_features()`, for files too large
  to hold in memory (a real CICIDS2017 day's features file is several GB).
  Verified numerically identical to the in-memory versions in
  `tests/test_payl_baseline_streaming.py`, and stress-tested successfully on
  a 2-million-row dataset in a 3.9 GB RAM sandbox.
- `top_n_rows()` — finds the top-N rows by a score column across a huge
  Parquet file without loading it all into memory (used for the "top 10
  highest-scoring payloads" report).
