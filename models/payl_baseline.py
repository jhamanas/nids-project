"""
Phase 4 — PAYL-style baseline detector.

Builds a per-(service, length_bucket) profile of "normal" traffic from
byte-histogram features (Wang & Stolfo 2004's original PAYL feature), then
scores new payloads by their diagonal Mahalanobis distance from the matching
profile. Payloads whose distance exceeds a percentile-based threshold
(learned from the training data itself) are flagged anomalous.

Deliberate scope decisions (documented, not hidden):

  - Uses byte_hist_* features only (256-dim), not the n-gram features from
    Phase 3. This matches the original PAYL paper exactly; n-gram-based
    scoring (the Anagram extension) would be a natural Phase 4b/extension
    but is out of scope here to keep the baseline faithful to PAYL.

  - Uses DIAGONAL covariance (per-byte-position variance only, no
    cross-feature covariance terms). A full 256x256 covariance matrix over
    real traffic is very likely to be singular or near-singular (many byte
    positions are constant or near-constant within a given service/length
    bucket), which makes Mahalanobis distance with a full covariance matrix
    numerically unstable. This is a standard, literature-acknowledged
    simplification for PAYL-style detectors -- see config.yaml's
    `payl_baseline.covariance` setting.

  - A (service, length_bucket) combination with fewer than
    `min_samples_per_profile` training examples is marked
    "insufficient_data" rather than given an unreliable profile. At scoring
    time, any payload whose (service, length_bucket) has no usable profile
    (either never seen in training, or marked insufficient) is scored as
    maximally anomalous (score = +inf, flagged). This is a deliberate
    choice: an unseen traffic *shape* is itself suspicious information for
    an anomaly detector, not a reason to silently pass it through.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("byte_hist_")]


def build_profiles(features_df: pd.DataFrame, config: dict) -> dict:
    """
    Build one profile per (service, length_bucket) group from training
    (assumed normal-only) traffic.

    Returns a dict keyed by (service, length_bucket) tuple, each value either:
        {"status": "ok", "mean": np.ndarray(256,), "var": np.ndarray(256,),
         "threshold": float, "n_samples": int,
         "train_distance_mean": float, "train_distance_max": float}
    or:
        {"status": "insufficient_data", "n_samples": int}
    """
    feat_cols = _feature_columns(features_df)
    payl_cfg = config["payl_baseline"]
    min_samples = payl_cfg["min_samples_per_profile"]
    epsilon = payl_cfg["variance_epsilon"]
    percentile = payl_cfg["alert_percentile"]

    profiles = {}
    for (service, length_bucket), group in features_df.groupby(["service", "length_bucket"]):
        n = len(group)
        key = (service, length_bucket)

        if n < min_samples:
            profiles[key] = {"status": "insufficient_data", "n_samples": n}
            continue

        X = group[feat_cols].to_numpy(dtype=np.float64)
        mean = X.mean(axis=0)
        var = X.var(axis=0) + epsilon

        diff = X - mean
        train_distances = np.sqrt(np.sum((diff ** 2) / var, axis=1))
        threshold = float(np.percentile(train_distances, percentile))

        profiles[key] = {
            "status": "ok",
            "mean": mean,
            "var": var,
            "threshold": threshold,
            "n_samples": n,
            "train_distance_mean": float(train_distances.mean()),
            "train_distance_max": float(train_distances.max()),
        }

    return profiles


def _compute_scores(features_df: pd.DataFrame, profiles: dict) -> tuple:
    """
    Core scoring logic, returning the three raw arrays (scores, is_anomaly,
    profile_available) without copying features_df. Shared by score_features()
    (which copies, for callers that expect not to have their input mutated)
    and the streaming path (which mutates its own private per-batch
    DataFrame directly, since making an extra full copy of a large batch is
    wasteful and was a real cause of excess memory use at scale).
    """
    feat_cols = _feature_columns(features_df)
    n = len(features_df)

    scores = np.full(n, np.inf)
    is_anomaly = np.ones(n, dtype=bool)
    profile_available = np.zeros(n, dtype=bool)

    # Vectorized per-group scoring (fast even at hundreds of thousands of
    # rows) rather than a per-row Python loop.
    for (service, length_bucket), group in features_df.groupby(["service", "length_bucket"]):
        profile = profiles.get((service, length_bucket))
        positions = features_df.index.get_indexer(group.index)

        if profile is None or profile["status"] != "ok":
            continue  # leave as inf / anomaly=True / profile_available=False

        X = group[feat_cols].to_numpy(dtype=np.float64)
        diff = X - profile["mean"]
        distances = np.sqrt(np.sum((diff ** 2) / profile["var"], axis=1))

        scores[positions] = distances
        is_anomaly[positions] = distances > profile["threshold"]
        profile_available[positions] = True

    return scores, is_anomaly, profile_available


def score_features(features_df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """
    Score every row of features_df against the matching profile.

    Adds three columns to a copy of features_df:
        payl_score              float -- Mahalanobis distance (np.inf if no usable profile)
        payl_anomaly            bool  -- True if score > profile threshold (or no profile at all)
        payl_profile_available  bool  -- False if (service, length_bucket) had no usable profile
    """
    scores, is_anomaly, profile_available = _compute_scores(features_df, profiles)

    result = features_df.copy()
    result["payl_score"] = scores
    result["payl_anomaly"] = is_anomaly
    result["payl_profile_available"] = profile_available
    return result


def save_profiles(profiles: dict, config: dict, path: Path):
    payload = {
        "profiles": profiles,
        "payl_config": config["payl_baseline"],
        "saved_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_profiles(path: Path) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload


# ---------------------------------------------------------------------------
# Streaming variants -- for training/scoring files too large to hold in memory
# (e.g. a full CICIDS2017 day's features file can be several GB / millions of
# rows). Numerically equivalent to build_profiles()/score_features() above
# (verified in tests/test_payl_baseline_streaming.py against the in-memory
# versions on identical data), just computed via two passes over the file
# in fixed-size batches instead of one pass over an in-memory DataFrame.
# ---------------------------------------------------------------------------

BYTE_HIST_COLUMNS = [f"byte_hist_{i:03d}" for i in range(256)]


def build_profiles_streaming(input_path: Path, config: dict, batch_size: int = 200_000,
                              verbose: bool = True) -> dict:
    """
    Two-pass streaming equivalent of build_profiles(), for training files too
    large to load into memory at once.

    Pass 1: accumulate per-(service, length_bucket) sum and sum-of-squares
    (not the raw samples) to compute mean/variance -- this is the standard
    "sum of squares" streaming-variance formula: Var(X) = E[X^2] - E[X]^2.
    Only the byte_hist_* + grouping columns are read from the Parquet file
    (not the n-gram or metadata columns), which cuts memory/IO further since
    training doesn't use those.

    Pass 2: with mean/variance now finalized, re-read the file to compute
    each training sample's own distance from its profile, needed for the
    percentile threshold. Distances are single floats (not 256-dim vectors),
    so accumulating all of them in memory is lightweight even at millions of
    rows (~8 bytes each).
    """
    import pyarrow.parquet as pq

    payl_cfg = config["payl_baseline"]
    min_samples = payl_cfg["min_samples_per_profile"]
    epsilon = payl_cfg["variance_epsilon"]
    percentile = payl_cfg["alert_percentile"]

    needed_cols = ["service", "length_bucket"] + BYTE_HIST_COLUMNS
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows

    # ---- Pass 1: streaming mean/variance per group ----
    accumulators: dict[tuple, dict] = {}
    rows_done = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=needed_cols):
        df = batch.to_pandas()
        for (service, length_bucket), group in df.groupby(["service", "length_bucket"]):
            key = (service, length_bucket)
            X = group[BYTE_HIST_COLUMNS].to_numpy(dtype=np.float64)
            acc = accumulators.setdefault(key, {
                "n": 0, "sum": np.zeros(256), "sumsq": np.zeros(256),
            })
            acc["n"] += len(X)
            acc["sum"] += X.sum(axis=0)
            acc["sumsq"] += np.square(X).sum(axis=0)
        rows_done += len(df)
        if verbose:
            print(f"  [pass 1/2] {rows_done:,}/{total_rows:,} rows "
                  f"({100 * rows_done / total_rows:.1f}%)")

    profiles = {}
    for key, acc in accumulators.items():
        n = acc["n"]
        if n < min_samples:
            profiles[key] = {"status": "insufficient_data", "n_samples": n}
            continue
        mean = acc["sum"] / n
        # E[X^2] - E[X]^2 can go slightly negative due to floating-point
        # cancellation when true variance is near zero -- clip before adding
        # epsilon rather than let it silently go negative.
        var = np.maximum(acc["sumsq"] / n - np.square(mean), 0.0) + epsilon
        profiles[key] = {"status": "ok", "mean": mean, "var": var, "n_samples": n}

    # ---- Pass 2: training-distance percentile threshold per group ----
    distances_by_group: dict[tuple, list] = {k: [] for k, p in profiles.items() if p["status"] == "ok"}
    rows_done = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=needed_cols):
        df = batch.to_pandas()
        for (service, length_bucket), group in df.groupby(["service", "length_bucket"]):
            key = (service, length_bucket)
            profile = profiles.get(key)
            if profile is None or profile["status"] != "ok":
                continue
            X = group[BYTE_HIST_COLUMNS].to_numpy(dtype=np.float64)
            diff = X - profile["mean"]
            distances = np.sqrt(np.sum((diff ** 2) / profile["var"], axis=1))
            distances_by_group[key].append(distances)
        rows_done += len(df)
        if verbose:
            print(f"  [pass 2/2] {rows_done:,}/{total_rows:,} rows "
                  f"({100 * rows_done / total_rows:.1f}%)")

    for key, chunks in distances_by_group.items():
        all_distances = np.concatenate(chunks)
        threshold = float(np.percentile(all_distances, percentile))
        profiles[key]["threshold"] = threshold
        profiles[key]["train_distance_mean"] = float(all_distances.mean())
        profiles[key]["train_distance_max"] = float(all_distances.max())

    return profiles


def score_features_streaming(input_path: Path, output_path: Path, profiles: dict,
                              batch_size: int = 200_000, verbose: bool = True) -> dict:
    """
    Streaming equivalent of score_features(), for files too large to load
    into memory at once. Reads input in batches, scores each batch, and
    writes results incrementally to output_path via ParquetWriter -- the
    full scored dataset is never held in memory at once.

    Two memory optimizations beyond just batching (both verified not to
    change results -- see tests/test_payl_baseline_streaming.py):
      - Drops ngram_hist_* columns on read. PAYL scoring only uses
        byte_hist_* (see this module's docstring); reading the n-gram
        columns too would nearly double memory per batch for columns that
        are never used here. The output file will not have ngram_hist_*
        columns -- if you need them, they're still in the input features file.
      - Scores each batch's own freshly-loaded DataFrame in place rather
        than making an extra full copy of it (score_features() copies,
        which is the right default for a function whose caller might reuse
        their input elsewhere -- but here, the batch DataFrame is private
        and thrown away right after, so an extra copy of it is pure waste).

    Returns a small summary dict (total scored, flagged count, per-service
    flagged counts) since the caller can't easily recompute these from the
    output file without re-reading it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows

    all_columns = parquet_file.schema.names
    read_columns = [c for c in all_columns if not c.startswith("ngram_hist_")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows_done = 0
    n_flagged = 0
    n_no_profile = 0
    service_flagged: dict[str, int] = {}
    service_total: dict[str, int] = {}

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
            df = batch.to_pandas()  # private per-iteration object -- safe to mutate directly
            scores, is_anomaly, profile_available = _compute_scores(df, profiles)
            df["payl_score"] = scores
            df["payl_anomaly"] = is_anomaly
            df["payl_profile_available"] = profile_available

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)

            n_flagged += int(is_anomaly.sum())
            n_no_profile += int((~profile_available).sum())
            if "service" in df.columns:
                grouped = df.groupby("service")["payl_anomaly"].agg(["sum", "count"])
                for service, row in grouped.iterrows():
                    service_flagged[service] = service_flagged.get(service, 0) + int(row["sum"])
                    service_total[service] = service_total.get(service, 0) + int(row["count"])

            rows_done += len(df)
            if verbose:
                print(f"  {rows_done:,}/{total_rows:,} rows scored "
                      f"({100 * rows_done / total_rows:.1f}%)")
    finally:
        if writer is not None:
            writer.close()

    return {
        "total_scored": rows_done,
        "n_flagged": n_flagged,
        "n_no_profile": n_no_profile,
        "service_flagged": service_flagged,
        "service_total": service_total,
    }


def top_n_rows(path: Path, n: int, score_col: str = "payl_score",
                columns: list[str] = None, batch_size: int = 200_000) -> pd.DataFrame:
    """
    Find the top-N rows by score_col in a (potentially huge) Parquet file,
    without ever loading the whole file into memory. Reads only `columns`
    (pass a slim list -- just what you need to display -- since this is
    meant for reporting, not for getting the full row back).
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    best = None
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        candidate = pd.concat([best, df], ignore_index=True) if best is not None else df
        best = candidate.nlargest(n, score_col, keep="first")
    return best.reset_index(drop=True) if best is not None else pd.DataFrame(columns=columns or [])
