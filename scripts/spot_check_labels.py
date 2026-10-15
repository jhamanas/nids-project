"""
spot_check_labels.py

Heuristic spot-check of PAYL anomaly scores against CICIDS2017's own labeled
flow CSVs (GeneratedLabelledFlows / TrafficLabelling / Friday-*.csv).

It answers two questions, with deliberately loose matching:

  1. PRECISION (top-N): of the highest-scoring anomalies, how many fall on
     host pairs that the labeled flows call attacks?
  2. RECALL: for each attack label, how many scored rows involve that
     attack's host pairs and ports, how many PAYL flagged (payl_anomaly),
     and where the best-scoring one ranks in the whole file?

THIS IS NOT A RIGOROUS EVALUATION. It is a sanity check before the real
Phase 6 methodology. Known limitations, by design:

  - Matching is on IP pair (both directions) plus port when available.
    It ignores timestamp: CICIDS2017 timestamps have minute-level
    granularity and timezone quirks, so exact joins are unreliable.
    A host pair that talks constantly (e.g. a DNS server) matches many
    labeled flows, so treat matches as corroborating evidence, not proof.
  - Port matching accepts either the flow's Source Port or Destination
    Port, because CICFlowMeter may record the reply side of a session.
  - Top-N results are deduplicated by (src_ip, dst_ip, dst_port) so one
    chatty session cannot fill the list with copies of itself.
  - Recall is measured per (host pair, port), not per flow. A key labeled
    as an attack is assumed to be attack traffic for all its rows, which
    is approximate: a key can also carry benign flows.
  - "No match" or "not flagged" does NOT mean "confirmed benign".

Usage (run each flow CSV on its own line in PowerShell):
    python scripts/spot_check_labels.py `
        --scored-parquet data\\processed\\scored_friday.parquet `
        --flow-csv data\\raw\\cicids2017\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Morning.pcap_ISCX.csv `
        --flow-csv data\\raw\\cicids2017\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv `
        --flow-csv data\\raw\\cicids2017\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv `
        --top-n 50 --output data\\processed\\spotcheck_friday.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BENIGN = "BENIGN"


# ---------------------------------------------------------------------------
# Step 1: top-N anomalies, streamed and deduplicated.
# ---------------------------------------------------------------------------

# Columns we WANT if they exist in the file. Missing ones are reported, and
# matching degrades gracefully without them.
DESIRED_COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "timestamp", "service", "payload_len", "payl_score",
]

# Flag column written by the PAYL scorer, if present. Used for recall.
# The real scored_friday.parquet uses payl_anomaly.
FLAG_CANDIDATES = ["payl_anomaly", "is_anomaly", "flagged"]


def _available_columns(parquet_path: Path) -> list:
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    cols = [c for c in DESIRED_COLUMNS if c in schema_names]
    missing = [c for c in DESIRED_COLUMNS if c not in schema_names]
    if missing:
        print(f"[WARN] scored parquet is missing columns {missing} — "
              f"matching will be weaker without them.", file=sys.stderr)
    if "src_ip" not in cols or "dst_ip" not in cols or "payl_score" not in cols:
        raise ValueError(
            "scored parquet must have at least src_ip, dst_ip, payl_score "
            f"to spot-check. Found: {cols}"
        )
    return cols


def _dedupe_key(df: pd.DataFrame) -> list:
    key = ["src_ip", "dst_ip"]
    if "dst_port" in df.columns:
        key.append("dst_port")
    return key


def _dedupe_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the highest-scoring row per (src, dst, dst_port), then the top n."""
    df = df.sort_values("payl_score", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=_dedupe_key(df), keep="first")
    return df.head(n)


def top_n_anomalies(parquet_path: Path, n: int, batch_size: int = 200_000) -> pd.DataFrame:
    """Top-n distinct anomalies by payl_score, streamed in bounded memory.

    Correct under trimming: a host key that falls outside the current top-n
    distinct maxima can only re-enter if a later row for it scores higher,
    and that later row is then seen fresh in its batch.
    """
    cols = _available_columns(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    best = None
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        best = df if best is None else pd.concat([best, df], ignore_index=True)
        if len(best) > n * 5:
            best = _dedupe_top(best, n).reset_index(drop=True)
    if best is None or best.empty:
        return pd.DataFrame(columns=cols)
    return _dedupe_top(best, n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: load the labeled flow CSVs, normalizing the CICIDS2017 quirks
# (leading-space column names, stray whitespace in label values).
# ---------------------------------------------------------------------------

FLOW_COLUMN_MAP = {
    "source ip": "Source IP",
    "src ip": "Source IP",
    "source port": "Source Port",
    "src port": "Source Port",
    "destination ip": "Destination IP",
    "dst ip": "Destination IP",
    "destination port": "Destination Port",
    "dst port": "Destination Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    "label": "Label",
}


def load_flow_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in FLOW_COLUMN_MAP:
            rename[c] = FLOW_COLUMN_MAP[key]
    df = df.rename(columns=rename)

    required = ["Source IP", "Destination IP", "Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing} after normalization")

    for col in ("Source Port", "Destination Port", "Protocol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label values can carry stray whitespace; "BENIGN " != "BENIGN" otherwise.
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Source IP"] = df["Source IP"].astype(str).str.strip()
    df["Destination IP"] = df["Destination IP"].astype(str).str.strip()

    df["__source_file"] = path.name
    return df


def load_all_flows(paths) -> pd.DataFrame:
    frames = [load_flow_csv(Path(p)) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3: heuristic matching for the top-N precision check.
# ---------------------------------------------------------------------------

def find_matches(anomaly: pd.Series, flows: pd.DataFrame) -> pd.DataFrame:
    """Flow rows that heuristically correspond to this anomaly.

    Matches on IP pair (either direction). If the anomaly has a port that
    appears on some of those flows, narrows to those; otherwise keeps all
    IP-pair matches, because payload-side port bookkeeping is the least
    reliable field.
    """
    ip_a = anomaly.get("src_ip")
    ip_b = anomaly.get("dst_ip")
    if ip_a is None or ip_b is None:
        return flows.iloc[0:0]

    ip_mask = (
        ((flows["Source IP"] == ip_a) & (flows["Destination IP"] == ip_b))
        | ((flows["Source IP"] == ip_b) & (flows["Destination IP"] == ip_a))
    )
    candidates = flows[ip_mask]
    if candidates.empty:
        return candidates

    ports = set()
    for key in ("src_port", "dst_port"):
        val = anomaly.get(key)
        if val is not None and not pd.isna(val):
            ports.add(val)

    if ports and "Source Port" in candidates.columns and "Destination Port" in candidates.columns:
        port_mask = (
            candidates["Source Port"].isin(ports) | candidates["Destination Port"].isin(ports)
        )
        narrowed = candidates[port_mask]
        if not narrowed.empty:
            candidates = narrowed

    return candidates


def spot_check(anomalies: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, anomaly in anomalies.iterrows():
        matches = find_matches(anomaly, flows)
        base = {
            "src_ip": anomaly.get("src_ip"),
            "dst_ip": anomaly.get("dst_ip"),
            "dst_port": anomaly.get("dst_port"),
            "service": anomaly.get("service"),
            "payl_score": anomaly.get("payl_score"),
        }
        if matches.empty:
            rows.append({**base, "n_matching_flows": 0, "n_attack_flows": 0,
                         "labels_found": "NO MATCH", "source_files": ""})
            continue

        label_counts = matches["Label"].value_counts()
        labels_str = "; ".join(f"{label} (x{count})" for label, count in label_counts.items())
        source_files = ", ".join(sorted(matches["__source_file"].unique()))
        rows.append({
            **base,
            "n_matching_flows": len(matches),
            "n_attack_flows": int((matches["Label"] != BENIGN).sum()),
            "labels_found": labels_str,
            "source_files": source_files,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: recall. For each attack label, which scored rows involve that
# label's (host pair, port) keys, how many were flagged, and where the best
# one ranks. Two streaming passes over the scored file.
# ---------------------------------------------------------------------------

def _port_str(x) -> str:
    """Normalize a port to a comparable string: 22.0, 22 and '22' -> '22'."""
    try:
        if x is None or pd.isna(x):
            return "nan"
        return str(int(x))
    except (TypeError, ValueError):
        return str(x)


def _pair_key(a, b) -> str:
    """Direction-insensitive host-pair key."""
    a, b = str(a).strip(), str(b).strip()
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _pair_keys(src: pd.Series, dst: pd.Series) -> pd.Series:
    return pd.Series([_pair_key(x, y) for x, y in zip(src, dst)], index=src.index)


def build_label_keys(flows: pd.DataFrame, use_port: bool) -> tuple[dict, dict]:
    """Map key -> single winning label (via majority vote of flow counts),
    and return (label_keys, ambiguity_info).

    Returns:
        label_keys: dict mapping key -> winning_label
        ambiguity_info: dict with stats on collisions/ambiguities
    """
    key_label_counts = {}
    for src, dst, sport, dport, label in flows[
        ["Source IP", "Destination IP", "Source Port", "Destination Port", "Label"]
    ].itertuples(index=False):
        pair = _pair_key(src, dst)
        if not use_port:
            counts = key_label_counts.setdefault(pair, {})
            counts[label] = counts.get(label, 0) + 1
            continue
        ports = [dport]
        if sport is not None and not pd.isna(sport) and sport < 1024:
            ports.append(sport)
        for port in ports:
            k = f"{pair}#{_port_str(port)}"
            counts = key_label_counts.setdefault(k, {})
            counts[label] = counts.get(label, 0) + 1

    label_keys = {}
    ambiguous_keys = {}
    for key, counts in key_label_counts.items():
        # Pick label with max flow count. Tie-breaker: prefer non-BENIGN, then alphabetical.
        sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0] != BENIGN, x[0]), reverse=True)
        winner = sorted_counts[0][0]
        label_keys[key] = winner
        if len(counts) > 1:
            ambiguous_keys[key] = counts

    # Check how many attack-to-attack ambiguities exist
    attack_attack_ambiguities = {}
    for key, counts in ambiguous_keys.items():
        attack_labels = [l for l in counts.keys() if l != BENIGN]
        if len(attack_labels) > 1:
            attack_attack_ambiguities[key] = counts

    ambiguity_info = {
        "total_keys": len(label_keys),
        "ambiguous_keys_count": len(ambiguous_keys),
        "attack_attack_ambiguities_count": len(attack_attack_ambiguities),
        "ambiguous_details": ambiguous_keys,
        "attack_attack_details": attack_attack_ambiguities,
    }
    return label_keys, ambiguity_info


# Backwards compatibility alias
def build_attack_keys(flows: pd.DataFrame, use_port: bool) -> dict:
    label_keys, _ = build_label_keys(flows, use_port)
    return {k: {v} for k, v in label_keys.items() if v != BENIGN}


RECALL_COLUMNS = ["label", "keys_in_scored", "matching_rows", "flagged_rows",
                  "flagged_pct", "best_score", "best_rank", "best_percentile"]


def recall_report(scored_path: Path, flows: pd.DataFrame, batch_size: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """One row per label (including BENIGN control baseline).
    Two streaming passes over the scored file.
    Returns (report_df, ambiguity_info).
    """
    pf = pq.ParquetFile(scored_path)
    schema = pf.schema_arrow.names
    use_port = "dst_port" in schema and "src_ip" in schema and "dst_ip" in schema
    flag_col = next((c for c in FLAG_CANDIDATES if c in schema), None)

    label_keys, ambiguity_info = build_label_keys(flows, use_port)
    attack_labels = {l for l in flows["Label"].unique() if l != BENIGN}
    if not attack_labels:
        return pd.DataFrame(columns=RECALL_COLUMNS), ambiguity_info

    cols = ["src_ip", "dst_ip", "payl_score"]
    if use_port:
        cols.append("dst_port")
    if flag_col:
        cols.append(flag_col)

    # Pass 1: per-key aggregates for all recognized keys
    key_stats = {}  # key -> {"rows", "flagged", "best"}
    total_rows = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        total_rows += len(df)
        keys = _pair_keys(df["src_ip"], df["dst_ip"])
        if use_port:
            keys = keys + "#" + df["dst_port"].map(_port_str)
        mask = keys.isin(label_keys.keys())
        if not mask.any():
            continue
        sub = df[mask].assign(__key=keys[mask].values)
        agg = {"rows": ("payl_score", "size"), "best": ("payl_score", "max")}
        if flag_col:
            agg["flagged"] = (flag_col, "sum")
        for key, row in sub.groupby("__key").agg(**agg).iterrows():
            s = key_stats.setdefault(key, {"rows": 0, "flagged": 0, "best": float("-inf")})
            s["rows"] += int(row["rows"])
            s["flagged"] += int(row["flagged"]) if flag_col else 0
            s["best"] = max(s["best"], float(row["best"]))

    # Roll key stats up to each single assigned label (no double-counting)
    label_stats = {}
    for key, s in key_stats.items():
        label = label_keys[key]
        ls = label_stats.setdefault(label, {"keys": 0, "rows": 0, "flagged": 0, "best": float("-inf")})
        ls["keys"] += 1
        ls["rows"] += s["rows"]
        ls["flagged"] += s["flagged"]
        ls["best"] = max(ls["best"], s["best"])

    # If no attack labels had any matching rows, return empty report
    if not any(l != BENIGN for l in label_stats):
        return pd.DataFrame(columns=RECALL_COLUMNS), ambiguity_info

    # Pass 2: rank of each label's best row = 1 + number of rows scoring higher.
    greater = {label: 0 for label in label_stats}
    for batch in pf.iter_batches(batch_size=batch_size, columns=["payl_score"]):
        scores = batch.to_pandas()["payl_score"].values
        for label, ls in label_stats.items():
            greater[label] += int((scores > ls["best"]).sum())

    out = []
    # Sort with BENIGN (control) first or last, attacks sorted
    sorted_labels = sorted(label_stats.keys(), key=lambda x: (x == BENIGN, x))
    for label in sorted_labels:
        ls = label_stats[label]
        best_rank = greater[label] + 1
        out.append({
            "label": f"{label} (control)" if label == BENIGN else label,
            "keys_in_scored": ls["keys"],
            "matching_rows": ls["rows"],
            "flagged_rows": ls["flagged"] if flag_col else None,
            "flagged_pct": round(100 * ls["flagged"] / ls["rows"], 2) if (flag_col and ls["rows"]) else None,
            "best_score": round(ls["best"], 3),
            "best_rank": best_rank,
            "best_percentile": round(100 * best_rank / total_rows, 4) if total_rows else None,
        })
    return pd.DataFrame(out, columns=RECALL_COLUMNS), ambiguity_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored-parquet", required=True, type=Path)
    parser.add_argument("--flow-csv", required=True, action="append",
                        help="A CICIDS2017 GeneratedLabelledFlows CSV. Repeat for multiple files.")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional CSV path for the top-N results.")
    parser.add_argument("--skip-recall", action="store_true",
                        help="Skip the recall pass (it re-reads the whole scored file twice).")
    args = parser.parse_args()

    print("=" * 70)
    print("PAYL SPOT-CHECK (heuristic host-pair match against CICIDS2017 labels)")
    print("This is NOT a rigorous evaluation. See script docstring for caveats.")
    print("=" * 70)

    print(f"\n[1/4] Reading top {args.top_n} distinct anomalies from {args.scored_parquet} ...")
    anomalies = top_n_anomalies(args.scored_parquet, args.top_n, args.batch_size)
    print(f"      got {len(anomalies)} rows.")

    print(f"\n[2/4] Loading {len(args.flow_csv)} flow-label CSV(s) ...")
    flows = load_all_flows(args.flow_csv)
    print(f"      {len(flows)} total labeled flow rows loaded.")
    for label, count in flows["Label"].value_counts().items():
        print(f"        {label}: {count}")

    print("\n[3/4] Matching top anomalies to flow labels ...")
    results = spot_check(anomalies, flows)
    n_total = len(results)
    n_matched = int((results["n_matching_flows"] > 0).sum())
    n_attack = int((results["n_attack_flows"] > 0).sum())
    print(f"      {n_matched}/{n_total} top anomalies matched at least one labeled flow.")
    print(f"      {n_attack}/{n_total} matched at least one NON-BENIGN flow.")
    print("\n" + results.to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote top-N results to {args.output}")

    if args.skip_recall:
        print("\n[4/4] Recall pass skipped (--skip-recall).")
        return

    print("\n[4/4] Recall: scored rows involving each label's host pairs and ports ...")
    recall, ambiguity = recall_report(args.scored_parquet, flows, args.batch_size)
    if ambiguity["ambiguous_keys_count"] > 0:
        print(f"\n      [KEY AMBIGUITY REPORT]")
        print(f"      Total unique (host_pair, port) keys: {ambiguity['total_keys']}")
        print(f"      Ambiguous keys (seen with >1 label): {ambiguity['ambiguous_keys_count']}")
        print(f"      Attack-vs-Attack collisions:         {ambiguity['attack_attack_ambiguities_count']}")
        if ambiguity["attack_attack_ambiguities_count"] > 0:
            print("      Sample attack collisions:")
            for k, counts in list(ambiguity["attack_attack_details"].items())[:5]:
                print(f"        {k} => {counts}")
        print("      Each key was assigned to the label with majority flow count (no double-counting).\n")

    if recall.empty:
        print("      No scored rows involve any attack host pair and port.")
        print("      => PAYL did not score any traffic from the labeled attack keys.")
    else:
        print("\n" + recall.to_string(index=False))
        print("\nBENIGN (control) = baseline flag rate on benign host pairs/ports using identical key logic.")
        print("flagged_* = rows PAYL flagged (payl_anomaly). Blank if the file has no flag column.")
        print("best_rank = where the single highest-scoring row for that label ranks in the")
        print("WHOLE scored file (1 = the most anomalous row). Low percentile = PAYL ranks")
        print("that attack's traffic near the top.")


if __name__ == "__main__":
    main()
