"""Tests for scripts/spot_check_labels.py using synthetic fixtures that mirror
the real CICIDS2017 GeneratedLabelledFlows schema (leading-space column names)
and the scored parquet schema from the PAYL scorer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spot_check_labels as sc  # noqa: E402


# ---------------------------------------------------------------- fixtures

FLOW_HEADER = [
    "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
    " Protocol", " Timestamp", " Flow Duration", " Label",
]


def _write_flow_csv(path: Path, rows):
    """Writes like CICFlowMeter does: leading-space column names."""
    pd.DataFrame(rows, columns=FLOW_HEADER).to_csv(path, index=False)
    return path


@pytest.fixture
def flow_csvs(tmp_path):
    morning = _write_flow_csv(tmp_path / "Friday-WorkingHours-Morning.pcap_ISCX.csv", [
        {"Flow ID": "10.0.0.5-1.2.3.4-1234-80-6", " Source IP": "10.0.0.5", " Source Port": 1234,
         " Destination IP": "1.2.3.4", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "7/7/2017 9:01", " Flow Duration": 100, " Label": "BENIGN"},
    ])
    portscan = _write_flow_csv(tmp_path / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", [
        {"Flow ID": "x", " Source IP": "172.16.0.1", " Source Port": 50001,
         " Destination IP": "192.168.10.50", " Destination Port": 22, " Protocol": 6,
         " Timestamp": "7/7/2017 14:00", " Flow Duration": 5, " Label": "PortScan"},
        {"Flow ID": "y", " Source IP": "172.16.0.1", " Source Port": 50002,
         " Destination IP": "192.168.10.50", " Destination Port": 23, " Protocol": 6,
         " Timestamp": "7/7/2017 14:00", " Flow Duration": 5, " Label": "PortScan"},
    ])
    ddos = _write_flow_csv(tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", [
        {"Flow ID": "z", " Source IP": "172.16.0.9", " Source Port": 40000,
         " Destination IP": "192.168.10.50", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "7/7/2017 15:10", " Flow Duration": 900, " Label": "DDoS"},
    ])
    return [morning, portscan, ddos]


@pytest.fixture
def scored_parquet(tmp_path):
    """Mirrors the real scored parquet's identifying columns (no flag column here)."""
    rows = [
        {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "src_port": 50001, "dst_port": 22,
         "protocol": 6, "timestamp": "2017-07-07 14:00:01", "service": "other",
         "payload_len": 1, "payl_score": 674.0},
        {"src_ip": "172.16.0.9", "dst_ip": "192.168.10.50", "src_port": 40000, "dst_port": 80,
         "protocol": 6, "timestamp": "2017-07-07 15:10:05", "service": "http",
         "payload_len": 1400, "payl_score": 418.0},
        {"src_ip": "8.8.8.8", "dst_ip": "10.0.0.99", "src_port": 53, "dst_port": 53000,
         "protocol": 17, "timestamp": "2017-07-07 10:00:00", "service": "dns",
         "payload_len": 80, "payl_score": 3.1},
        {"src_ip": "10.0.0.5", "dst_ip": "1.2.3.4", "src_port": 1234, "dst_port": 80,
         "protocol": 6, "timestamp": "2017-07-07 09:01:00", "service": "http",
         "payload_len": 300, "payl_score": 5.0},
    ]
    path = tmp_path / "scored_test.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------- top-N

def test_top_n_returns_highest_scores_in_order(scored_parquet):
    top = sc.top_n_anomalies(scored_parquet, n=2)
    assert list(top["payl_score"]) == [674.0, 418.0]


def test_top_n_bounded_streaming_matches_full_dedupe(tmp_path):
    """Trimming path must give the same answer as a full sort + dedupe."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1000, size=5000)
    df = pd.DataFrame({
        "src_ip": [f"10.0.{i % 200}.{i % 250}" for i in range(5000)],
        "dst_ip": ["1.1.1.1"] * 5000,
        "payl_score": scores,
    })
    path = tmp_path / "big.parquet"
    df.to_parquet(path, index=False)
    got = sc.top_n_anomalies(path, n=25, batch_size=300)
    expected = (df.groupby(["src_ip", "dst_ip"])["payl_score"].max()
                  .nlargest(25).sort_values(ascending=False).values)
    np.testing.assert_allclose(got["payl_score"].values, expected)
    assert not got.duplicated(subset=["src_ip", "dst_ip"]).any()


def test_top_n_dedupes_same_host_and_port(tmp_path):
    """One chatty session (same src/dst/dst_port, many rows) must count once."""
    rows = [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "dst_port": 80, "payl_score": s}
            for s in (900.0, 850.0, 800.0, 750.0)]
    rows.append({"src_ip": "3.3.3.3", "dst_ip": "2.2.2.2", "dst_port": 80, "payl_score": 500.0})
    rows.append({"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "dst_port": 443, "payl_score": 600.0})
    path = tmp_path / "chatty.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    top = sc.top_n_anomalies(path, n=3)
    assert list(top["payl_score"]) == [900.0, 600.0, 500.0]


# ---------------------------------------------------------------- flow loading

def test_load_flow_csv_strips_leading_spaces(flow_csvs):
    df = sc.load_flow_csv(flow_csvs[1])
    assert {"Source IP", "Destination IP", "Destination Port", "Label"} <= set(df.columns)
    assert set(df["Label"]) == {"PortScan"}


def test_label_whitespace_is_stripped(tmp_path):
    p = tmp_path / "ws.csv"
    pd.DataFrame([{"Flow ID": "q", " Source IP": "1.1.1.1", " Source Port": 1,
                   " Destination IP": "2.2.2.2", " Destination Port": 2, " Protocol": 6,
                   " Timestamp": "x", " Flow Duration": 1, " Label": "DDoS  "}],
                 columns=FLOW_HEADER).to_csv(p, index=False)
    assert sc.load_flow_csv(p)["Label"].iloc[0] == "DDoS"


# ---------------------------------------------------------------- top-N matching

def test_direct_direction_match(flow_csvs, scored_parquet):
    flows = sc.load_all_flows(flow_csvs)
    top = sc.top_n_anomalies(scored_parquet, n=1)
    res = sc.spot_check(top, flows)
    # Anomaly is on dst_port 22, so port narrowing keeps only the port-22 flow.
    assert res.iloc[0]["labels_found"] == "PortScan (x1)"
    assert res.iloc[0]["n_matching_flows"] == 1


def test_reversed_direction_match(flow_csvs, scored_parquet):
    flows = sc.load_all_flows(flow_csvs)
    top = sc.top_n_anomalies(scored_parquet, n=2)
    ddos_row = top[top["payl_score"] == 418.0].iloc[0]
    res = sc.find_matches(ddos_row, flows)
    assert len(res) == 1
    assert res.iloc[0]["Label"] == "DDoS"


def test_no_match_reported_honestly(flow_csvs):
    flows = sc.load_all_flows(flow_csvs)
    bad = pd.DataFrame([{"src_ip": "8.8.8.8", "dst_ip": "10.0.0.99", "src_port": 53,
                         "dst_port": 53000, "service": "dns", "payl_score": 3.1}])
    res = sc.spot_check(bad, flows)
    assert res.iloc[0]["labels_found"] == "NO MATCH"
    assert res.iloc[0]["n_matching_flows"] == 0


def test_port_narrowing_falls_back_to_ip_pair(flow_csvs):
    flows = sc.load_all_flows(flow_csvs)
    anomaly = pd.Series({"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50",
                         "src_port": 9999, "dst_port": 9999, "payl_score": 1.0})
    assert len(sc.find_matches(anomaly, flows)) == 2


def test_spot_check_reports_attack_flow_counts(flow_csvs, scored_parquet):
    flows = sc.load_all_flows(flow_csvs)
    top = sc.top_n_anomalies(scored_parquet, n=2)
    res = sc.spot_check(top, flows).set_index("payl_score")
    assert res.loc[674.0, "n_attack_flows"] == 1
    assert res.loc[418.0, "n_attack_flows"] == 1
    assert res.loc[418.0, "n_matching_flows"] == 1


def test_benign_only_match_has_zero_attack_flows(flow_csvs):
    flows = sc.load_all_flows(flow_csvs)
    anomaly = pd.DataFrame([{"src_ip": "10.0.0.5", "dst_ip": "1.2.3.4",
                             "dst_port": 80, "payl_score": 5.0}])
    res = sc.spot_check(anomaly, flows)
    assert res.iloc[0]["labels_found"] == "BENIGN (x1)"
    assert res.iloc[0]["n_attack_flows"] == 0


# ---------------------------------------------------------------- recall

def test_recall_ranks_each_attack_label(flow_csvs, scored_parquet):
    flows = sc.load_all_flows(flow_csvs)
    rec, ambiguity = sc.recall_report(scored_parquet, flows)
    rec = rec.set_index("label")
    assert "PortScan" in rec.index
    assert "DDoS" in rec.index
    assert "BENIGN (control)" in rec.index

    # PortScan (dst 22) scored 674 -> top row of the whole file
    assert rec.loc["PortScan", "matching_rows"] == 1
    assert rec.loc["PortScan", "best_rank"] == 1
    assert rec.loc["PortScan", "best_percentile"] == 25.0  # 1 of 4 rows

    # DDoS (recorded reversed: flow src port 80) scored 418 -> second-highest
    assert rec.loc["DDoS", "matching_rows"] == 1
    assert rec.loc["DDoS", "best_rank"] == 2


def test_recall_separates_labels_that_share_a_host_pair(tmp_path, flow_csvs):
    """PortScan and DDoS can share a host pair. Port-level keys must keep their
    rows apart, otherwise both labels double-count the same rows."""
    rows = [
        {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "dst_port": 22, "payl_score": 100.0, "payl_anomaly": True},
        {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "dst_port": 80, "payl_score": 50.0, "payl_anomaly": False},
    ]
    path = tmp_path / "shared.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    # Add a DDoS flow on the SAME pair, port 80 (not reversed).
    extra = _write_flow_csv(tmp_path / "shared-ddos.pcap_ISCX.csv", [
        {"Flow ID": "s", " Source IP": "172.16.0.1", " Source Port": 55555,
         " Destination IP": "192.168.10.50", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "DDoS"},
    ])
    flows = sc.load_all_flows(flow_csvs + [extra])
    rec, ambiguity = sc.recall_report(path, flows)
    rec = rec.set_index("label")
    assert rec.loc["PortScan", "matching_rows"] == 1   # port 22 row only
    assert rec.loc["DDoS", "matching_rows"] == 1       # port 80 row only
    assert rec.loc["PortScan", "best_score"] == 100.0
    assert rec.loc["DDoS", "best_score"] == 50.0


def test_recall_uses_payl_anomaly_flag_column(tmp_path, flow_csvs):
    """Real scored file uses payl_anomaly. No dst_port here -> pair-only keys."""
    rows = [
        {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "payl_score": 674.0, "payl_anomaly": True},
        {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "payl_score": 10.0, "payl_anomaly": False},
        {"src_ip": "8.8.8.8", "dst_ip": "10.0.0.99", "payl_score": 3.0, "payl_anomaly": False},
    ]
    path = tmp_path / "flagged.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    flows = sc.load_all_flows(flow_csvs)
    rec, ambiguity = sc.recall_report(path, flows)
    rec = rec.set_index("label")
    assert rec.loc["PortScan", "matching_rows"] == 2
    assert rec.loc["PortScan", "flagged_rows"] == 1
    assert rec.loc["PortScan", "flagged_pct"] == 50.0


def test_recall_flag_is_blank_without_flag_column(flow_csvs, scored_parquet):
    flows = sc.load_all_flows(flow_csvs)
    rec, ambiguity = sc.recall_report(scored_parquet, flows)
    assert rec["flagged_rows"].isna().all()


def test_recall_with_no_attack_flows_is_empty(scored_parquet, flow_csvs):
    benign_only = sc.load_flow_csv(flow_csvs[0])
    rec, ambiguity = sc.recall_report(scored_parquet, benign_only)
    assert rec.empty
    assert list(rec.columns) == sc.RECALL_COLUMNS


def test_build_label_keys_majority_vote_resolves_collision(tmp_path):
    """If a key has 2 PortScan flows and 1 DDoS flow, it must resolve to PortScan."""
    csv_path = _write_flow_csv(tmp_path / "colliding.csv", [
        {"Flow ID": "1", " Source IP": "10.0.0.1", " Source Port": 1234,
         " Destination IP": "10.0.0.2", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "PortScan"},
        {"Flow ID": "2", " Source IP": "10.0.0.1", " Source Port": 1235,
         " Destination IP": "10.0.0.2", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "PortScan"},
        {"Flow ID": "3", " Source IP": "10.0.0.1", " Source Port": 1236,
         " Destination IP": "10.0.0.2", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "DDoS"},
    ])
    flows = sc.load_flow_csv(csv_path)
    label_keys, ambiguity = sc.build_label_keys(flows, use_port=True)
    assert label_keys["10.0.0.1->10.0.0.2#80"] == "PortScan"
    assert ambiguity["attack_attack_ambiguities_count"] == 1


def test_ephemeral_source_port_is_not_indexed(tmp_path):
    """Ephemeral source port (>1024) must not become a key, but destination port must."""
    csv_path = _write_flow_csv(tmp_path / "ephemeral.csv", [
        {"Flow ID": "1", " Source IP": "172.16.0.1", " Source Port": 50001,
         " Destination IP": "192.168.10.50", " Destination Port": 22, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "PortScan"},
    ])
    flows = sc.load_flow_csv(csv_path)
    label_keys, _ = sc.build_label_keys(flows, use_port=True)
    assert "172.16.0.1->192.168.10.50#50001" not in label_keys
    assert "172.16.0.1->192.168.10.50#22" in label_keys


def test_reply_side_service_port_is_indexed(tmp_path):
    """Source port below 1024 on an attack flow is also indexed as a directional key."""
    csv_path = _write_flow_csv(tmp_path / "reply.csv", [
        {"Flow ID": "1", " Source IP": "192.168.10.50", " Source Port": 80,
         " Destination IP": "172.16.0.9", " Destination Port": 40000, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "DDoS"},
    ])
    flows = sc.load_flow_csv(csv_path)
    label_keys, _ = sc.build_label_keys(flows, use_port=True)
    assert "192.168.10.50->172.16.0.9#80" in label_keys


def test_collision_count_drops_when_ephemeral_ports_ignored(tmp_path):
    """Two flows on different destinations sharing an ephemeral source port must not collide."""
    csv_path = _write_flow_csv(tmp_path / "no_collision.csv", [
        {"Flow ID": "1", " Source IP": "10.0.0.1", " Source Port": 54321,
         " Destination IP": "10.0.0.2", " Destination Port": 80, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "PortScan"},
        {"Flow ID": "2", " Source IP": "10.0.0.1", " Source Port": 54321,
         " Destination IP": "10.0.0.3", " Destination Port": 443, " Protocol": 6,
         " Timestamp": "t", " Flow Duration": 1, " Label": "DDoS"},
    ])
    flows = sc.load_flow_csv(csv_path)
    _, ambiguity = sc.build_label_keys(flows, use_port=True)
    assert ambiguity["ambiguous_keys_count"] == 0


