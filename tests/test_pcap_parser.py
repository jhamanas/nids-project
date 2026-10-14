"""
Tests for preprocessing/pcap_parser.py, run against the synthetic fixture at
tests/fixtures/sample.pcap (generate it first with scripts/generate_test_pcap.py
if it's missing).
"""

from pathlib import Path

import pytest

from preprocessing.pcap_parser import extract_packet_payloads, extract_session_payloads
from preprocessing.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample.pcap"


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def ensure_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"{FIXTURE_PATH} not found — run: python scripts/generate_test_pcap.py"
        )
    return FIXTURE_PATH


# ---------- Packet-level extraction ----------

def test_packet_level_extracts_all_payload_packets(ensure_fixture, config):
    records = extract_packet_payloads(ensure_fixture, config)
    # 7 packets built by generate_test_pcap.py, all carry a Raw payload
    # (DNS packet's payload is inside the DNS layer, not Raw -- so 6 expected)
    assert len(records) == 6


def test_packet_level_service_classification(ensure_fixture, config):
    records = extract_packet_payloads(ensure_fixture, config)
    services = {r["service"] for r in records}
    assert "http" in services
    assert "ftp" in services
    # anomalous flow uses port 8080, not in config's known services -> "other"
    assert "other" in services


def test_packet_level_payload_bytes_preserved(ensure_fixture, config):
    records = extract_packet_payloads(ensure_fixture, config)
    ftp_records = [r for r in records if r["service"] == "ftp"]
    assert len(ftp_records) == 1
    assert ftp_records[0]["payload"] == b"USER anonymous\r\n"


def test_packet_level_respects_limit(ensure_fixture, config):
    records = extract_packet_payloads(ensure_fixture, config, limit=2)
    assert len(records) <= 2


# ---------- Session-level extraction (TCP reassembly) ----------

def test_session_level_reassembles_split_http_request(ensure_fixture, config):
    records = extract_session_payloads(ensure_fixture, config)
    http_fwd = [r for r in records if r["service"] == "http" and r["direction"] == "fwd"]
    assert len(http_fwd) == 1

    payload = http_fwd[0]["payload"]
    expected = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: test-agent\r\n\r\n"
    assert payload == expected, (
        f"Reassembled payload did not match expected request.\nGot: {payload!r}"
    )


def test_session_level_skips_retransmit_duplicate(ensure_fixture, config):
    # The fixture includes a duplicate retransmit of the first HTTP segment.
    # Reassembly should not duplicate those bytes in the output.
    records = extract_session_payloads(ensure_fixture, config)
    http_fwd = [r for r in records if r["service"] == "http" and r["direction"] == "fwd"][0]
    assert b"GET /index.html HTTP/1.1\r\nHost: GET" not in http_fwd["payload"]


def test_session_level_response_direction(ensure_fixture, config):
    records = extract_session_payloads(ensure_fixture, config)
    http_rev = [r for r in records if r["service"] == "http" and r["direction"] == "rev"]
    assert len(http_rev) == 1
    assert http_rev[0]["payload"].startswith(b"HTTP/1.1 200 OK")


def test_session_level_packet_counts(ensure_fixture, config):
    records = extract_session_payloads(ensure_fixture, config)
    http_fwd = [r for r in records if r["service"] == "http" and r["direction"] == "fwd"][0]
    # 2 real segments + 1 retransmit = 3 packets contributed to this direction
    assert http_fwd["packet_count"] == 3


def test_session_level_anomalous_payload_preserved(ensure_fixture, config):
    records = extract_session_payloads(ensure_fixture, config)
    anomalous = [r for r in records if r["dst_port"] == 8080]
    assert len(anomalous) == 1
    payload = anomalous[0]["payload"]
    assert payload.startswith(b"\x90" * 40)
    assert payload.count(b"\xde\xad\xbe\xef") == 10
