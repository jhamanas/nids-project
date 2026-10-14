#!/usr/bin/env python3
"""
Generate a small synthetic PCAP fixture for testing the Phase 2 parser.

This is NOT a substitute for real CICIDS2017 data — it exists purely so the
parsing code (preprocessing/pcap_parser.py) can be exercised and unit-tested
before the real (registration-gated) PCAPs are downloaded. It builds a
handful of synthetic flows:
  - An HTTP GET split across multiple TCP segments (tests session reassembly)
  - An HTTP response
  - A couple of DNS queries (UDP)
  - One payload containing a repeated byte pattern resembling a NOP-sled /
    buffer-overflow attempt, tagged for use as a sanity check that payload
    bytes come through unmodified (not meant to simulate a real exploit)

Usage:
    python scripts/generate_test_pcap.py
    -> writes tests/fixtures/sample.pcap
"""

from pathlib import Path

from scapy.all import wrpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from scapy.packet import Raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample.pcap"


def build_packets():
    packets = []
    t = 1000.0

    # --- Flow 1: HTTP GET request split across 2 TCP segments (client -> server:80) ---
    client_ip, server_ip = "10.0.0.5", "93.184.216.34"
    client_port, server_port = 51000, 80

    part1 = b"GET /index.html HTTP/1.1\r\nHost: "
    part2 = b"example.com\r\nUser-Agent: test-agent\r\n\r\n"

    seq = 1000
    pkt1 = IP(src=client_ip, dst=server_ip) / TCP(
        sport=client_port, dport=server_port, seq=seq, flags="PA"
    ) / Raw(load=part1)
    pkt1.time = t; t += 0.01
    packets.append(pkt1)

    seq2 = seq + len(part1)
    pkt2 = IP(src=client_ip, dst=server_ip) / TCP(
        sport=client_port, dport=server_port, seq=seq2, flags="PA"
    ) / Raw(load=part2)
    pkt2.time = t; t += 0.01
    packets.append(pkt2)

    # A duplicate/retransmit of segment 1 (tests overlap-skipping in reassembly)
    pkt1_retransmit = IP(src=client_ip, dst=server_ip) / TCP(
        sport=client_port, dport=server_port, seq=seq, flags="PA"
    ) / Raw(load=part1)
    pkt1_retransmit.time = t; t += 0.01
    packets.append(pkt1_retransmit)

    # --- Flow 1 response: HTTP response (server -> client), single segment ---
    resp_body = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\n\r\n"
        b"<html></html>"
    )
    resp_seq = 5000
    pkt3 = IP(src=server_ip, dst=client_ip) / TCP(
        sport=server_port, dport=client_port, seq=resp_seq, flags="PA"
    ) / Raw(load=resp_body)
    pkt3.time = t; t += 0.01
    packets.append(pkt3)

    # --- Flow 2: DNS query (UDP, port 53) ---
    dns_client = "10.0.0.6"
    dns_pkt = IP(src=dns_client, dst="8.8.8.8") / UDP(sport=52000, dport=53) / \
        DNS(rd=1, qd=DNSQR(qname="example.com"))
    dns_pkt.time = t; t += 0.01
    packets.append(dns_pkt)

    # --- Flow 3: payload with a repeated-byte pattern (anomalous-looking, for
    #     testing byte-level feature extraction later in Phase 3 — NOT a real
    #     exploit, just structurally distinct from the HTTP text above) ---
    anomalous_client, anomalous_server = "10.0.0.7", "192.168.1.50"
    anomalous_payload = b"\x90" * 40 + b"\xde\xad\xbe\xef" * 10
    pkt4 = IP(src=anomalous_client, dst=anomalous_server) / TCP(
        sport=51500, dport=8080, seq=2000, flags="PA"
    ) / Raw(load=anomalous_payload)
    pkt4.time = t; t += 0.01
    packets.append(pkt4)

    # --- Flow 4: FTP control channel (port 21), simple single-packet payload ---
    ftp_pkt = IP(src="10.0.0.8", dst="192.168.1.60") / TCP(
        sport=51600, dport=21, seq=3000, flags="PA"
    ) / Raw(load=b"USER anonymous\r\n")
    ftp_pkt.time = t; t += 0.01
    packets.append(ftp_pkt)

    return packets


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    packets = build_packets()
    wrpcap(str(OUTPUT_PATH), packets)
    print(f"Wrote {len(packets)} packets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
