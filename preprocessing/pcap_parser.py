"""
Phase 2 — Packet parsing & payload extraction.

Turns raw PCAP files into structured payload records ready for feature
engineering (Phase 3). Two extraction modes are provided:

  - extract_packet_payloads(): one record per packet with a non-empty payload.
    Simple, fast, memory-light (streams the file). Good default, and what
    PAYL's original paper actually operates on (per-packet payloads,
    segmented by length bucket).

  - extract_session_payloads(): reassembles each TCP stream (by 5-tuple +
    direction) into a single concatenated payload per direction. Slower and
    holds a session's packets in memory until the stream ends, but gives you
    the full request/response body rather than individual segments — useful
    if you want to look at, e.g., a complete HTTP request rather than
    whichever MTU-sized slice of it happened to land in one packet.

Known simplification (documented, not hidden): session reassembly here
handles the common case (in-order or lightly reordered segments) by sorting
on TCP sequence number and skipping fully-overlapping duplicates. It does
NOT handle every real-world TCP edge case (e.g. partial-overlap retransmits
with different content, PAWS/sequence wraparound on very long streams). For
a research/coursework NIDS project this is a standard, acceptable
simplification — production reassembly usually reuses a library (e.g. Zeek)
rather than hand-rolling this.
"""

from collections import defaultdict
from pathlib import Path
from typing import Iterator
import time

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw


def classify_service(port: int, services: dict) -> str:
    """Map a port number to a service name from config, else 'other'."""
    for name, svc_port in services.items():
        if port == svc_port:
            return name
    return "other"


def iter_packets(pcap_path: Path) -> Iterator:
    """Stream packets from a PCAP file one at a time (memory-light for large files)."""
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            yield pkt


def extract_packet_payloads(pcap_path: Path, config: dict, limit: int = None,
                             progress_every: int = None) -> list[dict]:
    """
    Per-packet payload extraction.

    Returns a list of dicts, one per packet that has IP + (TCP or UDP) + a
    non-empty Raw payload:
        {
            "pcap_file": str, "timestamp": float,
            "src_ip": str, "dst_ip": str, "src_port": int, "dst_port": int,
            "protocol": "tcp" | "udp", "service": str,
            "payload": bytes, "payload_len": int,
        }

    Set progress_every (e.g. 100_000) to print a status line every N packets
    read -- useful for large real-world PCAPs (multi-GB CICIDS2017 files can
    take many minutes; with no feedback, a slow-but-working run is
    indistinguishable from a hung one).
    """
    services = config["services"]
    max_len = config["features"]["max_payload_bytes"]
    records = []
    t_start = time.time()

    for i, pkt in enumerate(iter_packets(pcap_path)):
        if limit is not None and i >= limit:
            break
        if progress_every and i > 0 and i % progress_every == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"       ... {i:,} packets read ({rate:,.0f} pkt/s, "
                  f"{len(records):,} payloads extracted so far)")
        if IP not in pkt or Raw not in pkt:
            continue

        if TCP in pkt:
            proto = "tcp"
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            proto = "udp"
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            continue

        payload = bytes(pkt[Raw].load)[:max_len]
        if not payload:
            continue

        # Service is classified off whichever port is the "well-known" side —
        # try dst_port first (typical for client->server), fall back to src_port
        service = classify_service(dport, services)
        if service == "other":
            service = classify_service(sport, services)

        records.append({
            "pcap_file": pcap_path.name,
            "timestamp": float(pkt.time),
            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,
            "src_port": sport,
            "dst_port": dport,
            "protocol": proto,
            "service": service,
            "payload": payload,
            "payload_len": len(payload),
        })

    return records


def extract_session_payloads(pcap_path: Path, config: dict, limit: int = None,
                              progress_every: int = None) -> list[dict]:
    """
    TCP-session-level payload extraction: reassembles each (5-tuple, direction)
    stream into one concatenated payload.

    Returns a list of dicts, one per (session, direction):
        {
            "pcap_file": str, "session_id": str, "direction": "fwd" | "rev",
            "src_ip": str, "dst_ip": str, "src_port": int, "dst_port": int,
            "protocol": "tcp", "service": str,
            "payload": bytes, "payload_len": int, "packet_count": int,
            "start_time": float, "end_time": float,
        }

    Set progress_every (e.g. 100_000) to print a status line every N packets
    read -- see extract_packet_payloads for why this matters on large files.
    Note this function additionally holds all in-progress sessions in memory
    until the whole file is read (it can't flush a session until it knows no
    more packets belong to it), so it uses more memory than the packet-level
    function for the same file size.
    """
    services = config["services"]
    max_len = config["features"]["max_payload_bytes"]
    t_start = time.time()

    # Key: canonical 5-tuple (sorted ip:port pair) + which side is "fwd" (the
    # side that sent the first packet in the flow) — so both directions of
    # the same TCP connection group under one session_id.
    sessions: dict[tuple, dict] = defaultdict(lambda: {
        "fwd": {}, "rev": {}, "fwd_key": None, "meta": None
    })

    for i, pkt in enumerate(iter_packets(pcap_path)):
        if limit is not None and i >= limit:
            break
        if progress_every and i > 0 and i % progress_every == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"       ... {i:,} packets read ({rate:,.0f} pkt/s, "
                  f"{len(sessions):,} sessions tracked so far)")
        if IP not in pkt or TCP not in pkt or Raw not in pkt:
            continue

        payload = bytes(pkt[Raw].load)
        if not payload:
            continue

        ip, tcp = pkt[IP], pkt[TCP]
        a = (ip.src, tcp.sport)
        b = (ip.dst, tcp.dport)
        canonical = tuple(sorted([a, b]))
        session_key = canonical

        sess = sessions[session_key]
        if sess["fwd_key"] is None:
            sess["fwd_key"] = a  # first packet's source defines "forward" direction

        direction = "fwd" if a == sess["fwd_key"] else "rev"
        seq = tcp.seq
        sess[direction][seq] = payload[:max_len]

        if sess["meta"] is None:
            sess["meta"] = {
                "src_ip": ip.src, "dst_ip": ip.dst,
                "src_port": tcp.sport, "dst_port": tcp.dport,
            }
        sess.setdefault("start_time", float(pkt.time))
        sess["end_time"] = float(pkt.time)
        # Track packet counts per-direction (not per-session) so each output
        # record's packet_count reflects only the packets that contributed to it.
        pkt_counts = sess.setdefault("packet_counts", {"fwd": 0, "rev": 0})
        pkt_counts[direction] += 1

    records = []
    for session_key, sess in sessions.items():
        meta = sess["meta"]
        service = classify_service(meta["dst_port"], services)
        if service == "other":
            service = classify_service(meta["src_port"], services)

        for direction in ("fwd", "rev"):
            segments = sess[direction]
            if not segments:
                continue
            # Reassemble: sort by seq number, concatenate, skip byte ranges
            # already covered (handles simple retransmit duplicates).
            ordered_seqs = sorted(segments.keys())
            reassembled = bytearray()
            covered_up_to = None
            for seq in ordered_seqs:
                chunk = segments[seq]
                if covered_up_to is not None and seq < covered_up_to:
                    # Fully or partially overlapping with what we already have —
                    # skip the overlapping prefix, keep any new tail bytes.
                    overlap = covered_up_to - seq
                    if overlap < len(chunk):
                        reassembled.extend(chunk[overlap:])
                        covered_up_to = seq + len(chunk)
                    continue
                reassembled.extend(chunk)
                covered_up_to = seq + len(chunk)

            payload = bytes(reassembled)[:max_len]
            if not payload:
                continue

            src_ip, dst_ip = (meta["src_ip"], meta["dst_ip"]) if direction == "fwd" \
                else (meta["dst_ip"], meta["src_ip"])
            src_port, dst_port = (meta["src_port"], meta["dst_port"]) if direction == "fwd" \
                else (meta["dst_port"], meta["src_port"])

            records.append({
                "pcap_file": pcap_path.name,
                "session_id": f"{session_key[0][0]}:{session_key[0][1]}-"
                              f"{session_key[1][0]}:{session_key[1][1]}",
                "direction": direction,
                "src_ip": src_ip, "dst_ip": dst_ip,
                "src_port": src_port, "dst_port": dst_port,
                "protocol": "tcp",
                "service": service,
                "payload": payload,
                "payload_len": len(payload),
                "packet_count": sess["packet_counts"][direction],
                "start_time": sess["start_time"],
                "end_time": sess["end_time"],
            })

    return records
