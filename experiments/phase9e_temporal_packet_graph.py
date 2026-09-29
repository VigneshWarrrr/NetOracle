"""Phase 9E: Temporal packet graph construction and feasibility characterization.

Builds and rigorously measures a temporal communication graph from the single
validated PCAP member retrieved in Phase 9D (`pcap/UCAP172.31.69.22`, already
decompressed at `experiments/data/phase9d/UCAP172.31.69.22.pcap`). No network
access occurs in this phase -- the file is read locally only.

Scope is strictly limited to constructing and characterizing the graph
representation. No GNN, no transformer, no counterfactual simulation, no
attack labeling, and no CSV row-level matching are performed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.utils import PcapReader

try:
    from scapy.layers.inet6 import IPv6
except Exception:  # pragma: no cover - optional layer
    IPv6 = None

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

PCAP_PATH = EXPERIMENTS_DIR / "data" / "phase9d" / "UCAP172.31.69.22.pcap"
PHASE9D_REPORT_PATH = (
    EXPERIMENTS_DIR / "results" / "phase9d_pcap_validation" / "validation_report.json"
)
# Matches the path convention established in
# experiments/phase9d_pcap_validation.py (Path(__file__).resolve().parents[2]),
# i.e. one directory above the NetOracle project root, not inside it.
CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "CSE-CIC-IDS2018"
    / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
)

DATA_DIR = EXPERIMENTS_DIR / "data" / "phase9e"
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "phase9e_temporal_packet_graph"

# Reused directly from feature_engine/temporal_features.py (TemporalFeatureEngine
# default) and feature_engine/graph_builder.py (build_dynamic_graphs default),
# and matches scripts/build_phase6a_graph_dataset.py's WINDOW_SECONDS constant.
# This is an established, project-wide convention -- not a new choice.
WINDOW_SECONDS = 10

# Fixed protocol vocabulary used for fixed-width aggregation. "OTHER" covers any
# IP packet that is neither TCP, UDP, nor ICMP (none observed in this capture,
# kept for correctness rather than assumed absent).
PROTOCOL_VOCAB = ("TCP", "UDP", "ICMP", "OTHER")

# Domain-separation string for deterministic pseudonymization. This is NOT a
# secrecy mechanism -- IPv4 address space (2^32) is trivially brute-forceable
# against a known salt via SHA-256, so this provides consistent internal
# identifiers only, not privacy protection. Documented as a limitation below.
ANON_SALT = "netoracle-phase9e-pseudonymization-v1"

MAX_ARTIFACT_BYTES_BEFORE_SUMMARIZING = 5 * 1024 * 1024

TCP_FLAG_BITS = {"FIN": 0x01, "SYN": 0x02, "RST": 0x04, "ACK": 0x10}


# ---------------------------------------------------------------------------
# Step 1: existing conventions
# ---------------------------------------------------------------------------


def conventions_report() -> dict:
    return {
        "reused": [
            {
                "convention": "10-second temporal window",
                "source": (
                    "feature_engine/temporal_features.py "
                    "(TemporalFeatureEngine.__init__ default window_seconds=10), "
                    "feature_engine/graph_builder.py "
                    "(build_dynamic_graphs default window_seconds=10), "
                    "scripts/build_phase6a_graph_dataset.py (WINDOW_SECONDS = 10)"
                ),
                "note": (
                    "Confirmed as an established, project-wide convention across "
                    "three independent locations; reused verbatim as WINDOW_SECONDS "
                    "= 10 rather than re-deriving a new value."
                ),
            },
            {
                "convention": "canonical packet field naming",
                "source": "ingestion/pcap_reader.py (PcapReaderService._packet_to_event)",
                "note": (
                    "Field names (src_ip, dst_ip, protocol, ttl, tcp_flags, "
                    "tcp_window_size, byte_count) mirror this existing schema "
                    "where the underlying quantity is the same."
                ),
            },
            {
                "convention": "deterministic SHA-256 hashing for stable identifiers",
                "source": "scripts/build_phase6a_graph_dataset.py (hashlib.sha256 "
                "used for duplicate-state/sequence detection)",
                "note": (
                    "Reused SHA-256 as the hashing primitive for deterministic IP "
                    "pseudonymization, consistent with the only existing precedent "
                    "for content-based hashing in this repository."
                ),
            },
            {
                "convention": "one JSON report + one Markdown report per phase",
                "source": "experiments/phase9c_pcap_archive_inspection.py, "
                "experiments/phase9d_pcap_validation.py",
                "note": "Reused verbatim for output structure and naming.",
            },
        ],
        "intentionally_kept_separate": [
            {
                "item": "feature_engine/graph_builder.py (NetworkGraphBuilder)",
                "reason": (
                    "That builder operates on FLOW-level data produced by "
                    "FlowFeatureEngine (src_ip/dst_ip/packet_count/byte_count/"
                    "flow_duration_seconds already aggregated per flow). Phase 9E "
                    "instead builds directly from raw packets without a prior flow-"
                    "segmentation step, per this phase's explicit scope. Reusing "
                    "NetworkGraphBuilder would have required either fabricating "
                    "flow_duration_seconds/flow_start fields not directly available "
                    "from a single-endpoint raw capture, or running the full "
                    "FlowFeatureEngine pipeline, which is out of scope here. A new, "
                    "packet-native graph construction path was written instead."
                ),
            },
            {
                "item": "ingestion/pcap_reader.py (PcapReaderService)",
                "reason": (
                    "Not imported directly (to keep this experiment a "
                    "self-contained standalone script, consistent with the "
                    "Phase 9B/9C/9D convention, and to avoid coupling to the "
                    "`ingestion` Django app's import surface). Its packet-to-"
                    "event field mapping is mirrored by name above but "
                    "reimplemented locally with additional fields (payload "
                    "length, IP fragmentation, explicit SYN/ACK/FIN/RST booleans) "
                    "that PcapReaderService does not expose."
                ),
            },
            {
                "item": "TCP flags representation",
                "reason": (
                    "ingestion/pcap_reader.py stores tcp_flags as "
                    "str(tcp_layer.flags) (a letter code, e.g. 'SA'). Phase 9E "
                    "keeps this same letter-code string for the raw per-packet "
                    "field (tcp_flags_str) AND separately derives explicit boolean "
                    "SYN/ACK/FIN/RST bit checks, since the edge-feature spec "
                    "requires exact per-flag counts rather than a flag string."
                ),
            },
            {
                "item": "anonymization utility",
                "reason": (
                    "No existing anonymization/pseudonymization utility was found "
                    "anywhere in the repository (grep for anonymiz*/hash_ip found "
                    "no match). A new, small, deterministic SHA-256-based function "
                    "was written for this phase only."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Step 2: parse the validated PCAP
# ---------------------------------------------------------------------------


def _protocol_name(packet) -> str:
    if TCP in packet:
        return "TCP"
    if UDP in packet:
        return "UDP"
    if ICMP in packet:
        return "ICMP"
    return "OTHER"


def _extract_ip_packet(packet, timestamp: float) -> dict:
    ip_layer = packet[IP]

    src_port = None
    dst_port = None
    tcp_flags_str = None
    tcp_window = None
    tcp_syn = tcp_ack = tcp_fin = tcp_rst = False
    payload_length = None
    protocol = _protocol_name(packet)

    if protocol == "TCP":
        tcp_layer = packet[TCP]
        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        tcp_flags_str = str(tcp_layer.flags)
        tcp_window = int(tcp_layer.window)
        flag_bits = int(tcp_layer.flags)
        tcp_syn = bool(flag_bits & TCP_FLAG_BITS["SYN"])
        tcp_ack = bool(flag_bits & TCP_FLAG_BITS["ACK"])
        tcp_fin = bool(flag_bits & TCP_FLAG_BITS["FIN"])
        tcp_rst = bool(flag_bits & TCP_FLAG_BITS["RST"])
        payload_length = len(bytes(tcp_layer.payload))
    elif protocol == "UDP":
        udp_layer = packet[UDP]
        src_port = int(udp_layer.sport)
        dst_port = int(udp_layer.dport)
        payload_length = len(bytes(udp_layer.payload))
    elif protocol == "ICMP":
        icmp_layer = packet[ICMP]
        payload_length = len(bytes(icmp_layer.payload))

    return {
        "timestamp": timestamp,
        "packet_length": len(packet),
        "src_ip": ip_layer.src,
        "dst_ip": ip_layer.dst,
        "ip_version": int(ip_layer.version),
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "ttl": int(ip_layer.ttl),
        "tcp_flags_str": tcp_flags_str,
        "tcp_window": tcp_window,
        "tcp_syn": tcp_syn,
        "tcp_ack": tcp_ack,
        "tcp_fin": tcp_fin,
        "tcp_rst": tcp_rst,
        "payload_length": payload_length,
        "ip_flags_str": str(ip_layer.flags),
        "ip_fragment_offset": int(ip_layer.frag),
    }


def parse_full_capture(pcap_path: Path) -> dict:
    """Parse every packet in the capture. Safe here: ~1.25MB, ~10.4k packets."""

    if not pcap_path.exists():
        raise FileNotFoundError(
            f"Phase 9D artifact not found at {pcap_path}. Phase 9D must be run first."
        )

    ip_packets: list[dict] = []
    status_counts: Counter = Counter()
    parse_error_reasons: Counter = Counter()

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    total_packets = 0

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            total_packets += 1
            ts = float(packet.time)
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

            if IP in packet:
                try:
                    ip_packets.append(_extract_ip_packet(packet, ts))
                    status_counts["ok"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    status_counts["parse_error"] += 1
                    parse_error_reasons[str(exc)] += 1
                continue

            if IPv6 is not None and IPv6 in packet:
                status_counts["ipv6_not_implemented"] += 1
            else:
                status_counts["non_ip"] += 1

    if first_ts is None:
        raise ValueError("Capture contains zero packets; cannot proceed.")

    return {
        "ip_packets": ip_packets,
        "total_packets_in_capture": total_packets,
        "status_counts": dict(status_counts),
        "parse_error_reasons": dict(parse_error_reasons),
        "capture_start_ts": first_ts,
        "capture_end_ts": last_ts,
        "capture_duration_seconds": last_ts - first_ts,
    }


# ---------------------------------------------------------------------------
# Step 8 (defined early -- needed before graph construction): anonymization
# ---------------------------------------------------------------------------


def anonymize_ip(ip: str) -> str:
    digest = hashlib.sha256(f"{ANON_SALT}|{ip}".encode("utf-8")).hexdigest()
    return "host_" + digest[:16]


def build_and_validate_anonymization(raw_ips: set[str]) -> tuple[dict[str, str], dict]:
    mapping = {ip: anonymize_ip(ip) for ip in raw_ips}

    # Deterministic repeatability: recomputing must yield identical output.
    repeat_mapping = {ip: anonymize_ip(ip) for ip in raw_ips}
    deterministic_repeatable = mapping == repeat_mapping

    anon_ids = list(mapping.values())
    collision_count = len(anon_ids) - len(set(anon_ids))
    bijective = collision_count == 0 and len(set(mapping.keys())) == len(raw_ips)

    validation = {
        "unique_raw_ips": len(raw_ips),
        "unique_anonymized_ids": len(set(anon_ids)),
        "collision_count": collision_count,
        "bijective_mapping": bijective,
        "deterministic_repeatable": deterministic_repeatable,
        "id_bit_length": 16 * 4,  # 16 hex chars
        "method": "SHA-256(fixed_salt || raw_ip)[:16 hex chars]",
        "privacy_caveat": (
            "This is deterministic pseudonymization for internal artifact "
            "consistency, NOT cryptographically strong anonymization: IPv4 "
            "address space (2^32) is small enough that the mapping is "
            "brute-forceable against a known salt. It prevents casual raw-IP "
            "exposure in serialized artifacts; it does not guarantee "
            "non-reversibility."
        ),
    }
    return mapping, validation


# ---------------------------------------------------------------------------
# Step 3: temporal windows, nodes, edges
# ---------------------------------------------------------------------------


def assign_windows(ip_packets: list[dict], capture_start_ts: float) -> int:
    max_window_index = 0
    for pkt in ip_packets:
        offset = pkt["timestamp"] - capture_start_ts
        window_index = int(offset // WINDOW_SECONDS)
        pkt["offset"] = offset
        pkt["window_index"] = window_index
        if window_index > max_window_index:
            max_window_index = window_index
    return max_window_index


def build_node_window_features(ip_packets: list[dict], anon_map: dict[str, str]) -> dict:
    """window_index -> anon_node_id -> feature dict."""

    buckets: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for pkt in ip_packets:
        w = pkt["window_index"]
        buckets[w][anon_map[pkt["src_ip"]]].append(pkt)
        buckets[w][anon_map[pkt["dst_ip"]]].append(pkt)

    node_windows: dict[int, dict[str, dict]] = {}
    for w, nodes in buckets.items():
        node_windows[w] = {}
        for node_id, involved_packets in nodes.items():
            bytes_sent = 0
            bytes_received = 0
            dests = set()
            srcs = set()
            dst_ports = set()
            src_ports = set()
            protocol_counts = Counter()
            offsets = []
            for pkt in involved_packets:
                offsets.append(pkt["offset"])
                protocol_counts[pkt["protocol"]] += 1
                src_anon = anon_map[pkt["src_ip"]]
                dst_anon = anon_map[pkt["dst_ip"]]
                if src_anon == node_id:
                    bytes_sent += pkt["packet_length"]
                    dests.add(dst_anon)
                    if pkt["dst_port"] is not None:
                        dst_ports.add(pkt["dst_port"])
                if dst_anon == node_id:
                    bytes_received += pkt["packet_length"]
                    srcs.add(src_anon)
                    if pkt["src_port"] is not None:
                        src_ports.add(pkt["src_port"])

            node_windows[w][node_id] = {
                "packet_count": len(involved_packets),
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
                "unique_destinations": len(dests),
                "unique_sources": len(srcs),
                "unique_destination_ports": len(dst_ports),
                "unique_source_ports": len(src_ports),
                "protocol_counts": {p: protocol_counts.get(p, 0) for p in PROTOCOL_VOCAB},
                "first_seen_offset": min(offsets),
                "last_seen_offset": max(offsets),
            }

    return node_windows


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def build_edge_window_features(ip_packets: list[dict], anon_map: dict[str, str]) -> dict:
    """window_index -> (src_anon, dst_anon) -> feature dict."""

    buckets: dict[int, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pkt in ip_packets:
        key = (anon_map[pkt["src_ip"]], anon_map[pkt["dst_ip"]])
        buckets[pkt["window_index"]][key].append(pkt)

    edge_windows: dict[int, dict[tuple[str, str], dict]] = {}
    for w, edges in buckets.items():
        edge_windows[w] = {}
        for key, pkts in edges.items():
            pkts_sorted = sorted(pkts, key=lambda p: p["offset"])
            lengths = [p["packet_length"] for p in pkts_sorted]
            offsets = [p["offset"] for p in pkts_sorted]
            first_offset = offsets[0]
            last_offset = offsets[-1]
            edge_duration = last_offset - first_offset

            if len(offsets) >= 2:
                iats = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
                mean_iat = statistics.fmean(iats)
                std_iat = statistics.pstdev(iats) if len(iats) > 1 else 0.0
            else:
                mean_iat = None
                std_iat = None

            packets_per_second = (
                len(pkts_sorted) / edge_duration if edge_duration > 0 else None
            )
            total_bytes = sum(lengths)
            bytes_per_second = total_bytes / edge_duration if edge_duration > 0 else None

            src_ports = {p["src_port"] for p in pkts_sorted if p["src_port"] is not None}
            dst_ports = {p["dst_port"] for p in pkts_sorted if p["dst_port"] is not None}

            proto_counts = Counter(p["protocol"] for p in pkts_sorted)

            edge_windows[w][key] = {
                "packet_count": len(pkts_sorted),
                "total_bytes": total_bytes,
                "mean_packet_length": statistics.fmean(lengths),
                "min_packet_length": min(lengths),
                "max_packet_length": max(lengths),
                "packet_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
                "first_packet_offset": first_offset,
                "last_packet_offset": last_offset,
                "edge_duration": edge_duration,
                "packets_per_second": packets_per_second,
                "bytes_per_second": bytes_per_second,
                "mean_inter_arrival_time": mean_iat,
                "inter_arrival_time_std": std_iat,
                "tcp_packet_count": proto_counts.get("TCP", 0),
                "udp_packet_count": proto_counts.get("UDP", 0),
                "icmp_packet_count": proto_counts.get("ICMP", 0),
                "other_packet_count": proto_counts.get("OTHER", 0),
                "source_port_diversity": len(src_ports),
                "destination_port_diversity": len(dst_ports),
                "tcp_syn_count": sum(1 for p in pkts_sorted if p["tcp_syn"]),
                "tcp_ack_count": sum(1 for p in pkts_sorted if p["tcp_ack"]),
                "tcp_fin_count": sum(1 for p in pkts_sorted if p["tcp_fin"]),
                "tcp_rst_count": sum(1 for p in pkts_sorted if p["tcp_rst"]),
            }

    return edge_windows


# ---------------------------------------------------------------------------
# Step 4: exact graph statistics
# ---------------------------------------------------------------------------


def compute_capture_level_stats(
    ip_packets: list[dict],
    node_windows: dict,
    edge_windows: dict,
    total_windows: int,
) -> dict:
    all_nodes = set()
    all_pairs = set()
    total_edge_occurrences = 0

    for w, edges in edge_windows.items():
        total_edge_occurrences += len(edges)
        for (src, dst) in edges.keys():
            all_pairs.add((src, dst))
            all_nodes.add(src)
            all_nodes.add(dst)

    non_empty_windows = len(edge_windows)
    empty_windows = total_windows - non_empty_windows

    total_bytes = sum(p["packet_length"] for p in ip_packets)

    # Exact accounting check: sum of per-edge-window packet counts must equal
    # total IP packets parsed (every IP packet belongs to exactly one
    # (window, src, dst) bucket).
    sum_edge_packet_counts = sum(
        edge["packet_count"]
        for edges in edge_windows.values()
        for edge in edges.values()
    )

    return {
        "total_windows": total_windows,
        "non_empty_windows": non_empty_windows,
        "empty_windows": empty_windows,
        "total_nodes_observed": len(all_nodes),
        "total_unique_directed_source_destination_pairs": len(all_pairs),
        "unique_edges_across_entire_capture": len(all_pairs),
        "unique_edges_note": (
            "Identical to total_unique_directed_source_destination_pairs above "
            "-- both are the count of the set of distinct (src, dst) directed "
            "pairs observed anywhere in the capture. Reported under both labels "
            "for direct correspondence with the Phase 9E specification."
        ),
        "total_edges_across_windows": total_edge_occurrences,
        "total_edges_across_windows_note": (
            "Counts a given (src, dst) pair once per window it appears in "
            "(i.e. window-local edge instances), unlike the unique-pair count "
            "above which counts each pair once regardless of how many windows "
            "it appears in."
        ),
        "total_packets_represented": len(ip_packets),
        "total_bytes_represented": total_bytes,
        "packet_accounting_check": {
            "sum_of_edge_window_packet_counts": sum_edge_packet_counts,
            "total_ip_packets_parsed": len(ip_packets),
            "matches": sum_edge_packet_counts == len(ip_packets),
        },
    }


def _window_graph_metrics(nodes: dict, edges: dict) -> dict:
    import networkx as nx

    g = nx.DiGraph()
    for node_id in nodes.keys():
        g.add_node(node_id)
    for (src, dst) in edges.keys():
        g.add_edge(src, dst)

    n = g.number_of_nodes()
    m = g.number_of_edges()

    density = None
    if n >= 2:
        density = m / (n * (n - 1))

    avg_degree = (2 * m / n) if n >= 1 else None

    in_degrees = [d for _, d in g.in_degree()]
    out_degrees = [d for _, d in g.out_degree()]
    total_degrees = [g.in_degree(node) + g.out_degree(node) for node in g.nodes()]
    max_degree = max(total_degrees) if total_degrees else None
    mean_in_degree = statistics.fmean(in_degrees) if in_degrees else None
    mean_out_degree = statistics.fmean(out_degrees) if out_degrees else None
    max_in_degree = max(in_degrees) if in_degrees else None
    max_out_degree = max(out_degrees) if out_degrees else None

    isolated_node_count = sum(1 for d in total_degrees if d == 0)
    component_count = (
        nx.number_weakly_connected_components(g) if n > 0 else 0
    )

    packet_count = sum(e["packet_count"] for e in edges.values())
    byte_count = sum(e["total_bytes"] for e in edges.values())

    return {
        "node_count": n,
        "edge_count": m,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "density": density,
        "average_degree": avg_degree,
        "mean_in_degree": mean_in_degree,
        "mean_out_degree": mean_out_degree,
        "max_in_degree": max_in_degree,
        "max_out_degree": max_out_degree,
        "max_degree": max_degree,
        "isolated_node_count": isolated_node_count,
        "weakly_connected_component_count": component_count,
    }


def compute_per_window_distributions(
    node_windows: dict, edge_windows: dict, total_windows: int
) -> dict:
    metric_series: dict[str, list[float]] = defaultdict(list)

    for w in range(total_windows):
        nodes = node_windows.get(w, {})
        edges = edge_windows.get(w, {})
        metrics = _window_graph_metrics(nodes, edges)
        for key, value in metrics.items():
            if value is not None:
                metric_series[key].append(float(value))

    distributions = {}
    for key, values in metric_series.items():
        distributions[key] = {
            "windows_with_defined_value": len(values),
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "p95": _percentile(values, 95),
            "max": max(values) if values else None,
        }

    return {
        "note": (
            "Computed across ALL windows in [0, total_windows), including empty "
            "windows (contributing 0 for node_count/edge_count/packet_count/"
            "byte_count). Metrics that are mathematically undefined for a given "
            "window (density and average_degree when node_count < 2 or 0; "
            "in/out-degree when node_count == 0) are excluded from that metric's "
            "distribution, and windows_with_defined_value reports how many of "
            "the total_windows windows contributed a value for that specific "
            "metric. isolated_node_count is included for correctness-checking: "
            "by construction (a node is only added to a window's graph if it "
            "appears in at least one edge that window), it is expected to be 0 "
            "in every window -- this is verified empirically below rather than "
            "assumed."
        ),
        "metrics": distributions,
    }


# ---------------------------------------------------------------------------
# Step 5: temporal behavior analysis
# ---------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def compute_temporal_dynamics(
    node_windows: dict, edge_windows: dict, total_windows: int
) -> dict:
    node_sets = [set(node_windows.get(w, {}).keys()) for w in range(total_windows)]
    edge_sets = [set(edge_windows.get(w, {}).keys()) for w in range(total_windows)]

    node_churn = []
    edge_churn = []
    new_nodes_per_window = []
    disappearing_nodes_per_window = []
    new_edges_per_window = []
    disappearing_edges_per_window = []
    node_jaccard_series = []
    edge_jaccard_series = []

    for i in range(1, total_windows):
        prev_nodes, cur_nodes = node_sets[i - 1], node_sets[i]
        prev_edges, cur_edges = edge_sets[i - 1], edge_sets[i]

        new_n = cur_nodes - prev_nodes
        gone_n = prev_nodes - cur_nodes
        new_e = cur_edges - prev_edges
        gone_e = prev_edges - cur_edges

        new_nodes_per_window.append(len(new_n))
        disappearing_nodes_per_window.append(len(gone_n))
        new_edges_per_window.append(len(new_e))
        disappearing_edges_per_window.append(len(gone_e))
        node_churn.append(len(new_n) + len(gone_n))
        edge_churn.append(len(new_e) + len(gone_e))

        nj = _jaccard(prev_nodes, cur_nodes)
        ej = _jaccard(prev_edges, cur_edges)
        if nj is not None:
            node_jaccard_series.append(nj)
        if ej is not None:
            edge_jaccard_series.append(ej)

    # Longest run of consecutive non-empty windows.
    longest_run = 0
    current_run = 0
    total_non_empty_via_sets = 0
    for w in range(total_windows):
        if node_sets[w]:
            current_run += 1
            total_non_empty_via_sets += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    def _summary(values: list[float]) -> dict:
        if not values:
            return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
        return {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p95": _percentile(values, 95),
            "max": max(values),
        }

    return {
        "consecutive_window_pairs_evaluated": max(total_windows - 1, 0),
        "node_churn_per_window_pair": _summary(node_churn),
        "edge_churn_per_window_pair": _summary(edge_churn),
        "new_nodes_per_window": _summary(new_nodes_per_window),
        "disappearing_nodes_per_window": _summary(disappearing_nodes_per_window),
        "new_edges_per_window": _summary(new_edges_per_window),
        "disappearing_edges_per_window": _summary(disappearing_edges_per_window),
        "node_set_jaccard_consecutive": _summary(node_jaccard_series),
        "edge_set_jaccard_consecutive": _summary(edge_jaccard_series),
        "node_set_jaccard_defined_pairs": len(node_jaccard_series),
        "edge_set_jaccard_defined_pairs": len(edge_jaccard_series),
        "longest_consecutive_non_empty_window_run": longest_run,
        "total_non_empty_windows_cross_check": total_non_empty_via_sets,
        "interpretation": (
            "Low mean Jaccard similarity between consecutive windows' node/edge "
            "sets indicates the graph's membership changes substantially "
            "window-to-window (a genuinely temporal graph); a mean near 1.0 "
            "would indicate a near-static graph merely repeated across time."
        ),
    }


# ---------------------------------------------------------------------------
# Step 6: packet-level feature availability
# ---------------------------------------------------------------------------


def feature_availability_table(parse_result: dict, edge_windows: dict) -> list[dict]:
    total = parse_result["total_packets_in_capture"]
    ip_count = len(parse_result["ip_packets"])
    tcp_count = sum(1 for p in parse_result["ip_packets"] if p["protocol"] == "TCP")
    udp_count = sum(1 for p in parse_result["ip_packets"] if p["protocol"] == "UDP")
    tcp_udp_icmp_count = sum(
        1 for p in parse_result["ip_packets"] if p["protocol"] in ("TCP", "UDP", "ICMP")
    )

    edges_with_iat = sum(
        1
        for edges in edge_windows.values()
        for e in edges.values()
        if e["mean_inter_arrival_time"] is not None
    )
    total_edges = sum(len(edges) for edges in edge_windows.values())

    def pct(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "0.0%"
        return f"{100.0 * numerator / denominator:.1f}%"

    return [
        {
            "feature": "timestamp",
            "available": True,
            "derivation": "Direct scapy packet.time field.",
            "coverage": pct(total, total),
            "classification": "DIRECT",
        },
        {
            "feature": "packet length",
            "available": True,
            "derivation": "len(packet) -- direct wire length.",
            "coverage": pct(total, total),
            "classification": "DIRECT",
        },
        {
            "feature": "source/destination IP",
            "available": True,
            "derivation": "IP layer .src/.dst fields; present only on IP packets.",
            "coverage": pct(ip_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "protocol",
            "available": True,
            "derivation": "Layer membership check (TCP/UDP/ICMP presence in packet).",
            "coverage": pct(ip_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "ports",
            "available": True,
            "derivation": "TCP/UDP layer .sport/.dport fields; not applicable to ICMP.",
            "coverage": pct(tcp_count + udp_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "TTL",
            "available": True,
            "derivation": "IP layer .ttl field.",
            "coverage": pct(ip_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "TCP flags",
            "available": True,
            "derivation": "TCP layer .flags field (string form and bitmask checks).",
            "coverage": pct(tcp_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "TCP window",
            "available": True,
            "derivation": "TCP layer .window field.",
            "coverage": pct(tcp_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "payload length",
            "available": True,
            "derivation": "len(bytes(layer.payload)) for TCP/UDP/ICMP layers.",
            "coverage": pct(tcp_udp_icmp_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "fragmentation",
            "available": True,
            "derivation": "IP layer .flags (MF/DF) and .frag offset fields.",
            "coverage": pct(ip_count, total),
            "classification": "DIRECT",
        },
        {
            "feature": "inter-arrival time (IAT)",
            "available": True,
            "derivation": (
                "Exact arithmetic difference between consecutive direct "
                "timestamps of packets sharing the same (window, src, dst) "
                "edge key; requires >=2 packets on that edge in that window."
            ),
            "coverage": pct(edges_with_iat, total_edges) if total_edges else "0.0%",
            "classification": "DIRECT",
        },
        {
            "feature": "retransmission indicators",
            "available": False,
            "derivation": (
                "Would require per-flow TCP sequence-number tracking and "
                "judgment calls about duplicate vs. out-of-order vs. genuine "
                "retransmission -- inherently heuristic, not implemented."
            ),
            "coverage": "0.0%",
            "classification": "NOT IMPLEMENTED",
        },
        {
            "feature": "connection indicators (e.g. completed handshake)",
            "available": False,
            "derivation": (
                "Raw SYN/ACK/FIN/RST counts ARE computed per edge (DIRECT), but "
                "judging whether they form a 'completed connection' requires a "
                "stateful, threshold-based, heuristic interpretation across "
                "windows -- not implemented."
            ),
            "coverage": "0.0%",
            "classification": "NOT IMPLEMENTED",
        },
        {
            "feature": "port-scan indicators",
            "available": False,
            "derivation": (
                "unique_destination_ports is computed (DIRECT) and could feed "
                "a future heuristic, but scan detection itself requires an "
                "arbitrary threshold/time-window judgment call -- not "
                "implemented."
            ),
            "coverage": "0.0%",
            "classification": "NOT IMPLEMENTED",
        },
    ]


# ---------------------------------------------------------------------------
# Step 7: representation feasibility
# ---------------------------------------------------------------------------


def representation_feasibility(
    node_windows: dict,
    edge_windows: dict,
    capture_stats: dict,
    total_windows: int,
    artifact_bytes: Optional[int],
) -> dict:
    node_counts = [len(node_windows.get(w, {})) for w in range(total_windows)]
    edge_counts = [len(edge_windows.get(w, {})) for w in range(total_windows)]

    sample_node_features = next(
        iter(next(iter(node_windows.values())).values())
    ) if node_windows else {}
    sample_edge_features = next(
        iter(next(iter(edge_windows.values())).values())
    ) if edge_windows else {}

    node_feature_dim = len(sample_node_features) - 1 + len(PROTOCOL_VOCAB) if sample_node_features else 0
    edge_feature_dim = len(sample_edge_features) if sample_edge_features else 0

    n_hosts = capture_stats["total_nodes_observed"]
    possible_directed_pairs = n_hosts * (n_hosts - 1) if n_hosts > 1 else 0
    actual_pairs = capture_stats["total_unique_directed_source_destination_pairs"]
    sparsity = (
        1.0 - (actual_pairs / possible_directed_pairs) if possible_directed_pairs else None
    )

    return {
        "representation_a_edge_list_snapshots": {
            "description": (
                "Per window: node_ids (list of anonymized ids), edge_index "
                "(list of (src, dst) anonymized id pairs), edge_features "
                "(per-edge feature dict), node_features (per-node feature dict)."
            ),
            "max_nodes_in_a_single_window": max(node_counts) if node_counts else 0,
            "max_edges_in_a_single_window": max(edge_counts) if edge_counts else 0,
            "mean_nodes_per_non_empty_window": (
                statistics.fmean([c for c in node_counts if c > 0])
                if any(node_counts)
                else 0.0
            ),
            "mean_edges_per_non_empty_window": (
                statistics.fmean([c for c in edge_counts if c > 0])
                if any(edge_counts)
                else 0.0
            ),
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "variable_size_issue": (
                "Node and edge counts vary per window (min/median/mean/p95/max "
                "reported in graph_statistics.per_window_distributions), so this "
                "representation requires ragged/variable-size batching (e.g. "
                "padding+masking or a graph-batching library) rather than a "
                "fixed-size tensor."
            ),
            "storage_note": (
                "This is the representation actually serialized to disk for "
                "this phase (see graph_artifact_path)."
            ),
        },
        "representation_b_fixed_dimensional_state": {
            "description": (
                "Per window, summarize the whole graph into one fixed-length "
                "vector using transparent aggregate statistics: node_count, "
                "edge_count, total packet/byte counts, mean degree, density, "
                "protocol mix totals, SYN/ACK/FIN/RST totals, etc. (the same "
                "quantities already computed in graph_statistics)."
            ),
            "feasible": True,
            "estimated_dim": 12,
            "information_lost": [
                "Which specific host pairs communicated (only aggregate counts survive)",
                "Per-node identity and individual node behavior",
                "Full in/out-degree distribution shape (only mean/max survive)",
                "Edge directionality between specific pairs",
            ],
            "information_preserved": [
                "Aggregate traffic volume (packets, bytes)",
                "Host/edge count and rough density",
                "Protocol mix",
                "TCP control-flag totals",
            ],
        },
        "sparsity": {
            "total_nodes_observed": n_hosts,
            "possible_directed_pairs_excluding_self_loops": possible_directed_pairs,
            "actual_unique_directed_pairs_observed": actual_pairs,
            "sparsity_fraction": sparsity,
        },
        "storage_requirements": {
            "serialized_graph_artifact_bytes": artifact_bytes,
        },
        "decision_deferred": (
            "No representation is selected as superior here, per Phase 9E scope; "
            "this is a feasibility characterization only."
        ),
    }


# ---------------------------------------------------------------------------
# Step 9: CSV relationship
# ---------------------------------------------------------------------------


def csv_relationship_report() -> dict:
    csv_exists = CSV_PATH.exists()
    endpoint_identity_present = False
    protocol_column_present = False
    timestamp_column_present = False

    if csv_exists:
        with CSV_PATH.open("r", encoding="utf-8", errors="replace") as fh:
            header_line = fh.readline()
        columns = [c.strip() for c in header_line.split(",")]
        endpoint_identity_present = ("Src IP" in columns) and ("Dst IP" in columns)
        protocol_column_present = "Protocol" in columns
        timestamp_column_present = "Timestamp" in columns

    return {
        "csv_exists_for_this_day": csv_exists,
        "csv_path": str(CSV_PATH),
        "endpoint_identity_present_in_csv": endpoint_identity_present,
        "protocol_column_present": protocol_column_present,
        "timestamp_column_present": timestamp_column_present,
        "csv_temporal_granularity": (
            "Each CSV row is a single CICFlowMeter flow record with its own "
            "Timestamp (the flow's start time); the CSV format itself has no "
            "fixed temporal binning. This is a known property of the "
            "CICFlowMeter output format, not a claim derived from opening the "
            "full file in this phase."
        ),
        "packet_graph_temporal_granularity": f"Fixed {WINDOW_SECONDS}-second windows.",
        "theoretical_window_level_alignment": (
            "In principle, CSV rows could be bucketed into the same "
            f"{WINDOW_SECONDS}-second windows by their Timestamp field, "
            "enabling a window-level (not row-level) join against this "
            "capture's packet-graph windows."
        ),
        "what_prevents_proving_alignment_now": (
            "This day's CSV (confirmed in Phase 9D) lacks Src IP/Dst IP columns "
            "entirely, so even a window-level join cannot be verified to "
            "correspond to the SAME hosts seen in this packet capture -- only "
            "to the same time window in general. No such join was attempted "
            "in this phase."
        ),
        "row_level_correspondence_established": False,
        "explicit_statement": "No flow-row correspondence was established.",
    }


# ---------------------------------------------------------------------------
# Steps 10-11: narrative sections grounded in measured numbers
# ---------------------------------------------------------------------------


def multimodal_architecture_implications(capture_stats: dict, dynamics: dict) -> dict:
    return {
        "topology_change": {
            "observed_in_this_pcap": (
                f"Mean consecutive-window node-set Jaccard similarity is "
                f"{dynamics['node_set_jaccard_consecutive']['mean']}, and edge-set "
                f"Jaccard is {dynamics['edge_set_jaccard_consecutive']['mean']} "
                f"(over {dynamics['node_set_jaccard_defined_pairs']} defined "
                "window pairs) -- the graph's membership is not static across "
                "windows in this single capture."
            ),
            "hypothesized_future_use": (
                "A world model could consume per-window topology-change "
                "signals (churn, Jaccard) as an auxiliary input alongside flow "
                "state, on the hypothesis that topology instability precedes "
                "or accompanies anomalous activity."
            ),
        },
        "communication_expansion": {
            "observed_in_this_pcap": (
                f"{capture_stats['total_nodes_observed']} distinct hosts and "
                f"{capture_stats['total_unique_directed_source_destination_pairs']} "
                "unique directed pairs were observed from a single endpoint's "
                "capture over ~8.95 hours."
            ),
            "hypothesized_future_use": (
                "Sudden growth in unique_destinations for a node (already a "
                "DIRECT node feature) is a plausible predictive signal for "
                "future models; not evaluated for predictive value here."
            ),
        },
        "host_centrality_changes": {
            "observed_in_this_pcap": (
                "Per-window degree distributions (min/median/mean/p95/max) are "
                "computed in graph_statistics.per_window_distributions and show "
                "variation across windows."
            ),
            "hypothesized_future_use": (
                "Centrality shifts (e.g. a normally low-degree host suddenly "
                "gaining many peers) could be a future world-model feature."
            ),
        },
        "port_behavior": {
            "observed_in_this_pcap": (
                "Per-edge source/destination port diversity is computed exactly "
                "per window; not aggregated into a capture-wide claim here."
            ),
            "hypothesized_future_use": (
                "Port-diversity spikes could feed a future (not-yet-built) "
                "heuristic or learned scan-detection signal."
            ),
        },
        "protocol_behavior": {
            "observed_in_this_pcap": (
                "TCP/UDP/ICMP packet counts are tracked per edge and per node "
                "window; see graph_statistics and node/edge feature schemas."
            ),
            "hypothesized_future_use": (
                "Protocol-mix shift per window is a natural complementary "
                "signal to the flow-level protocol features already used "
                "elsewhere in this project."
            ),
        },
        "packet_volume_changes": {
            "observed_in_this_pcap": (
                "Per-window packet_count and byte_count distributions are "
                "reported exactly (min/median/mean/p95/max)."
            ),
            "hypothesized_future_use": (
                "Volume bursts relative to a host's own baseline could "
                "contribute to a future anomaly signal."
            ),
        },
        "burst_iat_behavior": {
            "observed_in_this_pcap": (
                "Mean/std inter-arrival time is computed exactly per edge "
                "(where >=2 packets exist on that edge in that window)."
            ),
            "hypothesized_future_use": (
                "IAT compression (packets arriving unusually close together) "
                "is a classic burst indicator that could feed a future model."
            ),
        },
        "flow_vs_packet_complementary_information": {
            "observed_in_this_pcap": (
                "This capture carries fields the existing flow-level CSV "
                "pipeline does not retain post-aggregation: per-packet TTL, "
                "TCP window, IP fragmentation flags, and exact per-packet "
                "timestamps rather than flow-level start/end times."
            ),
            "hypothesized_future_use": (
                "These packet-only fields are the concrete candidate "
                "contribution of the PACKET GRAPH branch in the architecture "
                "diagram, complementary to (not a replacement for) FLOW STATE."
            ),
        },
        "predictive_value_disclaimer": (
            "No predictive value is claimed for any signal above; all items "
            "are either directly observed structural facts about this one "
            "capture or explicitly labeled hypotheses for future work."
        ),
    }


def novelty_assessment(capture_stats: dict, dynamics: dict, repr_feasibility: dict) -> dict:
    return {
        "claim": (
            "This phase does NOT claim novelty merely from using a graph "
            "representation. It assesses only whether the packet modality is "
            "technically CAPABLE of supporting the intended future components."
        ),
        "multimodal_network_state": {
            "supported": True,
            "basis": (
                "A packet-derived graph state (Representation A or B) was "
                "successfully constructed per window alongside the existing "
                "flow-state machinery, with no fabricated fields -- confirms "
                "the packet modality can structurally sit beside FLOW STATE as "
                "a second input branch."
            ),
        },
        "learned_temporal_dynamics": {
            "supported": "CONDITIONALLY",
            "basis": (
                f"Consecutive-window churn is non-trivial (mean node churn "
                f"{dynamics['node_churn_per_window_pair']['mean']}, mean edge "
                f"churn {dynamics['edge_churn_per_window_pair']['mean']}), "
                "meaning there is genuine temporal signal to learn from in "
                "this one capture -- but this was measured on a single "
                "endpoint's traffic only, so generalization across hosts/days "
                "is NOT VERIFIED."
            ),
        },
        "multi_step_attack_trajectory": {
            "supported": "NOT VERIFIED",
            "basis": (
                "No attack labels exist for this capture (per hard constraint "
                "#7/#8), so nothing about trajectory-relevant signal content "
                "can be assessed here -- only that the graph has enough "
                "temporal structure (non-trivial churn) to be worth pairing "
                "with labels in a future, separate phase."
            ),
        },
        "uncertainty": {
            "supported": "NOT VERIFIED",
            "basis": (
                "Nothing in this phase touches probabilistic modeling; the "
                "graph construction here is entirely deterministic and exact."
            ),
        },
        "counterfactual_intervention": {
            "supported": "NOT VERIFIED",
            "basis": (
                "Out of scope per hard constraint #6; the edge/node feature "
                "schema captured here would need to support a future "
                "perturbation interface, which was not evaluated."
            ),
        },
        "overall": (
            "The packet modality provides a credible, technically grounded "
            "STRUCTURAL foundation (exact windowed graph with real topology "
            "change) for the future multimodal concept, but this single-"
            "endpoint, unlabeled experiment verifies structure only -- not "
            "predictive value, not generalization, and not the downstream "
            "components (trajectory/uncertainty/counterfactual) themselves."
        ),
    }


# ---------------------------------------------------------------------------
# Artifact serialization
# ---------------------------------------------------------------------------


def write_graph_artifact(
    node_windows: dict, edge_windows: dict, total_windows: int
) -> tuple[Path, int, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_path = DATA_DIR / "graph_snapshots.json"

    snapshots = []
    for w in range(total_windows):
        nodes = node_windows.get(w, {})
        edges = edge_windows.get(w, {})
        if not nodes and not edges:
            continue
        snapshots.append(
            {
                "window_index": w,
                "node_features": nodes,
                "edges": [
                    {"src": src, "dst": dst, "features": feats}
                    for (src, dst), feats in edges.items()
                ],
            }
        )

    payload = {"window_seconds": WINDOW_SECONDS, "total_windows": total_windows, "snapshots": snapshots}
    text = json.dumps(payload, indent=2)

    if len(text.encode("utf-8")) <= MAX_ARTIFACT_BYTES_BEFORE_SUMMARIZING:
        full_path.write_text(text, encoding="utf-8")
        return full_path, len(text.encode("utf-8")), "full_edge_list_snapshots"

    # Fallback: summary-only artifact (per-window scalar stats only).
    summary_path = DATA_DIR / "graph_window_summary.json"
    summary = []
    for snap in snapshots:
        summary.append(
            {
                "window_index": snap["window_index"],
                "node_count": len(snap["node_features"]),
                "edge_count": len(snap["edges"]),
                "packet_count": sum(e["features"]["packet_count"] for e in snap["edges"]),
                "byte_count": sum(e["features"]["total_bytes"] for e in snap["edges"]),
            }
        )
    summary_text = json.dumps(
        {"window_seconds": WINDOW_SECONDS, "total_windows": total_windows, "window_summaries": summary},
        indent=2,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path, len(summary_text.encode("utf-8")), "summary_only_fallback"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9E: Temporal Packet Graph Construction & Feasibility")
    lines.append("")

    verdict = report["verdict"]
    lines.append(f"## 1. EXECUTIVE RESULT")
    lines.append("")
    lines.append(f"Verdict: **{verdict}**")
    lines.append("")
    lines.append(
        f"A temporal, windowed, directed communication graph was constructed from "
        f"the single validated PCAP member `pcap/UCAP172.31.69.22` "
        f"({report['parsing_results']['total_ip_packets']} IP packets across "
        f"{report['temporal_window_structure']['total_windows']} "
        f"{WINDOW_SECONDS}-second windows, "
        f"{report['graph_statistics']['capture_level']['non_empty_windows']} of "
        f"them non-empty). CONFIRMED: graph construction is exact and internally "
        f"consistent (packet-accounting check: "
        f"{report['graph_statistics']['capture_level']['packet_accounting_check']['matches']}). "
        f"OBSERVED: the graph changes meaningfully over time (see section 9). "
        f"NOT VERIFIED: predictive value, cross-host/cross-day generalization, "
        f"or any attack relevance."
    )
    lines.append("")

    lines.append("## 2. INPUT PCAP")
    lines.append("")
    lines.append(f"- Path: `{PCAP_PATH}`")
    lines.append(
        f"- Source: Phase 9D validated artifact (already retrieved/decompressed; "
        f"no network access in this phase)"
    )
    lines.append(
        f"- Capture span: {report['temporal_window_structure']['capture_start_ts']} "
        f"to {report['temporal_window_structure']['capture_end_ts']} "
        f"({report['temporal_window_structure']['capture_duration_seconds']:.3f}s)"
    )
    lines.append(
        f"- Cross-check against Phase 9D persisted aggregates: "
        f"{report['temporal_window_structure']['matches_phase9d_aggregates']}"
    )
    lines.append("")

    lines.append("## 3. PARSING RESULTS")
    lines.append("")
    pr = report["parsing_results"]
    lines.append(f"- Total packets in capture: {pr['total_packets_in_capture']}")
    lines.append(f"- IP packets (parsed into graph): {pr['total_ip_packets']}")
    lines.append(f"- Status counts: {pr['status_counts']}")
    lines.append("")

    lines.append("## 4. TEMPORAL WINDOW STRUCTURE")
    lines.append("")
    tw = report["temporal_window_structure"]
    lines.append(f"- Window size: {tw['window_seconds']}s (CONFIRMED reused project convention)")
    lines.append(f"- Total windows: {tw['total_windows']}")
    lines.append("")

    lines.append("## 5. GRAPH STATISTICS")
    lines.append("")
    cl = report["graph_statistics"]["capture_level"]
    for k, v in cl.items():
        if isinstance(v, dict):
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Per-window distributions (min / median / mean / p95 / max):")
    lines.append("")
    lines.append("| metric | windows w/ value | min | median | mean | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric, dist in report["graph_statistics"]["per_window_distributions"]["metrics"].items():
        lines.append(
            f"| {metric} | {dist['windows_with_defined_value']} | {dist['min']} | "
            f"{dist['median']} | {dist['mean']} | {dist['p95']} | {dist['max']} |"
        )
    lines.append("")

    lines.append("## 6. NODE FEATURES")
    lines.append("")
    lines.append(
        "Per (window, anonymized host): packet_count, bytes_sent, bytes_received, "
        "unique_destinations, unique_sources, unique_destination_ports, "
        "unique_source_ports, protocol_counts (TCP/UDP/ICMP/OTHER), "
        "first_seen_offset, last_seen_offset. Topology (graph edges) is kept "
        "strictly separate from these observational node features."
    )
    lines.append("")

    lines.append("## 7. EDGE FEATURES")
    lines.append("")
    lines.append(
        "Per (window, src->dst): packet_count, total_bytes, mean/min/max/std "
        "packet_length, first/last_packet_offset, edge_duration, "
        "packets_per_second, bytes_per_second, mean/std inter-arrival time, "
        "TCP/UDP/ICMP/OTHER packet counts, source/destination port diversity, "
        "TCP SYN/ACK/FIN/RST counts. Rate and IAT fields are `null` when "
        "mathematically undefined (single packet, zero-duration edge) rather "
        "than fabricated as zero."
    )
    lines.append("")

    lines.append("## 8. PACKET FEATURE AVAILABILITY")
    lines.append("")
    lines.append("| feature | available | coverage | classification |")
    lines.append("|---|---|---:|---|")
    for row in report["packet_feature_availability"]:
        lines.append(
            f"| {row['feature']} | {row['available']} | {row['coverage']} | "
            f"{row['classification']} |"
        )
    lines.append("")

    lines.append("## 9. TEMPORAL DYNAMICS")
    lines.append("")
    td = report["temporal_dynamics"]
    lines.append(f"- Longest consecutive non-empty window run: {td['longest_consecutive_non_empty_window_run']}")
    lines.append(f"- Mean node-set Jaccard (consecutive windows): {td['node_set_jaccard_consecutive']['mean']}")
    lines.append(f"- Mean edge-set Jaccard (consecutive windows): {td['edge_set_jaccard_consecutive']['mean']}")
    lines.append(f"- Mean node churn per consecutive pair: {td['node_churn_per_window_pair']['mean']}")
    lines.append(f"- Mean edge churn per consecutive pair: {td['edge_churn_per_window_pair']['mean']}")
    lines.append(f"- {td['interpretation']}")
    lines.append("")

    lines.append("## 10. REPRESENTATION FEASIBILITY")
    lines.append("")
    rf = report["representation_feasibility"]
    lines.append(f"- Representation A (edge-list snapshots): max nodes/window = "
                  f"{rf['representation_a_edge_list_snapshots']['max_nodes_in_a_single_window']}, "
                  f"max edges/window = {rf['representation_a_edge_list_snapshots']['max_edges_in_a_single_window']}, "
                  f"node_feature_dim = {rf['representation_a_edge_list_snapshots']['node_feature_dim']}, "
                  f"edge_feature_dim = {rf['representation_a_edge_list_snapshots']['edge_feature_dim']}")
    lines.append(f"- Representation B (fixed-dim state): feasible = "
                  f"{rf['representation_b_fixed_dimensional_state']['feasible']}, "
                  f"estimated_dim = {rf['representation_b_fixed_dimensional_state']['estimated_dim']}")
    lines.append(f"- Sparsity: {rf['sparsity']['sparsity_fraction']}")
    lines.append(f"- Serialized artifact bytes: {rf['storage_requirements']['serialized_graph_artifact_bytes']}")
    lines.append(f"- {rf['decision_deferred']}")
    lines.append("")

    lines.append("## 11. CSV RELATIONSHIP")
    lines.append("")
    cr = report["csv_relationship"]
    for k in [
        "csv_exists_for_this_day",
        "endpoint_identity_present_in_csv",
        "protocol_column_present",
        "timestamp_column_present",
        "row_level_correspondence_established",
    ]:
        lines.append(f"- {k}: {cr[k]}")
    lines.append(f"- {cr['explicit_statement']}")
    lines.append("")

    lines.append("## 12. MULTIMODAL ARCHITECTURE IMPLICATIONS")
    lines.append("")
    for key, val in report["multimodal_architecture_implications"].items():
        if key == "predictive_value_disclaimer":
            continue
        lines.append(f"- **{key}**: OBSERVED: {val['observed_in_this_pcap']}")
        lines.append(f"  HYPOTHESIZED: {val['hypothesized_future_use']}")
    lines.append("")
    lines.append(report["multimodal_architecture_implications"]["predictive_value_disclaimer"])
    lines.append("")

    lines.append("## 13. NOVELTY IMPLICATIONS")
    lines.append("")
    lines.append(report["novelty_assessment"]["claim"])
    lines.append("")
    lines.append(f"Overall: {report['novelty_assessment']['overall']}")
    lines.append("")

    lines.append("## 14. LIMITATIONS")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 15. RECOMMENDED NEXT EXPERIMENT")
    lines.append("")
    lines.append(report["recommended_next_experiment"])
    lines.append("")

    lines.append(f"## FINAL VERDICT: {verdict}")
    lines.append("")
    lines.append("STOP AFTER PHASE 9E.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Refusing to overwrite a prior Phase 9E run."
        )

    conventions = conventions_report()

    parse_result = parse_full_capture(PCAP_PATH)
    ip_packets = parse_result["ip_packets"]

    max_window_index = assign_windows(ip_packets, parse_result["capture_start_ts"])
    total_windows = max_window_index + 1

    raw_ips = {p["src_ip"] for p in ip_packets} | {p["dst_ip"] for p in ip_packets}
    anon_map, anon_validation = build_and_validate_anonymization(raw_ips)

    node_windows = build_node_window_features(ip_packets, anon_map)
    edge_windows = build_edge_window_features(ip_packets, anon_map)

    capture_stats = compute_capture_level_stats(ip_packets, node_windows, edge_windows, total_windows)
    per_window_distributions = compute_per_window_distributions(node_windows, edge_windows, total_windows)
    dynamics = compute_temporal_dynamics(node_windows, edge_windows, total_windows)
    availability = feature_availability_table(parse_result, edge_windows)

    artifact_path, artifact_bytes, artifact_mode = write_graph_artifact(
        node_windows, edge_windows, total_windows
    )

    repr_feasibility = representation_feasibility(
        node_windows, edge_windows, capture_stats, total_windows, artifact_bytes
    )
    csv_rel = csv_relationship_report()
    multimodal = multimodal_architecture_implications(capture_stats, dynamics)
    novelty = novelty_assessment(capture_stats, dynamics, repr_feasibility)

    # Phase 9D cross-check (read-only comparison against the frozen artifact).
    matches_phase9d = False
    if PHASE9D_REPORT_PATH.exists():
        phase9d = json.loads(PHASE9D_REPORT_PATH.read_text(encoding="utf-8"))
        agg = phase9d.get("step4_capture_aggregates", {})
        matches_phase9d = (
            agg.get("total_packet_count") == parse_result["total_packets_in_capture"]
            and abs(agg.get("first_timestamp", 0) - parse_result["capture_start_ts"]) < 1e-6
            and abs(agg.get("last_timestamp", 0) - parse_result["capture_end_ts"]) < 1e-6
        )

    isolated_check = per_window_distributions["metrics"].get("isolated_node_count", {})
    limitations = [
        "This entire experiment is scoped to ONE endpoint's capture "
        "(pcap/UCAP172.31.69.22); no claim is made about the other 448 "
        "archive members or the dataset as a whole.",
        "No attack label exists for any window in this capture; nothing here "
        "should be read as evidence of attack or benign traffic.",
        "IPv6 is not implemented (none observed in this capture; a non-IPv4 "
        "packet is classified as ipv6_not_implemented or non_ip, not dropped "
        "silently).",
        "Retransmission indicators, connection-completion indicators, and "
        "port-scan indicators are NOT IMPLEMENTED (would require heuristic, "
        "threshold-based judgment, out of scope for this phase).",
        f"isolated_node_count is empirically {isolated_check.get('max')} "
        "(max across windows) confirming the by-construction expectation that "
        "every node in a window's graph has degree >= 1.",
        "Anonymization is deterministic pseudonymization for artifact "
        "hygiene, not cryptographically strong privacy protection (see "
        "anonymization_validation.privacy_caveat in the JSON report).",
        "Raw per-packet records are not persisted to disk (only the "
        "anonymized, window-aggregated graph artifact is written), per the "
        "'do not store raw PCAP-derived data unnecessarily' output constraint.",
        f"Packet accounting check: {capture_stats['packet_accounting_check']['matches']} "
        "(sum of per-edge-window packet counts equals total IP packets parsed).",
    ]

    recommended_next = (
        "If pursued further (not started here): parse a second, larger member "
        "(e.g. one of the capWIN-J6GMIG1DQE5-* hosts) with this same pipeline "
        "and compare its graph-statistics distributions against this host's, to "
        "determine whether the temporal/topological patterns observed here "
        "(non-trivial churn, sparse but multi-host structure) generalize across "
        "hosts before any model architecture is chosen."
    )

    verdict = "GREEN"
    if not capture_stats["packet_accounting_check"]["matches"] or not anon_validation["bijective_mapping"]:
        verdict = "RED"
    elif capture_stats["non_empty_windows"] < 2:
        verdict = "YELLOW"

    report = {
        "success": True,
        "verdict": verdict,
        "conventions": conventions,
        "parsing_results": {
            "total_packets_in_capture": parse_result["total_packets_in_capture"],
            "total_ip_packets": len(ip_packets),
            "status_counts": parse_result["status_counts"],
            "parse_error_reasons": parse_result["parse_error_reasons"],
        },
        "temporal_window_structure": {
            "window_seconds": WINDOW_SECONDS,
            "capture_start_ts": parse_result["capture_start_ts"],
            "capture_end_ts": parse_result["capture_end_ts"],
            "capture_duration_seconds": parse_result["capture_duration_seconds"],
            "total_windows": total_windows,
            "matches_phase9d_aggregates": matches_phase9d,
        },
        "anonymization_validation": anon_validation,
        "graph_statistics": {
            "capture_level": capture_stats,
            "per_window_distributions": per_window_distributions,
        },
        "temporal_dynamics": dynamics,
        "packet_feature_availability": availability,
        "representation_feasibility": repr_feasibility,
        "csv_relationship": csv_rel,
        "multimodal_architecture_implications": multimodal,
        "novelty_assessment": novelty,
        "graph_artifact_path": str(artifact_path),
        "graph_artifact_bytes": artifact_bytes,
        "graph_artifact_mode": artifact_mode,
        "limitations": limitations,
        "recommended_next_experiment": recommended_next,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "graph_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "graph_report.md", report)

    print(
        json.dumps(
            {
                "success": True,
                "verdict": verdict,
                "total_ip_packets": len(ip_packets),
                "total_windows": total_windows,
                "non_empty_windows": capture_stats["non_empty_windows"],
                "total_nodes_observed": capture_stats["total_nodes_observed"],
                "unique_directed_pairs": capture_stats["total_unique_directed_source_destination_pairs"],
                "packet_accounting_matches": capture_stats["packet_accounting_check"]["matches"],
                "anonymization_bijective": anon_validation["bijective_mapping"],
                "graph_artifact_bytes": artifact_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
