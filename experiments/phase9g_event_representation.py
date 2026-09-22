"""Phase 9G: temporal communication event representation feasibility.

Compares three ways of representing the same raw packet stream for a future
predictive world model:

  A.  10-second graph snapshots (Phase 9E/9F's existing representation)
  B1. one event per IP packet (no aggregation)
  B2. micro-batched communication events (packets sharing source,
      destination, protocol, source port, destination port, grouped within
      a small, fixed, undtuned temporal bucket)
  C.  a conceptual hybrid: the B2 event stream plus a 10-second window-state
      summary, defined and size-measured but NOT neural-fused

No model is trained, no GNN/Transformer/World Model is implemented, no
attack labels are assigned. Both PCAPs used here were already retrieved and
validated in Phases 9D and 9F -- this phase performs NO network access.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read-only reuse of Phase 9E's EXACT parsing/graph-construction pipeline --
# per hard constraint #12 (identical packet parsing conventions) and #13
# (deterministic anonymization), nothing here reimplements these.
from phase9e_temporal_packet_graph import (  # noqa: E402
    WINDOW_SECONDS,
    anonymize_ip,
    assign_windows,
    build_and_validate_anonymization,
    build_edge_window_features,
    build_node_window_features,
    compute_capture_level_stats,
    compute_per_window_distributions,
    compute_temporal_dynamics,
    parse_full_capture,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9g_event_representation"

# Fixed, documented, NOT tuned for any predictive objective (hard constraint:
# "Do not tune this for predictive performance"). 1 second is an order of
# magnitude finer than the existing WINDOW_SECONDS=10 graph-snapshot
# granularity, chosen only to demonstrate a genuinely different temporal
# resolution, not selected by any search or performance criterion.
MICRO_BATCH_SECONDS = 1

CAPTURES = [
    {
        "label": "phase9d_UCAP172.31.69.22",
        "pcap_path": EXPERIMENTS_DIR / "data/phase9d/UCAP172.31.69.22.pcap",
        "source_phase": "Phase 9D (already retrieved/validated; reused read-only)",
    },
    {
        "label": "phase9f_capWIN-J6GMIG1DQE5-172.31.64.89",
        "pcap_path": EXPERIMENTS_DIR / "data/phase9f/capWIN-J6GMIG1DQE5-172.31.64.89.pcap",
        "source_phase": "Phase 9F (already retrieved/validated; reused read-only)",
    },
]

TCP_FLAG_KEYS = ("tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst")


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    import math

    k = (len(s) - 1) * (pct / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


# ---------------------------------------------------------------------------
# Step 1: Representation A (reused pipeline; measured, not re-derived logic)
# ---------------------------------------------------------------------------


def build_representation_a(ip_packets: list[dict], anon_map: dict, total_windows: int) -> dict:
    node_windows = build_node_window_features(ip_packets, anon_map)
    edge_windows = build_edge_window_features(ip_packets, anon_map)
    capture_level = compute_capture_level_stats(ip_packets, node_windows, edge_windows, total_windows)
    per_window = compute_per_window_distributions(node_windows, edge_windows, total_windows)
    dynamics = compute_temporal_dynamics(node_windows, edge_windows, total_windows)

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
                "edges": [{"src": s, "dst": d, "features": f} for (s, d), f in edges.items()],
            }
        )
    serialized = json.dumps({"window_seconds": WINDOW_SECONDS, "total_windows": total_windows, "snapshots": snapshots})
    serialized_bytes = len(serialized.encode("utf-8"))

    return {
        "window_seconds": WINDOW_SECONDS,
        "total_windows": total_windows,
        "non_empty_windows": capture_level["non_empty_windows"],
        "total_window_edges": capture_level["total_edges_across_windows"],
        "unique_edges": capture_level["unique_edges_across_entire_capture"],
        "total_nodes_observed": capture_level["total_nodes_observed"],
        "per_window_distributions": per_window["metrics"],
        "temporal_transitions_represented": max(total_windows - 1, 0),
        "serialized_size_bytes": serialized_bytes,
        "packet_accounting_check": capture_level["packet_accounting_check"],
        "_node_windows": node_windows,
        "_edge_windows": edge_windows,
    }


# ---------------------------------------------------------------------------
# Step 2-3: Representation B1 (packet events) and B2 (micro-batch events)
# ---------------------------------------------------------------------------


def build_representation_b1(ip_packets: list[dict], anon_map: dict) -> dict:
    """One event per IP packet -- no aggregation. Events are not persisted
    (would duplicate the already-decompressed PCAP); only counts/size are
    measured, from a small sample used to estimate per-event serialized size."""

    def to_event(p: dict) -> dict:
        return {
            "timestamp": p["timestamp"],
            "src": anon_map[p["src_ip"]],
            "dst": anon_map[p["dst_ip"]],
            "protocol": p["protocol"],
            "src_port": p["src_port"],
            "dst_port": p["dst_port"],
            "packet_length": p["packet_length"],
            "payload_length": p["payload_length"],
            "tcp_flags": p["tcp_flags_str"],
        }

    sample_size = min(500, len(ip_packets))
    sample_events = [to_event(p) for p in ip_packets[:sample_size]]
    sample_bytes = len(json.dumps(sample_events).encode("utf-8"))
    estimated_bytes_per_event = sample_bytes / sample_size if sample_size else 0.0
    estimated_total_bytes = int(estimated_bytes_per_event * len(ip_packets))

    return {
        "event_count": len(ip_packets),
        "event_fields": list(to_event(ip_packets[0]).keys()) if ip_packets else [],
        "estimated_bytes_per_event": estimated_bytes_per_event,
        "estimated_serialized_size_bytes": estimated_total_bytes,
        "size_estimation_method": (
            f"Measured exact JSON size of the first {sample_size} events, then "
            "scaled linearly by total event_count (packet events are structurally "
            "uniform, so per-event size does not vary meaningfully across the "
            "capture)."
        ),
    }


def build_representation_b2(ip_packets: list[dict], anon_map: dict, capture_duration: float) -> dict:
    """Group packets sharing (source, destination, protocol, source port,
    destination port) within a MICRO_BATCH_SECONDS-wide temporal bucket."""

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in ip_packets:
        bucket_index = int(p["offset"] // MICRO_BATCH_SECONDS)
        key = (
            bucket_index,
            anon_map[p["src_ip"]],
            anon_map[p["dst_ip"]],
            p["protocol"],
            p["src_port"],
            p["dst_port"],
        )
        buckets[key].append(p)

    events = []
    for (bucket_index, src, dst, protocol, src_port, dst_port), pkts in buckets.items():
        pkts_sorted = sorted(pkts, key=lambda x: x["offset"])
        offsets = [x["offset"] for x in pkts_sorted]
        lengths = [x["packet_length"] for x in pkts_sorted]
        first_offset, last_offset = offsets[0], offsets[-1]
        duration = last_offset - first_offset

        if len(offsets) >= 2:
            iats = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
            mean_iat = statistics.fmean(iats)
            iat_std = statistics.pstdev(iats) if len(iats) > 1 else 0.0
        else:
            mean_iat = None
            iat_std = None

        src_ports_seen = {x["src_port"] for x in pkts_sorted if x["src_port"] is not None}
        dst_ports_seen = {x["dst_port"] for x in pkts_sorted if x["dst_port"] is not None}

        events.append(
            {
                "bucket_index": bucket_index,
                "event_timestamp_offset": first_offset,
                "src": src,
                "dst": dst,
                "protocol": protocol,
                "src_port": src_port,
                "dst_port": dst_port,
                "packet_count": len(pkts_sorted),
                "total_bytes": sum(lengths),
                "duration": duration,
                "mean_packet_length": statistics.fmean(lengths),
                "packet_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
                "mean_IAT": mean_iat,
                "IAT_std": iat_std,
                "tcp_syn_count": sum(1 for x in pkts_sorted if x["tcp_syn"]),
                "tcp_ack_count": sum(1 for x in pkts_sorted if x["tcp_ack"]),
                "tcp_fin_count": sum(1 for x in pkts_sorted if x["tcp_fin"]),
                "tcp_rst_count": sum(1 for x in pkts_sorted if x["tcp_rst"]),
                "source_port_diversity": len(src_ports_seen),
                "destination_port_diversity": len(dst_ports_seen),
            }
        )

    events.sort(key=lambda e: e["event_timestamp_offset"])

    total_packets_via_events = sum(e["packet_count"] for e in events)
    total_bytes_via_events = sum(e["total_bytes"] for e in events)
    burst_sizes = [e["packet_count"] for e in events]

    event_count = len(events)
    compression_ratio_packets_per_event = (len(ip_packets) / event_count) if event_count else None
    events_per_second = (event_count / capture_duration) if capture_duration else None

    # Step 5: event-sequence structure (chronological pass)
    inter_event_gaps = [
        events[i]["event_timestamp_offset"] - events[i - 1]["event_timestamp_offset"]
        for i in range(1, event_count)
    ]

    seen_pairs: set[tuple[str, str]] = set()
    seen_dst: set[str] = set()
    seen_src: set[str] = set()
    unseen_pair_count = 0
    unseen_dst_count = 0
    unseen_src_count = 0
    for e in events:
        pair = (e["src"], e["dst"])
        if pair not in seen_pairs:
            unseen_pair_count += 1
            seen_pairs.add(pair)
        if e["dst"] not in seen_dst:
            unseen_dst_count += 1
            seen_dst.add(e["dst"])
        if e["src"] not in seen_src:
            unseen_src_count += 1
            seen_src.add(e["src"])

    unique_hosts = seen_src | seen_dst

    structure_stats = {
        "total_events": event_count,
        "events_per_second": events_per_second,
        "median_inter_event_time": _percentile(inter_event_gaps, 50),
        "p95_inter_event_time": _percentile(inter_event_gaps, 95),
        "maximum_inter_event_gap": max(inter_event_gaps) if inter_event_gaps else None,
        "median_burst_size": _percentile(burst_sizes, 50),
        "p95_burst_size": _percentile(burst_sizes, 95),
        "maximum_burst_size": max(burst_sizes) if burst_sizes else None,
        "unique_communicating_pairs": len(seen_pairs),
        "unique_hosts": len(unique_hosts),
        "pct_events_previously_unseen_pair": (unseen_pair_count / event_count * 100.0) if event_count else None,
        "pct_events_previously_unseen_destination": (unseen_dst_count / event_count * 100.0) if event_count else None,
        "pct_events_previously_unseen_source": (unseen_src_count / event_count * 100.0) if event_count else None,
    }

    serialized_bytes = len(json.dumps(events).encode("utf-8"))

    return {
        "micro_batch_seconds": MICRO_BATCH_SECONDS,
        "event_count": event_count,
        "compression_ratio_packets_per_event": compression_ratio_packets_per_event,
        "compression_ratio_definition": (
            "packets_per_event = total IP packets / B2 event count -- e.g. 3.0 means "
            "each B2 event represents 3 raw packets on average."
        ),
        "events_per_second": events_per_second,
        "maximum_burst_size": structure_stats["maximum_burst_size"],
        "temporal_resolution_seconds": MICRO_BATCH_SECONDS,
        "structure_stats": structure_stats,
        "serialized_size_bytes": serialized_bytes,
        "reconciliation": {
            "total_packets_via_events": total_packets_via_events,
            "total_packets_via_raw_parse": len(ip_packets),
            "packets_reconcile_exactly": total_packets_via_events == len(ip_packets),
            "total_bytes_via_events": total_bytes_via_events,
        },
        "_events": events,  # kept in-memory only; stripped before persisting the JSON report
    }


# ---------------------------------------------------------------------------
# Step 6: information preservation check (B2 vs raw packet stream)
# ---------------------------------------------------------------------------


def check_information_preservation(
    ip_packets: list[dict], b2: dict, anon_map: dict, capture_start_ts: float, capture_end_ts: float
) -> dict:
    events = b2["_events"]

    raw_bytes = sum(p["packet_length"] for p in ip_packets)
    raw_protocol_counts: dict[str, int] = defaultdict(int)
    raw_src_ports: set[int] = set()
    raw_dst_ports: set[int] = set()
    raw_hosts: set[str] = set()
    raw_pairs: set[tuple] = set()
    for p in ip_packets:
        raw_protocol_counts[p["protocol"]] += 1
        if p["src_port"] is not None:
            raw_src_ports.add(p["src_port"])
        if p["dst_port"] is not None:
            raw_dst_ports.add(p["dst_port"])
        src_anon = anon_map[p["src_ip"]]
        dst_anon = anon_map[p["dst_ip"]]
        raw_hosts.add(src_anon)
        raw_hosts.add(dst_anon)
        raw_pairs.add((src_anon, dst_anon))

    event_protocol_counts: dict[str, int] = defaultdict(int)
    event_src_ports: set[int] = set()
    event_dst_ports: set[int] = set()
    event_pairs: set[tuple] = set()
    event_hosts: set[str] = set()
    for e in events:
        event_protocol_counts[e["protocol"]] += e["packet_count"]
        if e["src_port"] is not None:
            event_src_ports.add(e["src_port"])
        if e["dst_port"] is not None:
            event_dst_ports.add(e["dst_port"])
        event_pairs.add((e["src"], e["dst"]))
        event_hosts.add(e["src"])
        event_hosts.add(e["dst"])

    return {
        "total_packets": {
            "raw": len(ip_packets),
            "via_events": sum(e["packet_count"] for e in events),
            "matches": len(ip_packets) == sum(e["packet_count"] for e in events),
        },
        "total_bytes": {
            "raw": raw_bytes,
            "via_events": sum(e["total_bytes"] for e in events),
            "matches": raw_bytes == sum(e["total_bytes"] for e in events),
        },
        "first_timestamp": capture_start_ts,
        "last_timestamp": capture_end_ts,
        "unique_hosts": {
            "raw": len(raw_hosts),
            "via_events": len(event_hosts),
            "matches": raw_hosts == event_hosts,
        },
        "unique_directed_pairs": {
            "raw": len(raw_pairs),
            "via_events": len(event_pairs),
            "matches": raw_pairs == event_pairs,
        },
        "protocol_counts": {
            "raw": dict(raw_protocol_counts),
            "via_events": dict(event_protocol_counts),
            "matches": dict(raw_protocol_counts) == dict(event_protocol_counts),
        },
        "port_diversity": {
            "raw_source_ports": len(raw_src_ports),
            "via_events_source_ports": len(event_src_ports),
            "source_matches": raw_src_ports == event_src_ports,
            "raw_destination_ports": len(raw_dst_ports),
            "via_events_destination_ports": len(event_dst_ports),
            "destination_matches": raw_dst_ports == event_dst_ports,
        },
        "information_lost_by_b2": [
            "Exact arrival order and individual timestamps of packets WITHIN one "
            f"micro-batch bucket ({MICRO_BATCH_SECONDS}s) are not retained -- only "
            "packet_count, mean_packet_length/std, and mean_IAT/std survive per bucket.",
            "TTL, IP fragmentation flags, and TCP window size (present on every raw "
            "packet, per Phase 9E's feature-availability table) are not carried into "
            "B2 event features in this phase -- they were not in the Step 4 required "
            "feature list and were not fabricated here.",
        ],
    }


# ---------------------------------------------------------------------------
# Step 8: hybrid representation (definition + measured size, no fusion)
# ---------------------------------------------------------------------------


def build_hybrid_representation(rep_a: dict, rep_b2: dict, total_windows: int) -> dict:
    """C = the B2 event stream + a per-window state summary (Representation
    A's node/edge features, minus full per-node/per-edge listings -- just
    the same per-window scalar statistics already computed in Phase 9E's
    per_window_distributions machinery), with events additionally indexed
    by which 10-second window they fall in. No neural fusion is defined or
    implemented; this only measures the data representation's size."""

    events = rep_b2["_events"]
    window_index_of_event = [int(e["event_timestamp_offset"] // WINDOW_SECONDS) for e in events]

    window_state_summary = []
    for w in range(total_windows):
        nodes = rep_a["_node_windows"].get(w, {})
        edges = rep_a["_edge_windows"].get(w, {})
        if not nodes and not edges:
            continue
        protocol_totals: dict[str, int] = defaultdict(int)
        dst_ports: set = set()
        src_ports: set = set()
        packet_total = 0
        byte_total = 0
        for edge in edges.values():
            packet_total += edge["packet_count"]
            byte_total += edge["total_bytes"]
            protocol_totals["TCP"] += edge["tcp_packet_count"]
            protocol_totals["UDP"] += edge["udp_packet_count"]
            protocol_totals["ICMP"] += edge["icmp_packet_count"]
            protocol_totals["OTHER"] += edge["other_packet_count"]
        for node_feat in nodes.values():
            dst_ports_count = node_feat.get("unique_destination_ports", 0)
            src_ports_count = node_feat.get("unique_source_ports", 0)
            dst_ports.add(dst_ports_count)
            src_ports.add(src_ports_count)
        window_state_summary.append(
            {
                "window_index": w,
                "active_hosts": len(nodes),
                "active_edges": len(edges),
                "packet_volume": packet_total,
                "byte_volume": byte_total,
                "protocol_distribution": dict(protocol_totals),
                "max_burst_in_window": max((edge["packet_count"] for edge in edges.values()), default=0),
            }
        )

    naive_combined_bytes = rep_a["serialized_size_bytes"] + rep_b2["serialized_size_bytes"]

    index_only = {"window_index_of_event": window_index_of_event}
    index_overhead_bytes = len(json.dumps(index_only).encode("utf-8"))
    window_state_bytes = len(json.dumps(window_state_summary).encode("utf-8"))
    lean_hybrid_bytes = rep_b2["serialized_size_bytes"] + window_state_bytes + index_overhead_bytes

    return {
        "definition": (
            "C = the full B2 micro-batch event stream, PLUS one compact per-"
            f"{WINDOW_SECONDS}s-window state summary (active_hosts, active_edges, "
            "packet/byte volume, protocol distribution, max burst size), PLUS a "
            "window_index per event so any consumer can slice the event stream by "
            "window without re-deriving window boundaries. No topology detail "
            "(which specific hosts/edges) is duplicated in the window summary -- "
            "that detail lives only in the event stream, avoided as redundant "
            "storage."
        ),
        "window_state_summary_count": len(window_state_summary),
        "window_state_summary_bytes": window_state_bytes,
        "event_to_window_index_overhead_bytes": index_overhead_bytes,
        "naive_combined_bytes_A_plus_B2": naive_combined_bytes,
        "lean_hybrid_bytes_B2_plus_window_summary_plus_index": lean_hybrid_bytes,
        "size_saving_vs_naive_pct": (
            (1 - lean_hybrid_bytes / naive_combined_bytes) * 100.0 if naive_combined_bytes else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-capture orchestration
# ---------------------------------------------------------------------------


def analyze_capture(label: str, pcap_path: Path) -> dict:
    if not pcap_path.exists():
        raise FileNotFoundError(f"{pcap_path} not found -- expected an already-validated Phase 9D/9F artifact.")

    parse_result = parse_full_capture(pcap_path)
    ip_packets = parse_result["ip_packets"]
    max_window_index = assign_windows(ip_packets, parse_result["capture_start_ts"])
    total_windows = max_window_index + 1

    raw_ips = {p["src_ip"] for p in ip_packets} | {p["dst_ip"] for p in ip_packets}
    anon_map, anon_validation = build_and_validate_anonymization(raw_ips)

    rep_a = build_representation_a(ip_packets, anon_map, total_windows)
    rep_b1 = build_representation_b1(ip_packets, anon_map)
    rep_b2 = build_representation_b2(ip_packets, anon_map, parse_result["capture_duration_seconds"])
    preservation = check_information_preservation(
        ip_packets, rep_b2, anon_map, parse_result["capture_start_ts"], parse_result["capture_end_ts"]
    )
    hybrid = build_hybrid_representation(rep_a, rep_b2, total_windows)

    b2_smaller_than_b1 = rep_b2["serialized_size_bytes"] < rep_b1["estimated_serialized_size_bytes"]
    storage_comparison_note = (
        f"OBSERVED: B2 ({rep_b2['event_count']:,} events, {rep_b2['serialized_size_bytes']:,} bytes) is "
        f"{'SMALLER' if b2_smaller_than_b1 else 'LARGER'} than B1 "
        f"({rep_b1['event_count']:,} events, {rep_b1['estimated_serialized_size_bytes']:,} bytes) in "
        f"serialized bytes, despite having fewer events. Reducing EVENT COUNT does not automatically "
        f"reduce STORAGE SIZE: each B2 event carries more fields than a B1 event (aggregate mean/std/"
        f"count fields vs. B1's raw per-packet fields), and this capture's median burst size is "
        f"{rep_b2['structure_stats']['median_burst_size']} packets/event -- when most events represent "
        f"only 1 packet, the extra aggregate fields are pure overhead with no compression benefit to "
        f"offset them."
    )

    capture_summary = {
        "total_packets_in_capture": parse_result["total_packets_in_capture"],
        "total_ip_packets": len(ip_packets),
        "capture_duration_seconds": parse_result["capture_duration_seconds"],
        "unique_hosts": len(raw_ips),
        "anonymization_bijective": anon_validation["bijective_mapping"],
    }

    return {
        "label": label,
        "pcap_path": str(pcap_path),
        "capture_summary": capture_summary,
        "representation_a": {k: v for k, v in rep_a.items() if not k.startswith("_")},
        "representation_b1": rep_b1,
        "representation_b2": {k: v for k, v in rep_b2.items() if not k.startswith("_")},
        "b2_vs_b1_storage_note": storage_comparison_note,
        "information_preservation": preservation,
        "representation_c_hybrid": hybrid,
    }


# ---------------------------------------------------------------------------
# Step 7: A vs B vs C comparison table (qualitative, grounded in measurements)
# ---------------------------------------------------------------------------


def build_comparison_table(capture_results: list[dict]) -> list[dict]:
    a_sizes = [c["representation_a"]["serialized_size_bytes"] for c in capture_results]
    b1_sizes = [c["representation_b1"]["estimated_serialized_size_bytes"] for c in capture_results]
    b2_sizes = [c["representation_b2"]["serialized_size_bytes"] for c in capture_results]
    c_sizes = [c["representation_c_hybrid"]["lean_hybrid_bytes_B2_plus_window_summary_plus_index"] for c in capture_results]

    a_lengths = [c["representation_a"]["total_windows"] for c in capture_results]
    b1_lengths = [c["representation_b1"]["event_count"] for c in capture_results]
    b2_lengths = [c["representation_b2"]["event_count"] for c in capture_results]

    return [
        {
            "property": "temporal resolution",
            "graph_snapshot": f"{WINDOW_SECONDS}s (fixed window)",
            "packet_events": "per-packet (microsecond, as captured)",
            "micro_batch_events": f"{MICRO_BATCH_SECONDS}s bucket + exact intra-bucket mean/std IAT",
            "hybrid": f"{MICRO_BATCH_SECONDS}s events + {WINDOW_SECONDS}s summary",
        },
        {
            "property": "variable topology handling",
            "graph_snapshot": "OBSERVED (Phase 9F): windows with different node/edge counts require padding/masking for fixed-size batching",
            "packet_events": "Naturally variable-length; no padding concept needed",
            "micro_batch_events": "Naturally variable-length; no padding concept needed",
            "hybrid": "Event stream variable-length; window summary is fixed-dim per window",
        },
        {
            "property": "sparsity handling",
            "graph_snapshot": f"OBSERVED (Phase 9F): {round(100*sum(c['representation_a']['non_empty_windows'] for c in capture_results)/sum(c['representation_a']['total_windows'] for c in capture_results),1)}% windows non-empty across both captures combined -- empty windows still occupy a sequence slot",
            "packet_events": "No empty slots by construction (a packet either happened or it did not appear)",
            "micro_batch_events": "No empty slots by construction",
            "hybrid": "Event stream has no empty slots; window summary still has empty-window slots",
        },
        {
            "property": "sequence length (this run)",
            "graph_snapshot": f"{a_lengths} windows",
            "packet_events": f"{b1_lengths} events",
            "micro_batch_events": f"{b2_lengths} events",
            "hybrid": f"{b2_lengths} events + {a_lengths} window summaries",
        },
        {
            "property": "measured storage (serialized JSON, this run)",
            "graph_snapshot": f"{a_sizes} bytes",
            "packet_events": f"{b1_sizes} bytes (estimated)",
            "micro_batch_events": f"{b2_sizes} bytes",
            "hybrid": f"{c_sizes} bytes (lean form)",
        },
        {
            "property": "packet information preservation",
            "graph_snapshot": "DERIVED: aggregated per-window only; individual packet timing lost",
            "packet_events": "CONFIRMED: complete, by construction (one event per packet)",
            "micro_batch_events": "OBSERVED (Step 6): totals/protocol/port-set reconcile exactly; intra-bucket ordering lost",
            "hybrid": "Same as micro-batch events (hybrid does not change the event layer)",
        },
        {
            "property": "topology information",
            "graph_snapshot": "CONFIRMED: explicit node/edge structure per window",
            "packet_events": "DERIVED: topology recoverable by replaying events, but not explicit",
            "micro_batch_events": "DERIVED: topology recoverable by replaying events, but not explicit",
            "hybrid": "CONFIRMED: window summary carries active_hosts/active_edges counts explicitly; specific edge identities remain in the event layer only",
        },
        {
            "property": "burst information",
            "graph_snapshot": "DERIVED: only via edge packet_count aggregated over the full 10s window",
            "packet_events": "Implicit only (would need to re-derive bursts from timestamps)",
            "micro_batch_events": f"CONFIRMED: burst size is a first-class per-event field (max/median/p95 reported directly, {MICRO_BATCH_SECONDS}s resolution)",
            "hybrid": "Same as micro-batch events, plus a window-level max_burst_in_window summary field",
        },
        {
            "property": "port behavior",
            "graph_snapshot": "OBSERVED: unique_destination_ports/unique_source_ports tracked per node per window (aggregated across all edges of that node)",
            "packet_events": "CONFIRMED: exact port pair retained per event",
            "micro_batch_events": "OBSERVED: port pair is part of the grouping key, so source/destination-port diversity per event is trivially 1 by construction -- port behavior is visible at the SEQUENCE level (which ports appear, how often), not as an internal per-event diversity statistic",
            "hybrid": "Same as micro-batch events",
        },
        {
            "property": "implementation complexity",
            "graph_snapshot": "Reused as-is from Phase 9E (no new code needed)",
            "packet_events": "Trivial (no aggregation logic)",
            "micro_batch_events": "Moderate (bucketing + aggregate feature computation, this phase's main new code)",
            "hybrid": "Moderate-plus (micro-batch logic + a thin per-window summary + an index) -- no fusion mechanism implemented",
        },
        {
            "property": "suitability for future world-model input",
            "graph_snapshot": "HYPOTHESIZED: matches representation B in the project's architecture diagram (graph encoder branch)",
            "packet_events": "HYPOTHESIZED: too fine-grained/long as a direct model input without further batching",
            "micro_batch_events": "HYPOTHESIZED: plausible direct sequence-model input given its bounded compression ratio and explicit burst/IAT fields",
            "hybrid": "HYPOTHESIZED: matches the project's intended 'FLOW STATE + PACKET GRAPH -> MULTIMODAL NETWORK STATE' pattern most closely, at the representation-definition level only -- NOT VERIFIED as better for prediction, since no model was trained",
        },
    ]


# ---------------------------------------------------------------------------
# Steps 10-11: narrative sections
# ---------------------------------------------------------------------------


def future_world_model_implications(capture_results: list[dict]) -> dict:
    return {
        "candidate_A_event_encoder_only": {
            "assessment": "HYPOTHESIZED",
            "note": (
                "Supported by B2's exact reconciliation with the raw packet stream "
                "(Step 6) and its bounded compression ratio -- an event encoder would "
                "see nearly all packet-level information. Not chosen outright because "
                "it discards the explicit, cheap-to-compute window-level topology "
                "summary that Representation A already provides for free."
            ),
        },
        "candidate_B_graph_encoder_only": {
            "assessment": "HYPOTHESIZED",
            "note": (
                "Phase 9F already showed graph richness is host-dependent (30.9%-80.5% "
                "non-empty windows) and windows stay small (max 16 edges) -- a "
                "graph-only encoder would be working with a highly variable, often-"
                "empty input, which Phase 9F itself flagged as a limitation."
            ),
        },
        "candidate_C_event_plus_window_fused": {
            "assessment": "HYPOTHESIZED -- most consistent with measured evidence so far",
            "note": (
                "This phase's Representation C (Step 8) is a concrete, measured "
                "instance of this candidate's INPUT side (not the fusion mechanism, "
                "which is out of scope here). It is the only candidate that keeps "
                "both the exact reconciling event information (Step 6) AND an "
                "explicit per-window topology summary, at a measured storage cost "
                "shown in Section 8 of the report."
            ),
        },
        "candidate_D_hierarchical_event_window_network": {
            "assessment": "HYPOTHESIZED",
            "note": (
                "A plausible extension of C with an additional network-wide "
                "aggregation level above the window level; this phase did not "
                "construct or measure a network-wide layer, so this candidate is "
                "less grounded in this phase's actual measurements than C."
            ),
        },
        "disclaimer": (
            "None of these candidates were built or evaluated as models. This "
            "section reports which candidate's INPUT-side data requirements are "
            "best supported by the measurements in Sections 3-9 of the report -- "
            "not which candidate would predict better."
        ),
    }


def novelty_analysis() -> dict:
    return {
        "existing_technique": [
            "Representing network traffic as a sequence of communication events "
            "(flow records, connection logs, NetFlow/IPFIX-style records) is "
            "long-established practice in network intrusion detection -- this is "
            "not a novel idea on its own.",
            "Micro-batching/aggregating packets sharing a 5-tuple within a small "
            "time bucket is structurally similar to how flow exporters already "
            "aggregate packets into flow records (e.g. this project's own "
            "FlowFeatureEngine, which segments by a timeout rather than a fixed "
            "bucket) -- the aggregation CONCEPT is not new.",
            "Fixed-window graph snapshots of host communication are also "
            "well-established in network graph analysis literature.",
        ],
        "our_potential_system_differentiator": [
            "NOT the event representation itself, but the specific combination "
            "targeted at this project's stated architecture: FLOW STATE + PACKET "
            "GRAPH -> MULTIMODAL NETWORK STATE -> WORLD MODEL -> FUTURE TRAJECTORY, "
            "with the packet-derived branch specifically designed (Representation "
            "C) to carry both exact reconciling event detail AND cheap topology "
            "summaries at a measured, bounded storage cost.",
            "The deterministic, per-phase-consistent anonymization scheme "
            "(same SHA-256-based mapping reused unmodified since Phase 9E) means "
            "events, graph snapshots, and any future hybrid representation all "
            "reference the SAME anonymized host identity across representations -- "
            "a coherence property a naive combination of off-the-shelf flow "
            "exporter output and a separately-built graph tool would not "
            "automatically have.",
        ],
        "not_yet_verified": [
            "Whether any of these representations improve multi-step attack "
            "trajectory forecasting over the existing flow-level World Model "
            "(Phase 6B) -- no model was trained here.",
            "Whether uncertainty quantification or counterfactual intervention "
                "are supportable by these representations -- not evaluated.",
            "Whether the observed host-dependent variability (Phase 9F) and the "
            "trivial per-event port diversity (Step 7 caveat) cause any specific "
            "practical difficulty for a real encoder -- only representation-level "
            "structure was measured, not encoder behavior.",
            "Generalization beyond these two single-host, single-day captures.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9G: Temporal Communication Event Representation")
    lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**")
    lines.append("")

    lines.append("## 1. EXECUTIVE RESULT")
    lines.append("")
    lines.append(report["executive_summary"])
    lines.append("")

    lines.append("## 2. INPUT CAPTURES")
    lines.append("")
    for c in report["captures"]:
        cs = c["capture_summary"]
        lines.append(
            f"- CONFIRMED: `{c['label']}` -- {cs['total_ip_packets']:,} IP packets, "
            f"{cs['unique_hosts']} unique hosts, {cs['capture_duration_seconds']:.1f}s "
            f"duration. Source: reused read-only, no network access this phase."
        )
    lines.append("")

    for c in report["captures"]:
        lines.append(f"## Capture: `{c['label']}`")
        lines.append("")

        a = c["representation_a"]
        lines.append("### 3. REPRESENTATION A -- GRAPH SNAPSHOTS")
        lines.append("")
        lines.append(f"- CONFIRMED: total_windows = {a['total_windows']}, non_empty_windows = {a['non_empty_windows']}")
        lines.append(f"- CONFIRMED: total_window_edges = {a['total_window_edges']}, unique_edges = {a['unique_edges']}")
        lines.append(f"- CONFIRMED: temporal_transitions_represented = {a['temporal_transitions_represented']}")
        lines.append(f"- CONFIRMED: serialized_size_bytes = {a['serialized_size_bytes']:,}")
        lines.append("")

        b1 = c["representation_b1"]
        lines.append("### 4. REPRESENTATION B1 -- PACKET EVENTS")
        lines.append("")
        lines.append(f"- CONFIRMED: event_count = {b1['event_count']:,} (one per IP packet, by construction)")
        lines.append(f"- DERIVED: estimated_serialized_size_bytes = {b1['estimated_serialized_size_bytes']:,}")
        lines.append("")

        b2 = c["representation_b2"]
        s = b2["structure_stats"]
        lines.append("### 5. REPRESENTATION B2 -- MICRO-BATCH EVENTS")
        lines.append("")
        lines.append(f"- CONFIRMED: micro_batch_seconds = {b2['micro_batch_seconds']} (fixed, not tuned)")
        lines.append(f"- OBSERVED: event_count = {b2['event_count']:,}")
        lines.append(f"- DERIVED: compression_ratio_packets_per_event = {b2['compression_ratio_packets_per_event']}")
        lines.append(f"- OBSERVED: events_per_second = {b2['events_per_second']}")
        lines.append(f"- OBSERVED: maximum_burst_size = {b2['maximum_burst_size']}")
        lines.append(f"- OBSERVED: median/p95 inter-event time = {s['median_inter_event_time']} / {s['p95_inter_event_time']}")
        lines.append(f"- OBSERVED: maximum_inter_event_gap = {s['maximum_inter_event_gap']}")
        lines.append(f"- OBSERVED: median/p95 burst size = {s['median_burst_size']} / {s['p95_burst_size']}")
        lines.append(f"- OBSERVED: unique_communicating_pairs = {s['unique_communicating_pairs']}, unique_hosts = {s['unique_hosts']}")
        lines.append(f"- OBSERVED: %events previously-unseen pair/destination/source = {s['pct_events_previously_unseen_pair']:.2f}% / {s['pct_events_previously_unseen_destination']:.2f}% / {s['pct_events_previously_unseen_source']:.2f}%")
        lines.append(f"- CONFIRMED: serialized_size_bytes = {b2['serialized_size_bytes']:,}")
        lines.append(f"- {c['b2_vs_b1_storage_note']}")
        lines.append("")

        pres = c["information_preservation"]
        lines.append("### 7. INFORMATION PRESERVATION (B2 vs raw packet stream)")
        lines.append("")
        lines.append(f"- CONFIRMED: total_packets reconcile exactly = {pres['total_packets']['matches']} ({pres['total_packets']['via_events']:,} / {pres['total_packets']['raw']:,})")
        lines.append(f"- CONFIRMED: total_bytes reconcile exactly = {pres['total_bytes']['matches']}")
        lines.append(f"- CONFIRMED: unique_hosts reconcile exactly = {pres['unique_hosts']['matches']} ({pres['unique_hosts']['via_events']} / {pres['unique_hosts']['raw']})")
        lines.append(f"- CONFIRMED: unique_directed_pairs reconcile exactly = {pres['unique_directed_pairs']['matches']} ({pres['unique_directed_pairs']['via_events']} / {pres['unique_directed_pairs']['raw']})")
        lines.append(f"- CONFIRMED: protocol_counts reconcile exactly = {pres['protocol_counts']['matches']}")
        lines.append(f"- CONFIRMED: source/destination port sets reconcile exactly = {pres['port_diversity']['source_matches']} / {pres['port_diversity']['destination_matches']}")
        lines.append("- Information lost by B2:")
        for item in pres["information_lost_by_b2"]:
            lines.append(f"  - {item}")
        lines.append("")

        hy = c["representation_c_hybrid"]
        lines.append("### 6. REPRESENTATION C -- HYBRID")
        lines.append("")
        lines.append(hy["definition"])
        lines.append("")
        lines.append(f"- OBSERVED: window_state_summary_count = {hy['window_state_summary_count']}")
        lines.append(f"- OBSERVED: window_state_summary_bytes = {hy['window_state_summary_bytes']:,}")
        lines.append(f"- OBSERVED: naive_combined_bytes (A + B2) = {hy['naive_combined_bytes_A_plus_B2']:,}")
        lines.append(f"- OBSERVED: lean_hybrid_bytes (B2 + window summary + index) = {hy['lean_hybrid_bytes_B2_plus_window_summary_plus_index']:,}")
        lines.append(f"- DERIVED: size_saving_vs_naive_pct = {hy['size_saving_vs_naive_pct']:.1f}%")
        lines.append("")

    lines.append("## 8. COMPLEXITY / STORAGE SUMMARY")
    lines.append("")
    lines.append("See per-capture Sections 3-6 above and the comparison table below (measured storage row).")
    lines.append("")

    lines.append("## 9. A vs B vs C COMPARISON")
    lines.append("")
    lines.append("| property | graph snapshot (A) | packet events (B1) | micro-batch events (B2) | hybrid (C) |")
    lines.append("|---|---|---|---|---|")
    for row in report["comparison_table"]:
        lines.append(f"| {row['property']} | {row['graph_snapshot']} | {row['packet_events']} | {row['micro_batch_events']} | {row['hybrid']} |")
    lines.append("")

    lines.append("## 10. FUTURE WORLD MODEL IMPLICATIONS")
    lines.append("")
    for key, val in report["future_world_model_implications"].items():
        if key == "disclaimer":
            continue
        lines.append(f"- **{key}**: {val['assessment']}")
        lines.append(f"  {val['note']}")
    lines.append("")
    lines.append(report["future_world_model_implications"]["disclaimer"])
    lines.append("")

    lines.append("## 11. NOVELTY IMPLICATIONS")
    lines.append("")
    n = report["novelty_analysis"]
    lines.append("**EXISTING TECHNIQUE:**")
    for item in n["existing_technique"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**OUR POTENTIAL SYSTEM DIFFERENTIATOR:**")
    for item in n["our_potential_system_differentiator"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**NOT YET VERIFIED:**")
    for item in n["not_yet_verified"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 12. LIMITATIONS")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 13. RECOMMENDED NEXT EXPERIMENT")
    lines.append("")
    lines.append(report["recommended_next_experiment"])
    lines.append("")

    lines.append(f"## FINAL VERDICT: {report['verdict']}")
    lines.append("")
    lines.append("STOP AFTER PHASE 9G.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9G directory: {OUTPUT_DIR}")

    capture_results = []
    for cap in CAPTURES:
        result = analyze_capture(cap["label"], cap["pcap_path"])
        result["source_phase"] = cap["source_phase"]
        capture_results.append(result)

    comparison_table = build_comparison_table(capture_results)
    world_model = future_world_model_implications(capture_results)
    novelty = novelty_analysis()

    reconciliation_ok = all(
        c["information_preservation"]["total_packets"]["matches"]
        and c["information_preservation"]["total_bytes"]["matches"]
        and c["information_preservation"]["protocol_counts"]["matches"]
        and c["information_preservation"]["unique_hosts"]["matches"]
        and c["information_preservation"]["unique_directed_pairs"]["matches"]
        and c["information_preservation"]["port_diversity"]["source_matches"]
        and c["information_preservation"]["port_diversity"]["destination_matches"]
        for c in capture_results
    )
    b2_event_counts = [c["representation_b2"]["event_count"] for c in capture_results]
    non_empty_fracs = [
        c["representation_a"]["non_empty_windows"] / c["representation_a"]["total_windows"]
        for c in capture_results
    ]

    b2_smaller_than_b1_flags = [
        c["representation_b2"]["serialized_size_bytes"] < c["representation_b1"]["estimated_serialized_size_bytes"]
        for c in capture_results
    ]
    executive_summary = (
        f"Two already-validated PCAP captures ({', '.join(c['label'] for c in capture_results)}) were "
        f"re-parsed with the identical Phase 9E parsing pipeline (no network access). Representation B2 "
        f"(micro-batch events, {MICRO_BATCH_SECONDS}s buckets) reconciles EXACTLY against the raw packet "
        f"stream on every checked quantity (packet count, byte count, protocol counts, port sets) in both "
        f"captures: {reconciliation_ok}. B2 reduces "
        f"{[c['capture_summary']['total_ip_packets'] for c in capture_results]} raw packets to "
        f"{b2_event_counts} events respectively, while the graph-snapshot representation (A) stayed only "
        f"{[f'{f:.1%}' for f in non_empty_fracs]} non-empty (matching Phase 9F's finding). NOTABLY, fewer "
        f"events does NOT mean smaller storage: B2 is smaller than B1 in serialized bytes in "
        f"{sum(b2_smaller_than_b1_flags)}/{len(b2_smaller_than_b1_flags)} captures -- see the per-capture "
        f"storage notes in Section 5 for why (median burst size near 1 in both captures leaves little for "
        f"per-event aggregate fields to compress)."
    )

    limitations = [
        "Only the same two single-host, single-day captures already used in Phases 9D-9F were examined; "
        "no new PCAP was downloaded per hard constraint #10, so generalization beyond these two hosts is "
        "NOT VERIFIED.",
        "No model of any kind was trained or evaluated; all 'suitability for future world-model input' "
        "and Step 10 candidate assessments are HYPOTHESIZED, not measured predictive results.",
        f"Source/destination-port diversity is trivially 1 for every B2 event by construction (ports are "
        f"part of the grouping key) -- this is reported explicitly rather than silently included as if "
        f"informative at the per-event level.",
        "The hybrid representation (C) is defined and size-measured only; no fusion mechanism (neural or "
        "otherwise) was implemented, per hard constraint #6.",
        "TTL, IP fragmentation, and TCP window size are not carried into B2 event features (not in the "
        "required Step 4 feature list); this is a scoping choice, not evidence they are unavailable -- "
        "Phase 9E already classified them DIRECT and available.",
        f"B2's per-event compression ratio (packets-per-event) was modest in both captures "
        f"({[round(c['representation_b2']['compression_ratio_packets_per_event'], 2) for c in capture_results]}x) "
        f"because median burst size is 1.0 packet/event in both -- most 1-second buckets never accumulate "
        f"more than one packet for a given (src, dst, protocol, port pair) 5-tuple. As a direct consequence, "
        f"B2's serialized size is LARGER than B1's raw per-packet events in "
        f"{len(capture_results) - sum(b2_smaller_than_b1_flags)}/{len(capture_results)} captures -- reducing "
        f"event count does not automatically reduce storage when per-event overhead exceeds the compression "
        f"gained.",
    ]

    recommended_next_experiment = (
        "If pursued further (not started here): implement candidate C's fusion mechanism as a small, "
        "non-neural baseline first (e.g. concatenating a window's event-derived statistics with its "
        "window-summary vector) and compare information content against the flow-level World Model's "
        "existing state representation on a held-out capture, before committing to any neural encoder "
        "architecture."
    )

    if reconciliation_ok and min(b2_event_counts) > 0:
        verdict = "GREEN"
    elif min(b2_event_counts) > 0:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    report = {
        "success": True,
        "verdict": verdict,
        "executive_summary": executive_summary,
        "micro_batch_seconds": MICRO_BATCH_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "captures": capture_results,
        "comparison_table": comparison_table,
        "future_world_model_implications": world_model,
        "novelty_analysis": novelty,
        "limitations": limitations,
        "recommended_next_experiment": recommended_next_experiment,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "event_representation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "event_representation_report.md", report)

    print(
        json.dumps(
            {
                "success": True,
                "verdict": verdict,
                "reconciliation_ok": reconciliation_ok,
                "b2_event_counts": b2_event_counts,
                "captures": [c["label"] for c in capture_results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
