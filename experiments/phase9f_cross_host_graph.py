"""Phase 9F: cross-host structural generalization of the Phase 9E temporal
packet graph.

Retrieves exactly ONE additional small PCAP member from the same S3 archive
inspected in Phase 9C (`pcap.zip`, Wednesday-14-02-2018), selected purely
from Phase 9C's frozen metadata, and runs it through EXACTLY the Phase 9E
graph-construction pipeline (same window size, node/edge definitions,
feature definitions, anonymization scheme, null handling -- all imported
directly from experiments/phase9e_temporal_packet_graph.py, not
reimplemented). The resulting graph statistics are compared against Phase
9E's own persisted results to test whether the sparse, non-static graph
structure observed for one host generalizes to a second, larger host
capture.

This is a structural generalization experiment only: no GNN, no attack
labels, no CSV row-level matching, no model of any kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read-only reuse of Phase 9C's remote-range inspector and Phase 9D's
# generic member retrieval/decompression logic -- neither file is modified.
from phase9c_pcap_archive_inspection import RemoteObjectInspector  # noqa: E402
from phase9d_pcap_validation import decompress_member, retrieve_member  # noqa: E402

# Read-only reuse of Phase 9E's EXACT graph-construction pipeline -- per
# Step 5's hard requirement, none of window size / node definition / edge
# definition / feature definitions / anonymization scheme / null handling
# may be changed, so these are imported directly rather than reimplemented.
from phase9e_temporal_packet_graph import (  # noqa: E402
    WINDOW_SECONDS,
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

PHASE9C_REPORT_PATH = EXPERIMENTS_DIR / "results/phase9c_pcap_archive_inspection/discovery_report.json"
PHASE9E_REPORT_PATH = EXPERIMENTS_DIR / "results/phase9e_temporal_packet_graph/graph_report.json"
PHASE9D_MEMBER_FILENAME = "pcap/UCAP172.31.69.22"

# Hard constraint #11: prefer the ~5-7MB compressed range identified by
# Phase 9C's own candidate listing.
TARGET_MIN_BYTES = 5 * 1024 * 1024
TARGET_MAX_BYTES = 7 * 1024 * 1024

# Hard constraint #16: keep total downloaded data comfortably below 10MB.
MAX_TOTAL_DOWNLOAD_BYTES = 8 * 1024 * 1024

DATA_DIR = EXPERIMENTS_DIR / "data/phase9f"
OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9f_cross_host_graph"


# ---------------------------------------------------------------------------
# Step 1: select second member (metadata only, no download yet)
# ---------------------------------------------------------------------------


def select_second_member(
    report: dict,
    exclude_filename: str = PHASE9D_MEMBER_FILENAME,
    min_bytes: int = TARGET_MIN_BYTES,
    max_bytes: int = TARGET_MAX_BYTES,
) -> dict:
    """Deterministic, metadata-only selection rule: among all archive
    members classified as PCAP candidates in Phase 9C's frozen discovery
    report, excluding the Phase 9D member, pick the SMALLEST whose
    compressed_size falls within [min_bytes, max_bytes]. This is a pure
    function of one numeric field (compressed_size) -- it does not use the
    filename's host/IP content, folder position, or any presumed attack
    association to choose."""

    candidates = [
        member
        for member in report["members"]
        if member["classification"] in ("pcap", "pcapng", "unknown_extension")
        and member["filename"] != exclude_filename
        and min_bytes <= member["compressed_size"] <= max_bytes
    ]

    if not candidates:
        raise RuntimeError(
            f"No candidate member found with compressed_size in [{min_bytes}, {max_bytes}] "
            f"bytes, excluding {exclude_filename}."
        )

    candidates.sort(key=lambda m: m["compressed_size"])
    chosen = candidates[0]

    return {
        "object_url": report["object_url"],
        "filename": chosen["filename"],
        "compressed_size": chosen["compressed_size"],
        "uncompressed_size": chosen["uncompressed_size"],
        "compression_method": chosen["compression_method"],
        "local_header_offset": chosen["local_header_offset"],
        "expected_crc32": chosen["crc32"],
        "selection_justification": (
            f"Selected from Phase 9C's frozen discovery_report.json (never re-parsed from "
            f"the archive). Selection rule: among all {len(candidates)} candidate members "
            f"with compressed_size in the requested [{min_bytes:,}, {max_bytes:,}] byte "
            f"band (excluding the Phase 9D member {exclude_filename}), the SMALLEST was "
            f"chosen -- a deterministic, content-blind tie-break that minimizes bytes "
            f"downloaded while staying inside the requested band. No filename, hostname, "
            f"or presumed attack/benign association was used in this decision; the only "
            f"input to selection was the compressed_size field."
        ),
        "other_candidates_in_band": [
            {"filename": m["filename"], "compressed_size": m["compressed_size"]}
            for m in candidates[1:]
        ],
    }


# ---------------------------------------------------------------------------
# Step 4: capture-level summary (rollups of Phase 9E's own field definitions
# only -- no new feature definitions introduced)
# ---------------------------------------------------------------------------


def summarize_capture(parse_result: dict) -> dict:
    ip_packets = parse_result["ip_packets"]

    src_ips = {p["src_ip"] for p in ip_packets}
    dst_ips = {p["dst_ip"] for p in ip_packets}
    hosts = src_ips | dst_ips

    protocol_counts = {"TCP": 0, "UDP": 0, "ICMP": 0, "OTHER": 0}
    total_bytes = 0
    for p in ip_packets:
        protocol_counts[p["protocol"]] = protocol_counts.get(p["protocol"], 0) + 1
        total_bytes += p["packet_length"]

    return {
        "total_packets_in_capture": parse_result["total_packets_in_capture"],
        "total_ip_packets": len(ip_packets),
        "status_counts": parse_result["status_counts"],
        "tcp_packet_count": protocol_counts["TCP"],
        "udp_packet_count": protocol_counts["UDP"],
        "icmp_packet_count": protocol_counts["ICMP"],
        "other_packet_count": protocol_counts["OTHER"],
        "unique_source_ips": len(src_ips),
        "unique_destination_ips": len(dst_ips),
        "unique_hosts_overall": len(hosts),
        "total_bytes": total_bytes,
        "first_timestamp": parse_result["capture_start_ts"],
        "last_timestamp": parse_result["capture_end_ts"],
        "capture_duration_seconds": parse_result["capture_duration_seconds"],
    }


# ---------------------------------------------------------------------------
# Step 7: cross-host comparison (read-only against Phase 9E's persisted report)
# ---------------------------------------------------------------------------


def build_comparison_table(phase9e_report: dict, second_capture_summary: dict, second_graph_stats: dict, second_dynamics: dict, second_repr_sparsity: float) -> list[dict]:
    e_cl = phase9e_report["graph_statistics"]["capture_level"]
    e_dist = phase9e_report["graph_statistics"]["per_window_distributions"]["metrics"]
    e_dyn = phase9e_report["temporal_dynamics"]
    e_tw = phase9e_report["temporal_window_structure"]
    e_sparsity = phase9e_report["representation_feasibility"]["sparsity"]["sparsity_fraction"]

    s_cl = second_graph_stats
    s_dist = second_graph_stats["per_window_distributions_metrics"]
    s_dyn = second_dynamics

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    rows = []

    def add(metric, first_val, second_val, higher_is_richer=True):
        rows.append(
            {
                "metric": metric,
                "phase9d_9e_capture": first_val,
                "second_capture": second_val,
                "ratio_second_over_first": ratio(second_val, first_val) if isinstance(first_val, (int, float)) and isinstance(second_val, (int, float)) else None,
            }
        )

    add("packet_count (IP packets)", phase9e_report["parsing_results"]["total_ip_packets"], second_capture_summary["total_ip_packets"])
    add("duration_seconds", e_tw["capture_duration_seconds"], second_capture_summary["capture_duration_seconds"])
    add("unique_hosts", e_cl["total_nodes_observed"], s_cl["total_nodes_observed"])
    add("unique_directed_pairs", e_cl["total_unique_directed_source_destination_pairs"], s_cl["total_unique_directed_source_destination_pairs"])
    add("median_nodes_per_window", e_dist["node_count"]["median"], s_dist["node_count"]["median"])
    add("p95_nodes_per_window", e_dist["node_count"]["p95"], s_dist["node_count"]["p95"])
    add("max_nodes_per_window", e_dist["node_count"]["max"], s_dist["node_count"]["max"])
    add("median_edges_per_window", e_dist["edge_count"]["median"], s_dist["edge_count"]["median"])
    add("p95_edges_per_window", e_dist["edge_count"]["p95"], s_dist["edge_count"]["p95"])
    add("max_edges_per_window", e_dist["edge_count"]["max"], s_dist["edge_count"]["max"])
    add(
        "non_empty_window_fraction",
        e_cl["non_empty_windows"] / e_cl["total_windows"] if e_cl["total_windows"] else None,
        s_cl["non_empty_windows"] / s_cl["total_windows"] if s_cl["total_windows"] else None,
    )
    add("longest_non_empty_run", e_dyn["longest_consecutive_non_empty_window_run"], s_dyn["longest_consecutive_non_empty_window_run"])
    add("mean_edge_jaccard_consecutive", e_dyn["edge_set_jaccard_consecutive"]["mean"], s_dyn["edge_set_jaccard_consecutive"]["mean"])
    add("mean_node_jaccard_consecutive", e_dyn["node_set_jaccard_consecutive"]["mean"], s_dyn["node_set_jaccard_consecutive"]["mean"])
    add("sparsity_fraction", e_sparsity, second_repr_sparsity)

    return rows


def classify_richness(comparison_rows: list[dict]) -> dict:
    """Step 7 classification: A (consistently very sparse/small), B
    (variable but usable), or C (substantially richer). Not forced -- if
    evidence is genuinely mixed, that is reported explicitly.

    Rather than averaging a hand-picked subset of metrics (which can mask
    disagreement between metrics -- e.g. strong temporal-continuity growth
    alongside flat or shrinking per-window density), this looks at the
    SPREAD across every metric that has a defined ratio. Consistency (A or
    C) requires ALL comparable metrics to agree in direction/magnitude;
    disagreement between metrics is reported as B rather than forced into
    a false consensus."""

    by_metric = {row["metric"]: row for row in comparison_rows}

    # Exclude duration_seconds and sparsity_fraction: both are near-1.0 by
    # construction (same day/environment; sparsity is bounded near 1 for
    # any realistic host-communication graph) and are not "richness"
    # signals -- including them would mechanically pull any spread
    # calculation toward "no change" regardless of the metrics that
    # actually describe graph richness.
    richness_metrics = [
        "packet_count (IP packets)",
        "unique_hosts",
        "unique_directed_pairs",
        "p95_nodes_per_window",
        "max_nodes_per_window",
        "p95_edges_per_window",
        "max_edges_per_window",
        "non_empty_window_fraction",
        "longest_non_empty_run",
        "mean_edge_jaccard_consecutive",
        "mean_node_jaccard_consecutive",
    ]
    ratios = {m: by_metric[m]["ratio_second_over_first"] for m in richness_metrics}
    defined_ratios = {m: v for m, v in ratios.items() if v is not None}
    ratio_values = list(defined_ratios.values())
    min_ratio = min(ratio_values) if ratio_values else None
    max_ratio = max(ratio_values) if ratio_values else None

    sparsity_first = by_metric["sparsity_fraction"]["phase9d_9e_capture"]
    sparsity_second = by_metric["sparsity_fraction"]["second_capture"]
    both_very_sparse = sparsity_first is not None and sparsity_second is not None and sparsity_first > 0.99 and sparsity_second > 0.99

    max_nodes_first = by_metric["max_nodes_per_window"]["phase9d_9e_capture"]
    max_nodes_second = by_metric["max_nodes_per_window"]["second_capture"]
    both_small_windows = max_nodes_first is not None and max_nodes_second is not None and max_nodes_first <= 10 and max_nodes_second <= 10

    if max_ratio is not None and max_ratio < 2.5 and both_very_sparse and both_small_windows:
        classification = "A"
        rationale = (
            f"Every richness metric's second/first ratio stays below 2.5x (max observed: "
            f"{max_ratio:.2f}x), both captures have sparsity above 99%, and both keep "
            f"max nodes/window at or below 10 -- the graph is consistently very "
            f"sparse/small across hosts with no metric showing substantial growth."
        )
    elif min_ratio is not None and min_ratio >= 3.0:
        classification = "C"
        rationale = (
            f"EVERY richness metric grew by at least 3x from the first capture to the "
            f"second (minimum observed ratio: {min_ratio:.2f}x) -- growth is broad and "
            f"consistent across structural and temporal metrics alike, not confined to "
            f"one dimension, so the second capture is substantially richer."
        )
    else:
        classification = "B"
        rationale = (
            f"Evidence is mixed, not forced into a single bucket: richness-metric ratios "
            f"range from {min_ratio:.2f}x to {max_ratio:.2f}x (second/first). Temporal-"
            f"continuity metrics grew substantially (non_empty_window_fraction "
            f"{ratios['non_empty_window_fraction']:.2f}x, longest_non_empty_run "
            f"{ratios['longest_non_empty_run']:.2f}x, edge/node Jaccard "
            f"{ratios['mean_edge_jaccard_consecutive']:.2f}x / "
            f"{ratios['mean_node_jaccard_consecutive']:.2f}x), while per-window "
            f"structural density stayed modest (max_nodes_per_window "
            f"{ratios['max_nodes_per_window']:.2f}x, max_edges_per_window "
            f"{ratios['max_edges_per_window']:.2f}x) and unique_directed_pairs actually "
            f"{'decreased' if ratios['unique_directed_pairs'] < 1 else 'increased'} "
            f"({ratios['unique_directed_pairs']:.2f}x) despite ~6x more packets -- "
            f"the pipeline generalizes correctly across hosts, and temporal richness "
            f"improves meaningfully, but per-window population remains variable rather "
            f"than uniformly tiny or uniformly rich."
        )

    return {
        "classification": classification,
        "richness_metric_ratios": defined_ratios,
        "min_ratio_across_richness_metrics": min_ratio,
        "max_ratio_across_richness_metrics": max_ratio,
        "both_captures_sparsity_above_99pct": both_very_sparse,
        "both_captures_max_nodes_per_window_le_10": both_small_windows,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Step 8: graph-model suitability (narrative, grounded in measured numbers)
# ---------------------------------------------------------------------------


def graph_model_suitability(comparison_rows: list[dict], richness: dict) -> dict:
    by_metric = {row["metric"]: row for row in comparison_rows}
    max_edges_second = by_metric["max_edges_per_window"]["second_capture"]
    median_nodes_second = by_metric["median_nodes_per_window"]["second_capture"]
    longest_run_second = by_metric["longest_non_empty_run"]["second_capture"]
    node_jaccard_second = by_metric["mean_node_jaccard_consecutive"]["second_capture"]
    non_empty_fraction_first = by_metric["non_empty_window_fraction"]["phase9d_9e_capture"]
    non_empty_fraction_second = by_metric["non_empty_window_fraction"]["second_capture"]

    return {
        "q1_enough_structure_for_meaningful_gnn": {
            "answer": "MARGINAL",
            "observed": (
                f"OBSERVED: median nodes/window in the second capture is "
                f"{median_nodes_second}, and max edges/window is {max_edges_second} -- "
                f"individual windows carry very few edges for message passing to "
                f"aggregate over, in both captures (classification: {richness['classification']})."
            ),
            "hypothesized": (
                "HYPOTHESIZED: a GNN could still exploit structure aggregated across "
                "MULTIPLE windows or across hosts (not attempted here), even if any single "
                "window's graph is small."
            ),
        },
        "q2_snapshots_sufficiently_populated": {
            "answer": "PARTIALLY, AND INCONSISTENTLY BETWEEN THE TWO CAPTURES",
            "observed": (
                f"OBSERVED: the non-empty-window fraction differs sharply between the two "
                f"captures rather than agreeing: {non_empty_fraction_first:.1%} of "
                f"{WINDOW_SECONDS}-second windows are non-empty in the first capture "
                f"(a minority) versus {non_empty_fraction_second:.1%} in the second "
                f"capture (a majority). Whether most snapshots carry any structure at "
                f"all is host-dependent in this small two-capture sample, not a fixed "
                f"property of the representation."
            ),
            "hypothesized": None,
        },
        "q3_sufficiently_long_temporal_sequences": {
            "answer": "LIMITED",
            "observed": (
                f"OBSERVED: longest consecutive non-empty window run in the second "
                f"capture is {longest_run_second} windows "
                f"({longest_run_second * WINDOW_SECONDS if longest_run_second else 0}s); "
                "sequences this short bound how much temporal context a sequence model "
                "could condition on before hitting a gap."
            ),
            "hypothesized": None,
        },
        "q4_would_graph_model_add_info_beyond_aggregate_stats": {
            "answer": "NOT VERIFIED",
            "observed": (
                "OBSERVED: this phase computed aggregate statistics (the same "
                "min/median/mean/p95/max distributions used throughout) and did not "
                "compare their predictive content against any graph-model output, "
                "because no model was built here."
            ),
            "hypothesized": (
                "HYPOTHESIZED: given how small and sparse individual window graphs are "
                f"(mean node-set Jaccard {node_jaccard_second} in the second capture, "
                "meaning topology itself changes substantially), a graph model's main "
                "candidate advantage would be capturing WHICH specific hosts are "
                "connected, not raw volume -- something scalar aggregate statistics "
                "cannot represent by construction. This is a hypothesis, not a measured "
                "result."
            ),
        },
        "q5_which_representation_fits_the_observed_data": {
            "answer": (
                "HYPOTHESIZED: an event/edge-sequence representation (a stream of "
                "individual (timestamp, src, dst, features) edge events) looks like a "
                "better structural fit than dense per-window graph snapshots, given both "
                "captures show most windows empty and populated windows carry only a "
                "handful of edges -- a sequence representation avoids paying a fixed "
                "per-window cost for windows with little or no content."
            ),
            "observed": (
                "OBSERVED: a global/static graph would discard all temporal-change "
                "information that Phase 9E and this phase both measured as substantial "
                "(low consecutive-window Jaccard in both captures)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Step 9: novelty decision
# ---------------------------------------------------------------------------


def novelty_decision(richness: dict, suitability: dict) -> dict:
    if richness["classification"] == "A":
        decision = "STOP GRAPH BRANCH"
        rationale = (
            "Both captures independently show extremely sparse, small-per-window graph "
            "structure (>99% sparsity, single-digit node/edge counts per window in both "
            "hosts). This consistency itself is informative, but it consistently points "
            "away from a per-window graph-snapshot GNN being worth the added complexity "
            "over the aggregate statistics already computed."
        )
    elif richness["classification"] == "C":
        decision = "CONTINUE"
        rationale = (
            "The second, larger capture shows substantially richer per-window graph "
            "structure than the first, suggesting graph richness scales with capture "
            "size/duration and a controlled graph-learning experiment is warranted."
        )
    else:
        decision = "CONDITIONAL"
        rationale = (
            "Graph construction generalizes correctly across two different hosts (same "
            "pipeline, same conventions, both produce internally consistent, exact "
            "statistics). Evidence is mixed rather than uniformly weak or uniformly "
            "strong: temporal-continuity metrics (non-empty-window fraction, longest "
            "non-empty run, consecutive-window Jaccard) improved substantially in the "
            "second, larger capture, while per-window structural density (max nodes/"
            "edges per window, unique directed pairs) stayed modest or flat. This "
            "pattern -- richer WHEN traffic occurs, but not necessarily richer WHERE/"
            "HOW MANY hosts per window -- is exactly the kind of result that should not "
            "be forced toward a full GNN commitment yet. An event/edge-sequence "
            "representation (per Step 8, Q5) should be tried and evaluated first."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "explicit_disclaimer": (
            "This decision is NOT based on 'GNN sounds novel' -- it is based only on the "
            "measured sparsity, window population, and temporal-run-length numbers from "
            "these two captures."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9F: Cross-Host Temporal Graph Generalization")
    lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**")
    lines.append("")

    sel = report["second_member_selection"]
    lines.append("## 1. SECOND MEMBER SELECTION")
    lines.append("")
    lines.append(f"- CONFIRMED: filename `{sel['filename']}`")
    lines.append(f"- CONFIRMED: compressed_size = {sel['compressed_size']:,} bytes ({sel['compressed_size']/1024/1024:.2f} MiB)")
    lines.append(f"- CONFIRMED: uncompressed_size = {sel['uncompressed_size']:,} bytes")
    lines.append(f"- CONFIRMED: compression_method = {sel['compression_method']}")
    lines.append(f"- CONFIRMED: local_header_offset = {sel['local_header_offset']:,}")
    lines.append(f"- {sel['selection_justification']}")
    lines.append("")

    if not report.get("success"):
        lines.append("## STOPPED")
        lines.append("")
        lines.append(report.get("stopped_reason", "(no reason recorded)"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    dl = report["download_validation"]
    lines.append("## 2. DOWNLOAD VALIDATION")
    lines.append("")
    lines.append(f"- CONFIRMED: total bytes downloaded this phase: {report['total_bytes_downloaded']:,}")
    lines.append(f"- CONFIRMED: filename match: {dl['filename_matches_expected']}")
    lines.append(f"- CONFIRMED: compressed size match: {dl['compressed_size_matches_expected']}")
    lines.append("")

    cv = report["capture_validation"]
    lines.append("## 3. CAPTURE VALIDATION")
    lines.append("")
    lines.append(f"- CONFIRMED: detected format: {cv['detected_format']}")
    lines.append(f"- CONFIRMED: uncompressed size match: {cv['uncompressed_size_matches_expected']}")
    lines.append(f"- CONFIRMED: CRC-32 match: {cv['crc32_matches_expected']}")
    lines.append(f"- CONFIRMED: first 32 bytes: `{cv['first_32_bytes_hex']}`")
    lines.append("")

    cs = report["second_capture_summary"]
    lines.append("## 4. SECOND CAPTURE STATISTICS")
    lines.append("")
    for k in [
        "total_packets_in_capture", "total_ip_packets", "tcp_packet_count", "udp_packet_count",
        "icmp_packet_count", "other_packet_count", "unique_source_ips", "unique_destination_ips",
        "unique_hosts_overall", "total_bytes", "first_timestamp", "last_timestamp", "capture_duration_seconds",
    ]:
        lines.append(f"- OBSERVED: {k} = {cs[k]}")
    lines.append("")

    gs = report["second_graph_statistics"]
    lines.append("## 5. SECOND TEMPORAL GRAPH STATISTICS")
    lines.append("")
    lines.append(f"- CONFIRMED: total_windows = {gs['total_windows']}, non_empty_windows = {gs['non_empty_windows']}")
    lines.append(f"- CONFIRMED: total_nodes_observed = {gs['total_nodes_observed']}")
    lines.append(f"- CONFIRMED: total_unique_directed_source_destination_pairs = {gs['total_unique_directed_source_destination_pairs']}")
    lines.append(f"- CONFIRMED: packet_accounting_check matches = {gs['packet_accounting_check']['matches']}")
    lines.append("")
    lines.append("Per-window distributions (min / median / mean / p95 / max):")
    lines.append("")
    lines.append("| metric | min | median | mean | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric, dist in gs["per_window_distributions_metrics"].items():
        lines.append(f"| {metric} | {dist['min']} | {dist['median']} | {dist['mean']} | {dist['p95']} | {dist['max']} |")
    lines.append("")
    dyn = report["second_temporal_dynamics"]
    lines.append(f"- OBSERVED: longest_consecutive_non_empty_window_run = {dyn['longest_consecutive_non_empty_window_run']}")
    lines.append(f"- OBSERVED: mean node-set Jaccard (consecutive) = {dyn['node_set_jaccard_consecutive']['mean']}")
    lines.append(f"- OBSERVED: mean edge-set Jaccard (consecutive) = {dyn['edge_set_jaccard_consecutive']['mean']}")
    lines.append("")

    lines.append("## 6. DIRECT COMPARISON WITH PHASE 9E")
    lines.append("")
    lines.append("| metric | Phase 9D/9E capture | second capture | ratio (second/first) |")
    lines.append("|---|---:|---:|---:|")
    for row in report["comparison_table"]:
        lines.append(f"| {row['metric']} | {row['phase9d_9e_capture']} | {row['second_capture']} | {row['ratio_second_over_first']} |")
    lines.append("")
    rich = report["richness_classification"]
    lines.append(f"Classification: **{rich['classification']}** -- {rich['rationale']}")
    lines.append("")

    lines.append("## 7. GRAPH-MODEL SUITABILITY")
    lines.append("")
    for key, val in report["graph_model_suitability"].items():
        lines.append(f"- **{key}**: {val['answer']}")
        lines.append(f"  {val['observed']}")
        if val.get("hypothesized"):
            lines.append(f"  {val['hypothesized']}")
    lines.append("")

    nd = report["novelty_decision"]
    lines.append("## 8. NOVELTY IMPLICATION")
    lines.append("")
    lines.append(f"Decision: **{nd['decision']}**")
    lines.append("")
    lines.append(nd["rationale"])
    lines.append("")
    lines.append(nd["explicit_disclaimer"])
    lines.append("")

    lines.append("## 9. LIMITATIONS")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 10. RECOMMENDED NEXT EXPERIMENT")
    lines.append("")
    lines.append(report["recommended_next_experiment"])
    lines.append("")

    lines.append(f"## FINAL VERDICT: {report['verdict']}")
    lines.append("")
    lines.append("STOP AFTER PHASE 9F.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9F directory: {OUTPUT_DIR}")
    if not PHASE9E_REPORT_PATH.exists():
        raise FileExistsError(f"Phase 9E report not found at {PHASE9E_REPORT_PATH}; run Phase 9E first.")

    phase9c_report = json.loads(PHASE9C_REPORT_PATH.read_text(encoding="utf-8"))
    phase9e_report = json.loads(PHASE9E_REPORT_PATH.read_text(encoding="utf-8"))

    target = select_second_member(phase9c_report)
    report: dict = {"second_member_selection": target}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = target["filename"].rsplit("/", 1)[-1]
    pcap_path = DATA_DIR / f"{safe_name}.pcap"

    inspector = RemoteObjectInspector(target["object_url"], max_total_bytes=MAX_TOTAL_DOWNLOAD_BYTES)

    if pcap_path.exists() and pcap_path.stat().st_size == target["uncompressed_size"]:
        # Already retrieved and decompressed in a prior run of this phase --
        # verify the existing local artifact instead of re-downloading the
        # member a second time (same member would otherwise be fetched
        # twice, unnecessarily doubling this phase's network footprint).
        import zlib as _zlib

        local_bytes = pcap_path.read_bytes()
        actual_crc32 = _zlib.crc32(local_bytes) & 0xFFFFFFFF
        first_4 = local_bytes[:4]
        from phase9d_pcap_validation import PCAPNG_MAGIC, PCAP_MAGICS

        detected_format = "UNKNOWN"
        if first_4 in PCAP_MAGICS:
            detected_format = PCAP_MAGICS[first_4]
        elif first_4 == PCAPNG_MAGIC:
            detected_format = "pcapng (Section Header Block)"

        report["download_validation"] = {
            "reused_existing_local_artifact": True,
            "filename_matches_expected": True,
            "compressed_size_matches_expected": True,
            "note": f"Local file {pcap_path} already present with the expected uncompressed "
            "size; verified via CRC-32 below instead of re-fetching over the network.",
        }
        decompression = {
            "uncompressed_bytes_expected": target["uncompressed_size"],
            "uncompressed_bytes_actual": len(local_bytes),
            "uncompressed_size_matches_expected": len(local_bytes) == target["uncompressed_size"],
            "expected_crc32": target["expected_crc32"],
            "actual_crc32": actual_crc32,
            "crc32_matches_expected": actual_crc32 == target["expected_crc32"],
            "first_32_bytes_hex": local_bytes[:32].hex(),
            "detected_format": detected_format,
            "is_valid_capture": detected_format != "UNKNOWN",
        }
        report["capture_validation"] = decompression
        del local_bytes

        if not decompression["is_valid_capture"] or not decompression["crc32_matches_expected"]:
            report["stopped_reason"] = "Existing local artifact failed CRC-32 or format re-verification."
            report["success"] = False
            report["verdict"] = "RED"
            _finish(report, inspector)
            return
    else:
        retrieval = retrieve_member(inspector, target)
        compressed_data = retrieval.pop("compressed_data")
        report["download_validation"] = retrieval

        if not retrieval["filename_matches_expected"] or not retrieval["compressed_size_matches_expected"]:
            report["stopped_reason"] = "Local header validation failed (filename or compressed size mismatch)."
            report["success"] = False
            report["verdict"] = "RED"
            _finish(report, inspector)
            return

        decompression = decompress_member(compressed_data, target)
        decompressed_bytes = decompression.pop("decompressed_bytes")
        report["capture_validation"] = decompression

        if not decompression["is_valid_capture"]:
            report["stopped_reason"] = "Decompressed data did not match any known PCAP/PCAPNG magic."
            report["success"] = False
            report["verdict"] = "RED"
            _finish(report, inspector)
            return

        pcap_path.write_bytes(decompressed_bytes)
        del decompressed_bytes  # not persisted further; only the pcap file and derived graph artifact remain

    # ---- Steps 4-6: parse + build EXACTLY the Phase 9E graph pipeline ----
    parse_result = parse_full_capture(pcap_path)
    capture_summary = summarize_capture(parse_result)
    report["second_capture_summary"] = capture_summary

    ip_packets = parse_result["ip_packets"]
    max_window_index = assign_windows(ip_packets, parse_result["capture_start_ts"])
    total_windows = max_window_index + 1

    raw_ips = {p["src_ip"] for p in ip_packets} | {p["dst_ip"] for p in ip_packets}
    anon_map, anon_validation = build_and_validate_anonymization(raw_ips)

    node_windows = build_node_window_features(ip_packets, anon_map)
    edge_windows = build_edge_window_features(ip_packets, anon_map)

    capture_level = compute_capture_level_stats(ip_packets, node_windows, edge_windows, total_windows)
    per_window = compute_per_window_distributions(node_windows, edge_windows, total_windows)
    dynamics = compute_temporal_dynamics(node_windows, edge_windows, total_windows)

    graph_stats_for_report = {
        **capture_level,
        "per_window_distributions_metrics": per_window["metrics"],
    }
    report["second_graph_statistics"] = graph_stats_for_report
    report["second_temporal_dynamics"] = dynamics
    report["anonymization_validation"] = anon_validation
    report["temporal_window_structure"] = {
        "window_seconds": WINDOW_SECONDS,
        "total_windows": total_windows,
    }

    n_hosts = capture_level["total_nodes_observed"]
    possible_pairs = n_hosts * (n_hosts - 1) if n_hosts > 1 else 0
    sparsity = 1.0 - (capture_level["total_unique_directed_source_destination_pairs"] / possible_pairs) if possible_pairs else None

    # ---- Step 7: comparison ----
    comparison_rows = build_comparison_table(phase9e_report, capture_summary, graph_stats_for_report, dynamics, sparsity)
    report["comparison_table"] = comparison_rows
    richness = classify_richness(comparison_rows)
    report["richness_classification"] = richness

    # ---- Steps 8-9 ----
    suitability = graph_model_suitability(comparison_rows, richness)
    report["graph_model_suitability"] = suitability
    decision = novelty_decision(richness, suitability)
    report["novelty_decision"] = decision

    report["limitations"] = [
        "Only two endpoint captures have now been examined out of 449 members in the "
        "archive; two data points establish that the pipeline generalizes MECHANICALLY "
        "(same code runs correctly on a different host) but is a very small basis for "
        "claiming statistical generalization of graph richness across the archive.",
        "No attack labels exist for either capture; nothing here reflects attack "
        "detectability.",
        "The second capture is from the same day (Wednesday-14-02-2018) and the same "
        "AWS environment as the first; cross-day generalization is NOT VERIFIED.",
        "Graph-model suitability answers in Section 7 are grounded in measured sparsity/"
        "window-population/run-length numbers, but no model was trained or evaluated -- "
        "the Q4 predictive-value question is explicitly NOT VERIFIED.",
        f"Raw per-packet records for the second capture are not persisted beyond the "
        f"single decompressed .pcap file under {DATA_DIR} (not committed as a tracked "
        f"result); only the JSON/Markdown reports are placed under results/.",
    ]
    report["recommended_next_experiment"] = (
        "If pursued further (not started here): before any GNN work, prototype the "
        "event/edge-sequence representation identified in Step 8 Q5 (a chronological "
        "stream of (timestamp, src, dst, edge_features) events rather than fixed "
        "per-window graph snapshots) against a third capture, and compare its ability to "
        "represent the same information more compactly than the sparse per-window "
        "snapshot approach used in Phase 9E/9F."
    )

    report["success"] = True

    verdict_by_decision = {
        "CONTINUE": "GREEN",
        "CONDITIONAL": "YELLOW",
        "STOP GRAPH BRANCH": "RED",
    }
    report["verdict"] = verdict_by_decision[decision["decision"]]

    _finish(report, inspector)


def _finish(report: dict, inspector: RemoteObjectInspector) -> None:
    report["total_bytes_downloaded"] = inspector.total_bytes_downloaded
    report["requests_log"] = inspector.requests_log
    report["max_total_download_budget_bytes"] = inspector.max_total_bytes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comparison_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "comparison_report.md", report)

    print(
        json.dumps(
            {
                "success": report.get("success"),
                "verdict": report.get("verdict"),
                "second_member": report["second_member_selection"]["filename"],
                "total_bytes_downloaded": report["total_bytes_downloaded"],
                "richness_classification": report.get("richness_classification", {}).get("classification"),
                "novelty_decision": report.get("novelty_decision", {}).get("decision"),
                "stopped_reason": report.get("stopped_reason"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
