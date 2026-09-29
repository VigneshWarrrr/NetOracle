"""Phase 9D: retrieve, decompress, and validate ONE tiny ZIP member from the
pcap.zip archive identified by the frozen Phase 9C discovery.

Target member metadata is read directly from Phase 9C's own persisted,
frozen output (results/phase9c_pcap_archive_inspection/discovery_report.json)
rather than hardcoded or re-derived -- this guarantees the exact same
values Phase 9C already verified (filename, sizes, offset, CRC-32) are used
here, with zero risk of a copy/paste drift.

Only two HTTP Range requests are ever issued against the S3 object: one for
the local file header (a generous small probe), one for the exact
compressed byte range of this single member. No other ZIP member is ever
touched; the 37.17GB archive is never bulk-downloaded.

Everything after decompression (format detection, packet parsing, aggregate
statistics) operates entirely on local bytes -- no further network access.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase9c_pcap_archive_inspection import RemoteObjectInspector  # noqa: E402 -- read-only reuse, not modified

PHASE9C_REPORT_PATH = Path(__file__).resolve().parent / "results/phase9c_pcap_archive_inspection/discovery_report.json"
TARGET_MEMBER_FILENAME = "pcap/UCAP172.31.69.22"

LOCAL_HEADER_SIGNATURE = 0x04034B50
LOCAL_HEADER_FIXED_SIZE = 30
LOCAL_HEADER_PROBE_BYTES = 30 + 64 + 256  # fixed part + filename headroom + extra-field headroom

MAX_TOTAL_DOWNLOAD_BYTES = 2 * 1024 * 1024  # this member is ~538 KiB; 2 MB is a generous, still-tiny hard cap
MAX_SAMPLE_PACKETS = 100

PCAP_MAGICS = {
    b"\xa1\xb2\xc3\xd4": "pcap (microsecond resolution, big-endian byte order)",
    b"\xd4\xc3\xb2\xa1": "pcap (microsecond resolution, little-endian byte order)",
    b"\xa1\xb2\x3c\x4d": "pcap (nanosecond resolution, big-endian byte order)",
    b"\x4d\x3c\xb2\xa1": "pcap (nanosecond resolution, little-endian byte order)",
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

DATA_DIR = Path(__file__).resolve().parent / "data/phase9d"
OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase9d_pcap_validation"


def load_target_member_metadata() -> dict:
    """Reads Phase 9C's own frozen discovery report (never modified here)
    and returns the exact, already-verified metadata for our one target
    member."""
    report = json.loads(PHASE9C_REPORT_PATH.read_text(encoding="utf-8"))
    for member in report["members"]:
        if member["filename"] == TARGET_MEMBER_FILENAME:
            return {
                "object_url": report["object_url"],
                "filename": member["filename"],
                "compressed_size": member["compressed_size"],
                "uncompressed_size": member["uncompressed_size"],
                "compression_method": member["compression_method"],
                "local_header_offset": member["local_header_offset"],
                "expected_crc32": member["crc32"],
            }
    raise RuntimeError(f"{TARGET_MEMBER_FILENAME} not found in Phase 9C's frozen discovery report")


def retrieve_member(inspector: RemoteObjectInspector, target: dict) -> dict:
    """STEP 1: fetch the local file header, verify it, compute the exact
    compressed-data start offset, then fetch exactly the compressed bytes."""
    probe_start = target["local_header_offset"]
    probe_end = probe_start + LOCAL_HEADER_PROBE_BYTES - 1
    probe = inspector.get_range(probe_start, probe_end, purpose="local_header_probe")

    signature = struct.unpack_from("<I", probe, 0)[0]
    if signature != LOCAL_HEADER_SIGNATURE:
        raise RuntimeError(f"Local file header signature mismatch: expected 0x{LOCAL_HEADER_SIGNATURE:08x}, got 0x{signature:08x}")

    (_sig, version_needed, flags, method, _mtime, _mdate, header_crc32, header_comp_size, header_uncomp_size, fname_len, extra_len) = struct.unpack_from(
        "<IHHHHHIIIHH", probe, 0
    )

    header_total_size = LOCAL_HEADER_FIXED_SIZE + fname_len + extra_len
    if header_total_size > len(probe):
        # Defensive path: only triggers if the extra field is larger than our
        # generous 256-byte headroom (not expected here, but handled safely).
        probe = inspector.get_range(probe_start, probe_start + header_total_size - 1, purpose="local_header_probe_extended")

    filename_bytes = probe[LOCAL_HEADER_FIXED_SIZE : LOCAL_HEADER_FIXED_SIZE + fname_len]
    filename = filename_bytes.decode("utf-8", errors="replace")

    data_start = probe_start + header_total_size
    data_end = data_start + target["compressed_size"] - 1
    compressed_data = inspector.get_range(data_start, data_end, purpose="member_compressed_data")

    uses_data_descriptor = bool(flags & 0x0008)  # bit 3: sizes/crc in local header may be 0, real values trail the data

    return {
        "requested_byte_ranges": [
            {"purpose": "local_header_probe", "start": probe_start, "end": probe_end, "bytes": len(probe) if header_total_size <= LOCAL_HEADER_PROBE_BYTES else header_total_size},
            {"purpose": "member_compressed_data", "start": data_start, "end": data_end, "bytes": len(compressed_data)},
        ],
        "local_header_signature_valid": True,
        "local_header_filename": filename,
        "filename_matches_expected": filename == target["filename"],
        "local_header_compression_method": method,
        "local_header_uses_data_descriptor": uses_data_descriptor,
        "local_header_crc32_field": header_crc32,
        "header_total_size_bytes": header_total_size,
        "data_start_offset": data_start,
        "compressed_bytes_expected": target["compressed_size"],
        "compressed_bytes_received": len(compressed_data),
        "compressed_size_matches_expected": len(compressed_data) == target["compressed_size"],
        "compressed_data": compressed_data,
    }


def decompress_member(compressed_data: bytes, target: dict) -> dict:
    """STEP 2: raw DEFLATE decompression (ZIP's deflate stream has no
    zlib/gzip header, hence wbits=-15), CRC-32 cross-check against the
    value Phase 9C already extracted from the central directory, and
    capture-format magic detection."""
    if target["compression_method"] != 8:
        raise RuntimeError(f"Unsupported compression method {target['compression_method']} (expected 8 = DEFLATE)")

    decompressed = zlib.decompress(compressed_data, -15)
    actual_crc32 = zlib.crc32(decompressed) & 0xFFFFFFFF

    first_32 = decompressed[:32]
    first_4 = decompressed[:4]
    detected_format = "UNKNOWN"
    if first_4 in PCAP_MAGICS:
        detected_format = PCAP_MAGICS[first_4]
    elif first_4 == PCAPNG_MAGIC:
        detected_format = "pcapng (Section Header Block)"

    return {
        "uncompressed_bytes_expected": target["uncompressed_size"],
        "uncompressed_bytes_actual": len(decompressed),
        "uncompressed_size_matches_expected": len(decompressed) == target["uncompressed_size"],
        "expected_crc32": target["expected_crc32"],
        "actual_crc32": actual_crc32,
        "crc32_matches_expected": actual_crc32 == target["expected_crc32"],
        "first_32_bytes_hex": first_32.hex(),
        "detected_format": detected_format,
        "is_valid_capture": detected_format != "UNKNOWN",
        "decompressed_bytes": decompressed,
    }


def _protocol_name(packet) -> str:
    from scapy.layers.inet import ICMP, TCP, UDP

    if TCP in packet:
        return "TCP"
    if UDP in packet:
        return "UDP"
    if ICMP in packet:
        return "ICMP"
    return "OTHER"


def parse_packet_sample(pcap_path: Path, max_packets: int = MAX_SAMPLE_PACKETS) -> dict:
    """STEP 3: bounded parse of the first `max_packets` packets only."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    samples = []
    failures = []
    count = 0
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if count >= max_packets:
                break
            count += 1
            try:
                if IP not in packet:
                    failures.append({"packet_index": count, "reason": "no IPv4 layer present"})
                    continue
                ip_layer = packet[IP]
                entry = {
                    "packet_index": count,
                    "timestamp": float(packet.time),
                    "packet_length": len(packet),
                    "src_ip": ip_layer.src,
                    "dst_ip": ip_layer.dst,
                    "protocol": _protocol_name(packet),
                    "ttl": int(ip_layer.ttl),
                    "ip_flags": str(ip_layer.flags),
                    "ip_fragment_offset": int(ip_layer.frag),
                    "src_port": None,
                    "dst_port": None,
                    "tcp_flags": None,
                    "tcp_window": None,
                    "payload_length": None,
                }
                if TCP in packet:
                    tcp_layer = packet[TCP]
                    entry["src_port"] = int(tcp_layer.sport)
                    entry["dst_port"] = int(tcp_layer.dport)
                    entry["tcp_flags"] = str(tcp_layer.flags)
                    entry["tcp_window"] = int(tcp_layer.window)
                    entry["payload_length"] = len(bytes(tcp_layer.payload))
                elif UDP in packet:
                    udp_layer = packet[UDP]
                    entry["src_port"] = int(udp_layer.sport)
                    entry["dst_port"] = int(udp_layer.dport)
                    entry["payload_length"] = len(bytes(udp_layer.payload))
                samples.append(entry)
            except Exception as exc:  # noqa: BLE001 -- record and continue, never crash the bounded sample
                failures.append({"packet_index": count, "reason": str(exc)})

    return {"sample_size": len(samples), "packets": samples, "parsing_failures": failures, "bounded_at": max_packets}


def compute_aggregate_statistics(pcap_path: Path) -> dict:
    """STEP 4: full pass over the ENTIRE decompressed member (safe here --
    the file is only ~1.25 MB uncompressed). Aggregate capture metadata
    only, not model features."""
    from scapy.layers.inet import IP
    from scapy.utils import PcapReader

    total_packets = 0
    non_ip_packets = 0
    first_timestamp = None
    last_timestamp = None
    src_ips: set[str] = set()
    dst_ips: set[str] = set()
    protocols: set[str] = set()
    src_ports: set[int] = set()
    dst_ports: set[int] = set()
    total_bytes = 0
    protocol_counts = {"TCP": 0, "UDP": 0, "ICMP": 0, "OTHER": 0}

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            total_packets += 1
            timestamp = float(packet.time)
            first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
            total_bytes += len(packet)

            if IP not in packet:
                non_ip_packets += 1
                continue
            ip_layer = packet[IP]
            src_ips.add(ip_layer.src)
            dst_ips.add(ip_layer.dst)
            protocol = _protocol_name(packet)
            protocols.add(protocol)
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1

            from scapy.layers.inet import TCP, UDP

            if TCP in packet:
                src_ports.add(int(packet[TCP].sport))
                dst_ports.add(int(packet[TCP].dport))
            elif UDP in packet:
                src_ports.add(int(packet[UDP].sport))
                dst_ports.add(int(packet[UDP].dport))

    duration = (last_timestamp - first_timestamp) if (first_timestamp is not None and last_timestamp is not None) else None

    return {
        "total_packet_count": total_packets,
        "non_ip_packet_count": non_ip_packets,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "capture_duration_seconds": duration,
        "unique_source_ips": sorted(src_ips),
        "unique_destination_ips": sorted(dst_ips),
        "unique_hosts_overall": sorted(src_ips | dst_ips),
        "unique_protocols": sorted(protocols),
        "unique_source_ports_count": len(src_ports),
        "unique_destination_ports_count": len(dst_ports),
        "total_bytes": total_bytes,
        "protocol_counts": protocol_counts,
    }


def graph_feasibility_observation(aggregates: dict) -> dict:
    """STEP 5: observations from THIS ONE capture only -- not a dataset-wide claim."""
    hosts = aggregates["unique_hosts_overall"]
    src_ips = aggregates["unique_source_ips"]
    dst_ips = aggregates["unique_destination_ips"]

    # distinct directed (src, dst) pairs would require re-scanning per-packet
    # pairs; approximate a safe upper/lower bound from src/dst set sizes
    # instead of re-parsing (kept here as a documented estimate, not re-read).
    directed_pairs_upper_bound_estimate = len(src_ips) * len(dst_ips)

    duration = aggregates["capture_duration_seconds"]
    ten_second_aggregation_appears_feasible = duration is not None and duration >= 10.0

    return {
        "distinct_hosts_observed": len(hosts),
        "directed_pairs_upper_bound_estimate": directed_pairs_upper_bound_estimate,
        "directed_pairs_estimate_note": "Upper bound = |unique_src_ips| x |unique_dst_ips|; the true distinct-pair "
        "count was not separately tallied in this pass and would be <= this estimate.",
        "multiple_hosts_communicate": len(hosts) > 1,
        "structurally_supports_host_communication_graph": len(hosts) > 1,
        "ten_second_temporal_aggregation_appears_feasible": ten_second_aggregation_appears_feasible,
        "capture_duration_seconds": duration,
        "caveat": "This capture is a SINGLE endpoint's traffic (per Phase 9C's per-host archive naming convention). "
        "These observations describe only this one file and must not be generalized to the full 449-member "
        "archive or to dataset-wide graph feasibility.",
    }


def csv_comparison() -> dict:
    """STEP 6: broad metadata comparison ONLY -- no row-level matching attempted."""
    csv_path = Path(__file__).resolve().parents[2] / "data/raw/CSE-CIC-IDS2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
    csv_exists = csv_path.exists()
    header_columns: list[str] = []
    if csv_exists:
        with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
            header_columns = stream.readline().strip().split(",")

    return {
        "csv_exists_for_this_day": csv_exists,
        "csv_path": str(csv_path),
        "csv_header_columns": header_columns,
        "endpoint_identity_present_in_csv": "Src IP" in header_columns or "Dst IP" in header_columns,
        "protocol_column_present": "Protocol" in header_columns,
        "timestamp_column_present": "Timestamp" in header_columns,
        "row_level_correspondence_established": False,
        "note": "Per the already-audited Phase 2 dataset readiness finding, Wednesday-14-02-2018 is one of the "
        "9 CSV files WITHOUT Src IP/Dst IP columns. Broad protocol and timestamp columns are present, but NO "
        "flow-level, row-level, or timestamp-level correspondence between this PCAP member and any CSV row "
        "has been attempted or established here.",
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9D directory: {OUTPUT_DIR}")

    target = load_target_member_metadata()
    inspector = RemoteObjectInspector(target["object_url"], max_total_bytes=MAX_TOTAL_DOWNLOAD_BYTES)

    report: dict = {"target_member": {key: value for key, value in target.items()}}

    retrieval = retrieve_member(inspector, target)
    compressed_data = retrieval.pop("compressed_data")
    report["step1_download_validation"] = retrieval

    if not retrieval["filename_matches_expected"] or not retrieval["compressed_size_matches_expected"]:
        report["stopped_reason"] = "Local header validation failed (filename or compressed size mismatch)."
        report["success"] = False
        _finish(report, inspector)
        return

    decompression = decompress_member(compressed_data, target)
    decompressed_bytes = decompression.pop("decompressed_bytes")
    report["step2_zip_member_validation"] = decompression

    if not decompression["is_valid_capture"]:
        report["stopped_reason"] = f"Decompressed data does not match any known PCAP/PCAPNG magic (first 4 bytes: {decompression['first_32_bytes_hex'][:8]})."
        report["success"] = False
        _finish(report, inspector)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pcap_path = DATA_DIR / "UCAP172.31.69.22.pcap"
    pcap_path.write_bytes(decompressed_bytes)
    report["saved_pcap_path"] = str(pcap_path)
    report["saved_pcap_bytes"] = len(decompressed_bytes)

    report["step3_packet_sample"] = parse_packet_sample(pcap_path)
    aggregates = compute_aggregate_statistics(pcap_path)
    report["step4_capture_aggregates"] = aggregates
    report["step5_graph_feasibility"] = graph_feasibility_observation(aggregates)
    report["step6_csv_comparison"] = csv_comparison()
    report["success"] = True

    _finish(report, inspector)


def _finish(report: dict, inspector: RemoteObjectInspector) -> None:
    report["total_bytes_downloaded"] = inspector.total_bytes_downloaded
    report["requests_log"] = inspector.requests_log
    report["max_total_download_budget_bytes"] = inspector.max_total_bytes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "validation_report.md", report)

    print(
        json.dumps(
            {
                "success": report.get("success"),
                "total_bytes_downloaded": report["total_bytes_downloaded"],
                "detected_format": report.get("step2_zip_member_validation", {}).get("detected_format"),
                "sample_packets": report.get("step3_packet_sample", {}).get("sample_size"),
                "total_packets": report.get("step4_capture_aggregates", {}).get("total_packet_count"),
                "stopped_reason": report.get("stopped_reason"),
            },
            indent=2,
        )
    )


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Phase 9D: Single PCAP Member Validation",
        "",
        f"Target member: `{report['target_member']['filename']}`",
        f"Success: **{report.get('success')}**",
        "",
    ]
    if not report.get("success"):
        lines += ["## Stopped", "", report.get("stopped_reason", "(no reason recorded)"), ""]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    d = report["step1_download_validation"]
    lines += [
        "## 1. DOWNLOAD VALIDATION",
        "",
        "| Purpose | Start | End | Bytes |",
        "|---|---:|---:|---:|",
    ]
    for r in d["requested_byte_ranges"]:
        lines.append(f"| {r['purpose']} | {r['start']:,} | {r['end']:,} | {r['bytes']:,} |")
    lines += [
        "",
        f"- Total bytes downloaded: **{report['total_bytes_downloaded']:,}**",
        f"- Compressed size expected/received: {d['compressed_bytes_expected']:,} / {d['compressed_bytes_received']:,} -- match: **{d['compressed_size_matches_expected']}**",
        f"- Local header CRC-32 field: {d['local_header_crc32_field']} (data-descriptor flag set: {d['local_header_uses_data_descriptor']})",
        "",
        "## 2. ZIP MEMBER VALIDATION",
        "",
        f"- Local header signature valid: **{d['local_header_signature_valid']}**",
        f"- Local header filename: `{d['local_header_filename']}` -- matches expected `{report['target_member']['filename']}`: **{d['filename_matches_expected']}**",
    ]

    z = report["step2_zip_member_validation"]
    lines += [
        f"- Uncompressed size expected/actual: {z['uncompressed_bytes_expected']:,} / {z['uncompressed_bytes_actual']:,} -- match: **{z['uncompressed_size_matches_expected']}**",
        f"- CRC-32 expected/actual: {z['expected_crc32']} / {z['actual_crc32']} -- match: **{z['crc32_matches_expected']}**",
        f"- First 32 bytes (hex): `{z['first_32_bytes_hex']}`",
        "",
        "## 3. CAPTURE FORMAT",
        "",
        f"Detected format: **{z['detected_format']}**",
        "",
    ]

    sample = report["step3_packet_sample"]
    lines += [
        "## 4. PACKET SAMPLE",
        "",
        f"Bounded sample size: {sample['sample_size']} (cap {sample['bounded_at']}); parsing failures: {len(sample['parsing_failures'])}",
        "",
        "First 5 sampled packets:",
        "",
        "| # | timestamp | len | src_ip | dst_ip | proto | src_port | dst_port | ttl | tcp_flags | window | payload_len | ip_flags | frag |",
        "|---|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for p in sample["packets"][:5]:
        lines.append(
            f"| {p['packet_index']} | {p['timestamp']:.6f} | {p['packet_length']} | {p['src_ip']} | {p['dst_ip']} | "
            f"{p['protocol']} | {p['src_port']} | {p['dst_port']} | {p['ttl']} | {p['tcp_flags']} | {p['tcp_window']} | "
            f"{p['payload_length']} | {p['ip_flags']} | {p['ip_fragment_offset']} |"
        )

    agg = report["step4_capture_aggregates"]
    lines += [
        "",
        "## 5. COMPLETE CAPTURE AGGREGATES",
        "",
        f"- Total packets: {agg['total_packet_count']:,} (non-IP: {agg['non_ip_packet_count']})",
        f"- First timestamp: {agg['first_timestamp']}",
        f"- Last timestamp: {agg['last_timestamp']}",
        f"- Duration: {agg['capture_duration_seconds']} seconds",
        f"- Unique source IPs: {len(agg['unique_source_ips'])}",
        f"- Unique destination IPs: {len(agg['unique_destination_ips'])}",
        f"- Unique hosts overall: {len(agg['unique_hosts_overall'])} -- {agg['unique_hosts_overall']}",
        f"- Unique protocols: {agg['unique_protocols']}",
        f"- Unique source ports: {agg['unique_source_ports_count']}",
        f"- Unique destination ports: {agg['unique_destination_ports_count']}",
        f"- Total bytes: {agg['total_bytes']:,}",
        f"- Protocol counts: {agg['protocol_counts']}",
        "",
    ]

    graph = report["step5_graph_feasibility"]
    lines += [
        "## 6. GRAPH FEASIBILITY (this one capture only)",
        "",
        f"- Distinct hosts observed: {graph['distinct_hosts_observed']}",
        f"- Directed src->dst pairs (upper-bound estimate): {graph['directed_pairs_upper_bound_estimate']}",
        f"- Multiple hosts communicate: {graph['multiple_hosts_communicate']}",
        f"- Structurally supports a host-communication graph: {graph['structurally_supports_host_communication_graph']}",
        f"- 10-second temporal aggregation appears feasible: {graph['ten_second_temporal_aggregation_appears_feasible']}",
        "",
        f"{graph['caveat']}",
        "",
    ]

    csvc = report["step6_csv_comparison"]
    lines += [
        "## 7. CSV COMPARISON (metadata only)",
        "",
        f"- CSV exists for this day: {csvc['csv_exists_for_this_day']} (`{csvc['csv_path']}`)",
        f"- Endpoint identity present in CSV: {csvc['endpoint_identity_present_in_csv']}",
        f"- Protocol column present: {csvc['protocol_column_present']}",
        f"- Timestamp column present: {csvc['timestamp_column_present']}",
        f"- Row-level correspondence established: {csvc['row_level_correspondence_established']}",
        "",
        csvc["note"],
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
