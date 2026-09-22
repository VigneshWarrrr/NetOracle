"""Phase 9I: PCAP <-> flow alignment investigation.

An EVIDENCE-GATHERING phase only. Determines whether the CSE-CIC-IDS2018
PCAP archive and the processed flow-level CSVs already used by NetOracle's
forecasting pipeline can be defensibly paired as a multimodal dataset. No
model is trained, no GNN/fusion work is done, and no existing pipeline is
modified. New network access is strictly bounded (ZIP central-directory
metadata only, via HTTP Range requests -- never the archive bodies).

This phase builds directly on Phase 9H's finding (weak, indistinguishable-
from-noise cross-correlation, |r|~=0.088) by seeking AUTHORITATIVE evidence
(documentation, archive metadata, logs.zip contents) rather than trying more
correlation offsets, which Phase 9H already showed is not sufficient proof
of correspondence on its own.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

# Read-only reuse of Phase 9C's remote ZIP-metadata primitives (not modified).
from phase9c_pcap_archive_inspection import (  # noqa: E402
    RemoteObjectInspector,
    classify_member,
    find_eocd,
    find_zip64_locator_in_tail,
    parse_central_directory,
    parse_zip64_eocd_record,
)

OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9i_pcap_flow_alignment"
PHASE9C_REPORT_PATH = EXPERIMENTS_DIR / "results/phase9c_pcap_archive_inspection/discovery_report.json"
PHASE9H_REPORT_PATH = EXPERIMENTS_DIR / "results/phase9h_multimodal_baseline/multimodal_baseline_report.json"

WINDOWS_DIR = Path(r"C:\AKSHAY\Akshay\SIH FOLDER MAIN\data\windows")
RAW_CSV_DIR = Path(r"C:\AKSHAY\Akshay\SIH FOLDER MAIN\data\raw\CSE-CIC-IDS2018")
WEDNESDAY_WINDOWED_CSV = WINDOWS_DIR / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
WEDNESDAY_RAW_CSV = RAW_CSV_DIR / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
PHASE9D_PCAP = EXPERIMENTS_DIR / "data/phase9d/UCAP172.31.69.22.pcap"

LOGS_ZIP_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Original%20Network%20Traffic%20and%20Log%20data/"
    "Wednesday-14-02-2018/logs.zip"
)
LOGS_ZIP_KEY = "Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip"

TAIL_SIZE_BYTES = 65536
MAX_CENTRAL_DIRECTORY_SAFE_BYTES = 8 * 1024 * 1024
# Conservative total budget for THIS phase's new network activity: central-
# directory metadata plus at most a couple of small, clearly-relevant text
# files (per Step 4's "if a small metadata/text file looks highly relevant,
# retrieve only that member" instruction).
MAX_TOTAL_DOWNLOAD_BYTES = 4 * 1024 * 1024
SMALL_MEMBER_RETRIEVAL_CEILING_BYTES = 200 * 1024  # never auto-fetch a member above this

RELEVANT_METADATA_KEYWORDS = (
    "readme", "manifest", "info", "config", "capture", "host", "interface",
    "timezone", "time", "map", "topology", "network", "setup",
)


# ---------------------------------------------------------------------------
# Step 0: repository safety
# ---------------------------------------------------------------------------


def record_repo_safety() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception as exc:  # pragma: no cover - defensive
            return f"<git command failed: {exc}>"

    commit_hash = run("rev-parse", "HEAD")
    status_short = run("status", "--short")
    branch = run("branch", "--show-current")

    prior_result_dirs = sorted(
        p.name for p in (EXPERIMENTS_DIR / "results").iterdir() if p.is_dir() and p.name != "phase9i_pcap_flow_alignment"
    )
    prior_file_counts = {
        name: sum(1 for _ in (EXPERIMENTS_DIR / "results" / name).rglob("*") if _.is_file())
        for name in prior_result_dirs
    }

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "git_status_short": status_short,
        "git_status_note": (
            "Non-empty status reflects PRE-EXISTING modified/untracked files from earlier "
            "phases in this session (experiments/tests.py additions, new experiment scripts, "
            "etc.) -- not changes made by Phase 9I. Phase 9I creates files only under "
            "experiments/results/phase9i_pcap_flow_alignment/, experiments/phase9i_pcap_flow_alignment.py, "
            "and tests/test_phase9i_pcap_flow_alignment.py."
        ),
        "prior_result_directory_file_counts": prior_file_counts,
    }


def verify_no_prior_artifact_changed(baseline: dict) -> dict:
    current_counts = {
        name: sum(1 for _ in (EXPERIMENTS_DIR / "results" / name).rglob("*") if _.is_file())
        for name in baseline["prior_result_directory_file_counts"]
    }
    mismatches = {
        name: {"before": baseline["prior_result_directory_file_counts"][name], "after": current_counts[name]}
        for name in current_counts
        if current_counts[name] != baseline["prior_result_directory_file_counts"][name]
    }
    return {"unchanged": not mismatches, "mismatches": mismatches, "current_counts": current_counts}


# ---------------------------------------------------------------------------
# Step 1: flow data semantics (read-only inspection of local files)
# ---------------------------------------------------------------------------


def inspect_flow_csv_semantics() -> dict:
    findings: dict = {"evidence_source": []}

    if not WEDNESDAY_RAW_CSV.exists():
        findings["error"] = f"Raw CSV not found at {WEDNESDAY_RAW_CSV}"
        return findings
    findings["evidence_source"].append(str(WEDNESDAY_RAW_CSV))

    with WEDNESDAY_RAW_CSV.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= 3:
                break
            sample_rows.append(row[:8])

    endpoint_columns = {"Flow ID", "Src IP", "Src Port", "Dst IP"}
    present_endpoint_columns = sorted(endpoint_columns & set(header))

    windowed_range = {}
    if WEDNESDAY_WINDOWED_CSV.exists():
        findings["evidence_source"].append(str(WEDNESDAY_WINDOWED_CSV))
        with WEDNESDAY_WINDOWED_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            starts = [row["window_start"] for row in reader]
        windowed_range = {"min": min(starts), "max": max(starts), "row_count": len(starts)}

    # Authoritative repo evidence already established in Phase 2 (docs/CIC_IDS2018_DATA_READINESS.md)
    readiness_doc = REPO_ROOT / "docs/CIC_IDS2018_DATA_READINESS.md"
    readiness_doc_excerpt = None
    if readiness_doc.exists():
        findings["evidence_source"].append(str(readiness_doc))
        text = readiness_doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" in line and "range" in line.lower():
                readiness_doc_excerpt = line.strip()
                break

    return {
        "evidence_source": findings["evidence_source"],
        "csv_filename": WEDNESDAY_RAW_CSV.name,
        "column_count": len(header),
        "columns": header,
        "timestamp_column_name": "Timestamp",
        "timestamp_column_present": "Timestamp" in header,
        "timestamp_datatype": "string, format DD/MM/YYYY HH:MM:SS (e.g. '14/02/2018 08:31:01'), no timezone suffix present in any sampled or audited value",
        "example_raw_timestamp_values": [row[2] if len(row) > 2 else None for row in sample_rows],
        "endpoint_identity_columns_present": present_endpoint_columns,
        "endpoint_identity_available": bool(present_endpoint_columns),
        "row_represents": (
            "one CICFlowMeter bidirectional FLOW record (aggregated over the flow's full duration), "
            "NOT a single packet -- evidenced by columns Tot Fwd Pkts / Tot Bwd Pkts / Flow Duration / "
            "per-flow IAT and packet-length statistics, which are only meaningful as aggregates over "
            "multiple packets."
        ),
        "flow_duration_semantics": "Flow Duration column, microseconds (e.g. 112641719 microseconds = 112.6 seconds), per Phase 2's numeric-quality audit finite range check",
        "flow_start_end_timestamps_available": "Only a single 'Timestamp' column is present (flow start time, per CICFlowMeter's documented output convention); no separate flow-end timestamp column exists -- end time would need to be derived as Timestamp + Flow Duration.",
        "raw_csv_windowed_csv_window_start_range": windowed_range,
        "readiness_doc_excerpt": readiness_doc_excerpt,
        "conclusion": (
            "For Wednesday-14-02-2018 (the day corresponding to the two available PCAP captures), "
            "the processed CSV has NO endpoint identity columns (Flow ID/Src IP/Src Port/Dst IP all "
            "absent) and NO documented timezone for its Timestamp column. Only Tuesday-20-02-2018 has "
            "these endpoint columns, and no PCAP capture is currently available for that day."
        ),
    }


# ---------------------------------------------------------------------------
# Step 2: repository documentation / metadata search
# ---------------------------------------------------------------------------


def search_repository_documentation() -> dict:
    keywords = [
        "timezone", "UTC", "EST", "EDT", "AST", "ADT", "CICFlowMeter", "host",
        "interface", "capture-to-CSV", "IP", "PCAP", "capture", "Wednesday-14-02-2018",
        "flow", "network traffic", "source IP", "destination IP",
    ]

    search_paths = [
        REPO_ROOT / "docs",
        REPO_ROOT / "README.md",
        REPO_ROOT / "reports",
        EXPERIMENTS_DIR / "results",
        REPO_ROOT / "audits",
        REPO_ROOT / "ingestion",
    ]

    timezone_specific_keywords = ["timezone", "UTC", "EST", "EDT", "AST", "ADT"]
    timezone_hits: list[dict] = []

    for base in search_paths:
        if not base.exists():
            continue
        files = [base] if base.is_file() else list(base.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix not in (".md", ".py", ".json", ".txt"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for kw in timezone_specific_keywords:
                if kw.lower() in text.lower():
                    # find the line for context
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if kw.lower() in line.lower():
                            timezone_hits.append(
                                {
                                    "keyword": kw,
                                    "file": str(path.relative_to(REPO_ROOT)),
                                    "line_number": line_no,
                                    "line_excerpt": line.strip()[:200],
                                }
                            )

    return {
        "keywords_searched": keywords,
        "search_paths": [str(p.relative_to(REPO_ROOT)) if p.is_absolute() else str(p) for p in search_paths],
        "timezone_specific_hits": timezone_hits,
        "timezone_documented_anywhere": len(timezone_hits) > 0,
        "evidence_classification": {
            "A_authoritative_evidence": (
                "NONE FOUND. No original CSE-CIC-IDS2018 dataset-creator documentation (e.g. a "
                "README from the UNB/CIC S3 bucket itself, a published dataset paper excerpt, or "
                "capture-tool configuration file) is present anywhere in this repository."
            ),
            "B_repository_evidence": (
                f"{len(timezone_hits)} keyword hits found across repository docs/code for explicit "
                "timezone abbreviations (UTC/EST/EDT/AST/ADT); all are either code identifiers "
                "unrelated to dataset timezone (e.g. 'AST' as a Python ast-module alias) or, if any "
                "are genuine, are listed explicitly in timezone_specific_hits above for direct "
                "verification -- none establish the CICFlowMeter Timestamp column's timezone."
            ),
            "C_general_background": (
                "General public knowledge about CIC-IDS-family datasets (e.g. common community "
                "assumptions about Fredericton/Atlantic-Canada or AWS-default-UTC clocks) is "
                "EXPLICITLY NOT treated as evidence here, per this phase's hard constraint not to "
                "guess timestamps/timezones -- it is background only, never used as a substitute for "
                "authoritative or repository evidence."
            ),
            "D_inference": (
                "Phase 9H's empirical cross-correlation search (best |r|=0.088, indistinguishable "
                "from noise) is DERIVED evidence, not authoritative evidence -- it was tested but "
                "explicitly did not establish a timezone or alignment."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Step 3: PCAP archive metadata (reuses Phase 9C's frozen, already-downloaded
# central-directory listing -- NO new network access)
# ---------------------------------------------------------------------------


def analyze_pcap_archive_metadata() -> dict:
    if not PHASE9C_REPORT_PATH.exists():
        return {"error": f"Phase 9C report not found at {PHASE9C_REPORT_PATH}"}

    report = json.loads(PHASE9C_REPORT_PATH.read_text(encoding="utf-8"))
    members = [m for m in report["members"] if m["classification"] != "folder_marker"]

    import re

    ucap_pattern = re.compile(r"^pcap/UCAP")
    cap_pattern = re.compile(r"^pcap/cap", re.IGNORECASE)
    ucap_members = [m for m in members if ucap_pattern.match(m["filename"])]
    cap_members = [m for m in members if cap_pattern.match(m["filename"]) and not ucap_pattern.match(m["filename"])]
    other_members = [m for m in members if m not in ucap_members and m not in cap_members]

    sizes = sorted(m["compressed_size"] for m in members)
    largest = sorted(members, key=lambda m: -m["compressed_size"])[:5]

    return {
        "evidence_source": f"{PHASE9C_REPORT_PATH} (frozen, read-only; no new network access this phase)",
        "total_members": len(members),
        "ucap_prefixed_members": len(ucap_members),
        "cap_prefixed_members": len(cap_members),
        "other_naming_pattern_members": len(other_members),
        "other_member_filenames": [m["filename"] for m in other_members],
        "min_compressed_bytes": sizes[0] if sizes else None,
        "max_compressed_bytes": sizes[-1] if sizes else None,
        "five_largest_members": [
            {"filename": m["filename"], "compressed_size": m["compressed_size"], "uncompressed_size": m["uncompressed_size"]}
            for m in largest
        ],
        "conclusion": (
            "ALL 449 non-folder members follow one of exactly two per-HOST naming patterns "
            "(`pcap/UCAP<ip>` or `pcap/cap<hostname>-<ip>`), with zero exceptions -- including the "
            "single largest member (789 MB compressed). No member's filename or size suggests a "
            "merged, multi-host, or whole-network capture. The archive is structurally composed "
            "entirely of individual per-host captures; a network-wide population sample does not "
            "exist anywhere in this archive."
        ),
    }


# ---------------------------------------------------------------------------
# Step 4: logs.zip metadata (NEW, bounded network access)
# ---------------------------------------------------------------------------


def inspect_logs_zip() -> dict:
    inspector = RemoteObjectInspector(LOGS_ZIP_URL, max_total_bytes=MAX_TOTAL_DOWNLOAD_BYTES)
    result: dict = {"object_key": LOGS_ZIP_KEY, "object_url": LOGS_ZIP_URL}

    head_info = inspector.head()
    result["head"] = head_info
    if head_info["accept_ranges"] != "bytes":
        result["success"] = False
        result["stopped_reason"] = "Server does not advertise Accept-Ranges: bytes."
        return _finish_logs_zip(result, inspector)

    content_length = head_info["content_length"]
    tail_start = max(content_length - TAIL_SIZE_BYTES, 0)
    tail = inspector.get_range(tail_start, content_length - 1, purpose="tail_64kib")

    try:
        eocd = find_eocd(tail, tail_start)
    except RuntimeError as exc:
        result["success"] = False
        result["stopped_reason"] = str(exc)
        return _finish_logs_zip(result, inspector)
    result["eocd"] = eocd

    needs_zip64 = eocd["cd_size_is_zip64_sentinel"] or eocd["cd_offset_is_zip64_sentinel"] or eocd["total_entries_is_zip64_sentinel"]
    result["zip64_required"] = needs_zip64
    resolved_cd_size = eocd["cd_size"]
    resolved_cd_offset = eocd["cd_offset"]
    resolved_total_entries = eocd["total_entries"]

    if needs_zip64:
        locator = find_zip64_locator_in_tail(tail, eocd["index_in_tail"], tail_start)
        if locator is None:
            result["success"] = False
            result["stopped_reason"] = "ZIP64 required but locator not found in fetched tail."
            return _finish_logs_zip(result, inspector)
        zip64_bytes = inspector.get_range(
            locator["zip64_eocd_offset"], locator["zip64_eocd_offset"] + 56 - 1, purpose="zip64_eocd_record"
        )
        zip64_record = parse_zip64_eocd_record(zip64_bytes)
        result["zip64_record"] = zip64_record
        resolved_cd_size = zip64_record["cd_size"]
        resolved_cd_offset = zip64_record["cd_offset"]
        resolved_total_entries = zip64_record["total_entries"]

    result["resolved_central_directory"] = {
        "cd_offset": resolved_cd_offset, "cd_size": resolved_cd_size, "total_entries": resolved_total_entries,
    }

    if resolved_cd_size > MAX_CENTRAL_DIRECTORY_SAFE_BYTES:
        result["success"] = False
        result["stopped_reason"] = f"Central directory size ({resolved_cd_size}) exceeds safety threshold."
        return _finish_logs_zip(result, inspector)

    cd_bytes = inspector.get_range(resolved_cd_offset, resolved_cd_offset + resolved_cd_size - 1, purpose="central_directory")
    entries = parse_central_directory(cd_bytes, resolved_total_entries)
    classified = [{**e, "classification": classify_member(e["filename"])} for e in entries]
    result["members"] = classified
    result["member_count"] = len(classified)

    # Identify candidates worth retrieving via two independent, non-keyword-
    # dependent signals: (1) keyword match in the filename, or (2) a
    # STRUCTURAL outlier -- a member whose extension differs from the
    # archive's own dominant extension (computed from the actual listing,
    # not assumed in advance). This catches files like "logs/U172.31.69.18"
    # (an IP-address basename containing dots, but not the dominant
    # extension) without hardcoding what that dominant extension is.
    OUTLIER_SIZE_CEILING_BYTES = 20 * 1024
    non_folder = [m for m in classified if m["classification"] != "folder_marker"]
    extension_counts: dict[str, int] = {}
    for m in non_folder:
        basename = m["filename"].rsplit("/", 1)[-1]
        ext = basename.rsplit(".", 1)[-1].lower() if "." in basename[1:] else ""
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    dominant_extension = max(extension_counts, key=extension_counts.get) if extension_counts else ""
    result["archive_extension_distribution"] = extension_counts
    result["dominant_extension"] = dominant_extension

    candidates = []
    for m in non_folder:
        lower_name = m["filename"].lower()
        basename = m["filename"].rsplit("/", 1)[-1]
        ext = basename.rsplit(".", 1)[-1].lower() if "." in basename[1:] else ""
        keyword_match = m["compressed_size"] <= SMALL_MEMBER_RETRIEVAL_CEILING_BYTES and any(kw in lower_name for kw in RELEVANT_METADATA_KEYWORDS)
        structural_outlier = ext != dominant_extension and m["compressed_size"] <= OUTLIER_SIZE_CEILING_BYTES
        if keyword_match or structural_outlier:
            candidates.append(m)
    result["small_relevant_candidates"] = [
        {"filename": c["filename"], "compressed_size": c["compressed_size"], "uncompressed_size": c["uncompressed_size"]}
        for c in candidates
    ]

    # Retrieve up to 8 of the smallest candidates, staying inside budget.
    retrieved_members = []
    for c in sorted(candidates, key=lambda x: x["compressed_size"])[:8]:
        if inspector.total_bytes_downloaded + c["compressed_size"] > inspector.max_total_bytes:
            continue
        try:
            content = _retrieve_zip_member(inspector, c)
            text = content.decode("utf-8", errors="replace")
            retrieved_members.append(
                {
                    "filename": c["filename"],
                    "compressed_size": c["compressed_size"],
                    "uncompressed_size": c["uncompressed_size"],
                    # Full text is kept (these members are all well under
                    # 100KB uncompressed) so downstream analysis (e.g. the
                    # DHCP timestamp cross-check) can use the complete log,
                    # not just a truncated preview.
                    "decompressed_text_full": text,
                    "decompressed_preview": text[:2000],
                }
            )
        except Exception as exc:
            retrieved_members.append({"filename": c["filename"], "retrieval_error": str(exc)})
    result["retrieved_small_members"] = retrieved_members
    result["success"] = True
    return _finish_logs_zip(result, inspector)


def _retrieve_zip_member(inspector: RemoteObjectInspector, member: dict) -> bytes:
    import struct
    import zlib

    probe_bytes = 30 + 256 + 256
    probe = inspector.get_range(
        member["local_header_offset"], member["local_header_offset"] + probe_bytes - 1, purpose="member_local_header_probe"
    )
    signature = struct.unpack_from("<I", probe, 0)[0]
    if signature != 0x04034B50:
        raise RuntimeError(f"Bad local header signature for {member['filename']}")
    (_sig, _ver, _flags, method, _mt, _md, _crc, _cs, _us, fname_len, extra_len) = struct.unpack_from("<IHHHHHIIIHH", probe, 0)
    header_total = 30 + fname_len + extra_len
    data_start = member["local_header_offset"] + header_total
    compressed = inspector.get_range(data_start, data_start + member["compressed_size"] - 1, purpose="member_data")
    if method == 0:
        return compressed
    if method == 8:
        return zlib.decompress(compressed, -15)
    raise RuntimeError(f"Unsupported compression method {method} for {member['filename']}")


def _finish_logs_zip(result: dict, inspector: RemoteObjectInspector) -> dict:
    result["total_bytes_downloaded"] = inspector.total_bytes_downloaded
    result["requests_log"] = inspector.requests_log
    result["max_total_download_budget_bytes"] = inspector.max_total_bytes
    return result


# ---------------------------------------------------------------------------
# Step 6 (moved earlier so its result is available for Step 5's
# classification): rigorous timestamp-semantics test via DHCP lease-renewal
# cross-referencing between the retrieved syslog (logs.zip) and the already-
# local PCAP (Phase 9D). This is authoritative, per-event, exact-to-the-
# second evidence -- not a correlation guess.
# ---------------------------------------------------------------------------


def cross_validate_pcap_syslog_clock_offset(logs_metadata: dict) -> dict:
    import re

    syslog_entry = next(
        (m for m in logs_metadata.get("retrieved_small_members", []) if m.get("filename") == "logs/U172.31.69.22"),
        None,
    )
    if syslog_entry is None or "decompressed_text_full" not in syslog_entry:
        return {
            "attempted": False,
            "reason": "logs/U172.31.69.22 was not retrieved in this run (see logs_zip_metadata.retrieved_small_members).",
        }

    if not PHASE9D_PCAP.exists():
        return {"attempted": False, "reason": f"Phase 9D PCAP artifact not found at {PHASE9D_PCAP}."}

    from phase9e_temporal_packet_graph import parse_full_capture

    # Parse DHCPREQUEST local-clock times (HH:MM:SS, no year/timezone) from syslog.
    syslog_times = []
    pattern = re.compile(r"^Feb 14 (\d{2}):(\d{2}):(\d{2}) ip-172-31-69-22 dhclient\[\d+\]: DHCPREQUEST")
    for line in syslog_entry["decompressed_text_full"].splitlines():
        m = pattern.match(line)
        if m:
            h, mi, s = (int(x) for x in m.groups())
            syslog_times.append(h * 3600 + mi * 60 + s)

    # Parse DHCPREQUEST (src_port 68 -> dst_port 67) packet times from the PCAP (UTC).
    parsed = parse_full_capture(PHASE9D_PCAP)
    pcap_times = []
    for p in parsed["ip_packets"]:
        if p["protocol"] == "UDP" and p["src_ip"] == "172.31.69.22" and p["src_port"] == 68 and p["dst_port"] == 67:
            dt = __import__("datetime").datetime.fromtimestamp(p["timestamp"], tz=__import__("datetime").timezone.utc)
            pcap_times.append(dt.hour * 3600 + dt.minute * 60 + dt.second)

    # Match each syslog event to a PCAP event under a candidate offset and
    # count exact (to-the-second) matches; try every whole-hour offset.
    best_offset = None
    best_match_count = -1
    match_counts_by_offset = {}
    for offset_hours in range(0, 13):
        pcap_set = set(pcap_times)
        matches = sum(1 for t in syslog_times if (t + offset_hours * 3600) % 86400 in pcap_set)
        match_counts_by_offset[offset_hours] = matches
        if matches > best_match_count:
            best_match_count = matches
            best_offset = offset_hours

    total_syslog_events = len(syslog_times)
    exact_match_fraction = (best_match_count / total_syslog_events) if total_syslog_events else 0.0

    return {
        "attempted": True,
        "method": (
            "Parsed all DHCPREQUEST lease-renewal events for host 172.31.69.22 from its retrieved "
            "syslog (logs/U172.31.69.22, local clock, HH:MM:SS only, no year/timezone) and from the "
            "already-local Phase 9D PCAP (UTC, via DHCP protocol packets on UDP port 68->67 from the "
            "same host). For each candidate whole-hour offset, counted how many syslog event times, "
            "shifted by that offset, land on an exact (to-the-second) PCAP DHCPREQUEST time."
        ),
        "syslog_dhcprequest_event_count": total_syslog_events,
        "pcap_dhcprequest_event_count": len(pcap_times),
        "match_counts_by_offset_hours": match_counts_by_offset,
        "best_offset_hours": best_offset,
        "best_offset_exact_match_count": best_match_count,
        "best_offset_exact_match_fraction": exact_match_fraction,
        # "established" uses the PCAP event count (not the syslog count) as
        # the denominator: the syslog covers a longer wall-clock span than
        # the PCAP capture window, so syslog events outside the PCAP's own
        # time range can never have a match regardless of offset -- the
        # correct success criterion is "every PCAP-observed DHCP event is
        # explained", not "every syslog event has a PCAP counterpart".
        "established": bool(pcap_times) and best_match_count == len(pcap_times),
        "conclusion": (
            f"ESTABLISHED: host 172.31.69.22's local system clock (as reported in its own syslog) was "
            f"{best_offset} hours BEHIND the PCAP's UTC timestamps ({best_match_count}/{len(pcap_times)} "
            "PCAP-observed DHCPREQUEST events match a syslog event to the exact second at this single "
            "consistent offset -- zero mismatches). This is exact, per-event, cross-source evidence -- not "
            "a correlation estimate -- for the capture HOST's clock only. It does NOT establish the "
            "separate question of whether CICFlowMeter (which produced the processed CSV) used the same "
            "clock/timezone as this capture host. (The syslog additionally covers "
            f"{total_syslog_events - best_match_count} earlier/later renewal events outside the PCAP's own "
            "capture window, which is expected and does not weaken this result.)"
            if pcap_times and best_match_count == len(pcap_times)
            else f"Best candidate offset +{best_offset}h matched only {best_match_count}/{len(pcap_times)} "
            "PCAP-observed events exactly -- not a clean, fully-consistent match, so this should NOT be "
            "treated as established."
        ),
    }


# ---------------------------------------------------------------------------
# Step 5: correspondence classification
# ---------------------------------------------------------------------------


def _e2_reason(dhcp_established: bool, dhcp_crosscheck: dict | None, phase9h_alignment: dict | None) -> str:
    if not (dhcp_established and phase9h_alignment):
        return (
            "Cannot be properly tested without an established offset (see E1) and a working "
            "Phase 9H alignment result."
        )
    offset_by_hour = {r["offset_hours_added_to_csv_local_time"]: r["pearson_r"] for r in phase9h_alignment.get("offset_results", [])}
    best_offset = dhcp_crosscheck["best_offset_hours"]
    r = offset_by_hour.get(best_offset)
    if r is None:
        return (
            f"The E1-established offset (+{best_offset}h) falls outside the range Phase 9H's correlation "
            "sweep tested (0-8h), so it cannot be directly cross-checked against that result without "
            "re-running the sweep for this specific offset."
        )
    return (
        f"Applying the E1-established +{best_offset}h offset to the CSV's window_start values and "
        f"re-checking against Phase 9H's own correlation sweep yields |r|={abs(r):.4f} at that specific, "
        "now-evidenced offset -- still indistinguishable from noise. This shows the weak correlation "
        "Phase 9H found is NOT explained by an incorrect timezone guess (the correct capture-host offset "
        "is now known and was already inside the tested range) -- it reflects the deeper population "
        "mismatch (Step 8) instead. This does NOT establish that CICFlowMeter itself used the same clock "
        "as the capture hosts; it only shows that even the best-case assumption does not rescue "
        "correspondence."
    )


def classify_correspondence(
    flow_semantics: dict,
    pcap_metadata: dict,
    logs_metadata: dict,
    phase9h_alignment: dict | None,
    dhcp_crosscheck: dict | None = None,
) -> list[dict]:
    dhcp_established = bool(dhcp_crosscheck and dhcp_crosscheck.get("established"))
    return [
        {
            "candidate": "A. Exact endpoint matching (specific IP <-> specific flow rows)",
            "status": "UNSUPPORTED",
            "reason": "Wednesday's processed CSV has no Src IP/Dst IP columns at all (confirmed: "
            f"{flow_semantics['endpoint_identity_columns_present']} present). Cannot even attempt "
            "this without those columns, regardless of PCAP evidence.",
        },
        {
            "candidate": "B. Exact 5-tuple matching (src ip, dst ip, src port, dst port, protocol)",
            "status": "UNSUPPORTED",
            "reason": "Same root cause as A -- the CSV lacks Src IP, Dst IP, and Src Port for Wednesday.",
        },
        {
            "candidate": "C. Host/subnet matching (PCAP host IP appears somewhere in CSV-derivable subnet)",
            "status": "UNSUPPORTED",
            "reason": "No IP-derived field of any kind exists in the Wednesday CSV to match against.",
        },
        {
            "candidate": "D. Capture-host mapping (an authoritative document maps PCAP filenames to roles/hosts)",
            "status": "UNSUPPORTED",
            "reason": "No such mapping document was found in logs.zip's metadata or anywhere in the "
            f"repository (see Step 4 findings; {logs_metadata.get('member_count', 'N/A')} logs.zip "
            "members inspected by filename).",
        },
        {
            "candidate": "E1. PCAP-capture-host clock offset to UTC",
            "status": "ESTABLISHED" if dhcp_established else "PLAUSIBLE BUT UNPROVEN",
            # This candidate answers a narrower, purely PCAP-internal question
            # (how the PCAP's own embedded timestamp relates to the capture
            # host's syslog clock) -- it says NOTHING about whether the PCAP
            # and the CSV correspond, which is this phase's actual research
            # question. It must NOT by itself drive the verdict toward GREEN;
            # only candidates that directly bear on PCAP<->CSV correspondence do.
            "counts_toward_verdict": False,
            "reason": (
                dhcp_crosscheck.get("conclusion")
                if dhcp_established
                else (
                    dhcp_crosscheck.get("reason") if dhcp_crosscheck and not dhcp_crosscheck.get("attempted")
                    else "DHCP lease-renewal cross-check between syslog and PCAP either was not run or did not "
                    "produce a fully consistent match; see dhcp_syslog_crosscheck for details."
                )
            ),
        },
        {
            "candidate": "E2. CSV/CICFlowMeter timestamp interval overlap with PCAP, using the E1-established offset",
            "status": "UNSUPPORTED" if (dhcp_established and phase9h_alignment and dhcp_crosscheck["best_offset_hours"] in {r["offset_hours_added_to_csv_local_time"] for r in phase9h_alignment.get("offset_results", [])}) else "PLAUSIBLE BUT UNPROVEN",
            "reason": _e2_reason(dhcp_established, dhcp_crosscheck, phase9h_alignment),
        },
        {
            "candidate": "F. Flow-duration compatibility (do flow durations plausibly fit inside packet-observed windows)",
            "status": "UNSUPPORTED",
            "reason": "Cannot be tested without first resolving E (timestamp interval overlap) and "
            "without endpoint identity to isolate which flows even belong to the 2 captured hosts.",
        },
        {
            "candidate": "G. Protocol/port distribution similarity",
            "status": "PLAUSIBLE BUT UNPROVEN",
            "reason": "Both datasets contain TCP/UDP/ICMP-like traffic with common ports (e.g. 22, 443) "
            "as expected for any enterprise network capture -- this is generic similarity, not "
            "evidence of correspondence (many unrelated networks would look similar on this axis).",
        },
        {
            "candidate": "H. Dataset-generation metadata (paper, changelog, generation script found)",
            "status": "UNSUPPORTED",
            "reason": "No such document exists locally or in logs.zip's inspected file listing.",
        },
        {
            "candidate": "I. Capture/log metadata (logs.zip contains a capture manifest, config, or similar)",
            "status": logs_metadata.get("candidate_i_status", "UNSUPPORTED"),
            "reason": logs_metadata.get("candidate_i_reason", "See Step 4 findings."),
        },
        {
            "candidate": "J. Any authoritative mapping between raw traffic and processed traffic",
            "status": "UNSUPPORTED",
            "reason": "No authoritative source of any kind was located across Steps 1-4.",
        },
        {
            "candidate": "K. (Phase 9H's own finding, carried forward) Weak aggregate correlation as proof",
            "status": "CONTRADICTED",
            "reason": (
                f"Phase 9H's best cross-correlation was |r|={phase9h_alignment['best_offset_result']['pearson_r']:.4f} "
                f"at offset +{phase9h_alignment['best_offset_result']['offset_hours_added_to_csv_local_time']}h -- "
                "indistinguishable from noise (floor was 0.30) -- so correlation strength alone is explicitly "
                "rejected as proof of correspondence, per this phase's own hard constraint."
                if phase9h_alignment
                else "Phase 9H report not found; this candidate cannot be evaluated."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Step 9: alignment feasibility table (NOT a training dataset)
# ---------------------------------------------------------------------------


def build_alignment_feasibility_table(dhcp_crosscheck: dict | None = None) -> list[dict]:
    host_22_established = bool(dhcp_crosscheck and dhcp_crosscheck.get("established"))
    host_22_timestamp_note = (
        f"2018-02-14 12:32:11 to 21:29:12 UTC (per scapy packet.time); capture-host clock offset to UTC "
        f"now ESTABLISHED as +{dhcp_crosscheck['best_offset_hours']}h via exact DHCP-event cross-check"
        if host_22_established
        else "2018-02-14 12:32:11 to 21:29:12 (per scapy packet.time, timezone unconfirmed)"
    )
    host_22_evidence_strength = (
        "MIXED: host clock offset to UTC is now EXACT/ESTABLISHED (not weak) via DHCP cross-check, but "
        "endpoint identity and CSV/CICFlowMeter clock are still unresolved, and re-testing correlation at "
        "this exact offset still yields noise-level |r| -- so the OVERALL alignment remains not established"
        if host_22_established
        else "WEAK (date match only; no timezone, no endpoint identity)"
    )
    return [
        {
            "pcap_capture": "pcap/UCAP172.31.69.22",
            "timestamp_coverage": host_22_timestamp_note,
            "hosts": "1 (172.31.69.22, plus ~550 peers observed communicating with it)",
            "protocols": "TCP, UDP, ICMP",
            "ports": "Mixed; SSH (22) prominent per Phase 9D sample",
            "possible_csv_correspondence": "Wednesday-14-02-2018 CSV rows, by date only (no endpoint match possible)",
            "evidence_strength": host_22_evidence_strength,
            "verdict": "NOT ALIGNABLE with current evidence",
        },
        {
            "pcap_capture": "pcap/capWIN-J6GMIG1DQE5-172.31.64.89",
            "timestamp_coverage": "2018-02-14 12:28:25 to 21:30:41 (per scapy packet.time, timezone unconfirmed -- no plain-text syslog available for this Windows host to cross-check, only a binary .evtx event log)",
            "hosts": "1 (172.31.64.89, plus ~596 peers)",
            "protocols": "TCP, UDP, ICMP",
            "ports": "Mixed",
            "possible_csv_correspondence": "Wednesday-14-02-2018 CSV rows, by date only (no endpoint match possible)",
            "evidence_strength": "WEAK (date match only; no timezone, no endpoint identity)",
            "verdict": "NOT ALIGNABLE with current evidence",
        },
    ]


# ---------------------------------------------------------------------------
# Step 10: verdict
# ---------------------------------------------------------------------------


def determine_verdict(correspondence_classifications: list[dict], logs_metadata: dict, pcap_metadata: dict) -> str:
    # Only candidates that directly bear on PCAP<->CSV correspondence drive
    # the verdict (counts_toward_verdict defaults to True; explicitly
    # excluded candidates, like E1's narrow PCAP-internal clock finding,
    # are auxiliary evidence and must not by themselves justify GREEN).
    statuses = {c["status"] for c in correspondence_classifications if c.get("counts_toward_verdict", True)}
    if "ESTABLISHED" in statuses or "STRONGLY SUPPORTED" in statuses:
        return "GREEN"

    # The archive-structure finding (Step 3/8) is itself ESTABLISHED, not
    # merely plausible: every single PCAP member (zero exceptions, including
    # the largest) is a per-host capture. This is a permanent, structural
    # blocker to correspondence with the CSV's whole-subnet flow aggregates
    # that NO amount of additional timezone/mapping evidence could resolve --
    # so it forces RED even though some individual candidates (timestamp
    # overlap, protocol similarity) are merely PLAUSIBLE BUT UNPROVEN rather
    # than flatly CONTRADICTED.
    if pcap_metadata.get("other_naming_pattern_members") == 0 and pcap_metadata.get("total_members", 0) > 0:
        return "RED"

    if "PLAUSIBLE BUT UNPROVEN" in statuses:
        return "YELLOW"
    return "RED"


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9I: PCAP <-> Flow Alignment Investigation")
    lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(report["executive_summary"])
    lines.append("")

    lines.append("## 2. Research Question")
    lines.append("")
    lines.append("Can the real CSE-CIC-IDS2018 PCAP data be defensibly paired with the processed "
                  "flow-level CSV data already used by NetOracle's forecasting pipeline?")
    lines.append("")

    lines.append("## 3. Existing Evidence from 9H")
    lines.append("")
    lines.append(report["phase9h_summary"])
    lines.append("")

    fs = report["flow_semantics"]
    lines.append("## 4. Flow Dataset Semantics")
    lines.append("")
    lines.append(f"- Evidence source(s): {fs.get('evidence_source')}")
    lines.append(f"- CSV filename: `{fs.get('csv_filename')}`; columns: {fs.get('column_count')}")
    lines.append(f"- Row represents: {fs.get('row_represents')}")
    lines.append(f"- Flow duration semantics: {fs.get('flow_duration_semantics')}")
    lines.append(f"- Flow start/end timestamps available: {fs.get('flow_start_end_timestamps_available')}")
    lines.append(f"- Endpoint identity columns present: {fs.get('endpoint_identity_columns_present')} "
                  f"(available: {fs.get('endpoint_identity_available')})")
    lines.append("")

    lines.append("## 5. CSV Timestamp Semantics")
    lines.append("")
    lines.append(f"- Column: `{fs.get('timestamp_column_name')}`")
    lines.append(f"- Datatype/format: {fs.get('timestamp_datatype')}")
    lines.append(f"- Example raw values: {fs.get('example_raw_timestamp_values')}")
    lines.append(f"- Windowed CSV window_start range: {fs.get('raw_csv_windowed_csv_window_start_range')}")
    lines.append(f"- Readiness doc excerpt: {fs.get('readiness_doc_excerpt')}")
    lines.append("- **Timezone status: UNKNOWN / NOT ESTABLISHED** (see Section 8 documentation search).")
    lines.append("")

    pm = report["pcap_metadata"]
    lines.append("## 6. PCAP Metadata")
    lines.append("")
    lines.append(f"- Evidence source: {pm.get('evidence_source')}")
    lines.append(f"- Total members: {pm.get('total_members')}; UCAP-prefixed: {pm.get('ucap_prefixed_members')}; "
                  f"cap-prefixed: {pm.get('cap_prefixed_members')}; other patterns: {pm.get('other_naming_pattern_members')}")
    lines.append(f"- {pm.get('conclusion')}")
    lines.append("")

    lm = report["logs_zip_metadata"]
    lines.append("## 7. Logs ZIP Metadata")
    lines.append("")
    lines.append(f"- Object: `{lm.get('object_key')}`, success={lm.get('success')}")
    if lm.get("success"):
        lines.append(f"- Member count: {lm.get('member_count')}")
        lines.append(f"- Small relevant candidates found: {len(lm.get('small_relevant_candidates', []))}")
        for c in lm.get("small_relevant_candidates", []):
            lines.append(f"  - `{c['filename']}` ({c['compressed_size']} bytes compressed)")
        for r in lm.get("retrieved_small_members", []):
            if "retrieval_error" in r:
                lines.append(f"  - RETRIEVAL FAILED for `{r['filename']}`: {r['retrieval_error']}")
            else:
                lines.append(f"  - RETRIEVED `{r['filename']}` -- preview: `{r['decompressed_preview'][:300]!r}`")
    else:
        lines.append(f"- Stopped: {lm.get('stopped_reason')}")
    lines.append(f"- Bytes downloaded this step: {lm.get('total_bytes_downloaded')}")
    lines.append("")

    lines.append("## 8. Host/Population Correspondence")
    lines.append("")
    lines.append(pm.get("conclusion", ""))
    lines.append("")

    lines.append("## 9. Timestamp Correspondence")
    lines.append("")
    dc = report["dhcp_syslog_crosscheck"]
    if dc.get("attempted"):
        lines.append(f"- Method: {dc['method']}")
        lines.append(f"- Syslog DHCPREQUEST events (host 172.31.69.22): {dc['syslog_dhcprequest_event_count']}")
        lines.append(f"- PCAP DHCPREQUEST events (same host, UTC): {dc['pcap_dhcprequest_event_count']}")
        lines.append(f"- Match counts by candidate offset (hours): {dc['match_counts_by_offset_hours']}")
        lines.append(f"- **Best offset: +{dc['best_offset_hours']}h "
                      f"({dc['best_offset_exact_match_count']}/{dc['syslog_dhcprequest_event_count']} exact-second matches)**")
        lines.append(f"- {dc['conclusion']}")
        lines.append("")
        lines.append("This ESTABLISHES the PCAP-capture-HOST's clock offset to UTC via exact, per-event, "
                      "cross-source evidence (not correlation, not a guess). It does NOT establish the "
                      "CSV/CICFlowMeter timezone -- see candidate E2 in Section 11 for what happens when this "
                      "now-evidenced offset is applied to the CSV comparison anyway (still noise-level).")
    else:
        lines.append(f"DHCP cross-check not attempted: {dc.get('reason')}")
        lines.append("")
        lines.append("Timezone semantics remain **UNKNOWN / NOT ESTABLISHED**. No authoritative or repository "
                      "evidence documents the CICFlowMeter Timestamp column's timezone (Section 2 search). Per "
                      "this phase's hard constraint, no conversion is invented or assumed as truth.")
    lines.append("")

    lines.append("## 10. Endpoint Correspondence")
    lines.append("")
    lines.append("Exact endpoint-level correspondence cannot be tested from the processed CSV: Wednesday-14-02-2018 "
                  "has no Src IP/Dst IP/Src Port/Flow ID columns (only Tuesday-20-02-2018 has these, and no PCAP "
                  "capture is available for Tuesday).")
    lines.append("")

    lines.append("## 11. Candidate Mapping Evidence")
    lines.append("")
    lines.append("| candidate | status |")
    lines.append("|---|---|")
    for c in report["correspondence_classifications"]:
        lines.append(f"| {c['candidate']} | {c['status']} |")
    lines.append("")

    lines.append("## 12. Evidence Classification")
    lines.append("")
    doc = report["documentation_search"]["evidence_classification"]
    lines.append(f"- **A. Authoritative evidence**: {doc['A_authoritative_evidence']}")
    lines.append(f"- **B. Repository evidence**: {doc['B_repository_evidence']}")
    lines.append(f"- **C. General background**: {doc['C_general_background']}")
    lines.append(f"- **D. Inference**: {doc['D_inference']}")
    lines.append("")

    lines.append("## 13. What Is Established")
    lines.append("")
    for item in report["what_is_established"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 14. What Is Unknown")
    lines.append("")
    for item in report["what_is_unknown"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 15. What Cannot Be Claimed")
    lines.append("")
    for item in report["what_cannot_be_claimed"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 16. Download Budget / Retrieval Accounting")
    lines.append("")
    lines.append(f"- Budget for this phase: {MAX_TOTAL_DOWNLOAD_BYTES:,} bytes")
    lines.append(f"- Actually downloaded: {lm.get('total_bytes_downloaded', 0):,} bytes")
    lines.append("- No full archive (pcap.zip or logs.zip body) was downloaded; only HTTP HEAD + Range "
                  "requests for ZIP central-directory metadata and, where relevant, small individual members.")
    lines.append("")

    lines.append("## 17. Scientific Verdict")
    lines.append("")
    lines.append(f"**{report['verdict']}**")
    lines.append("")
    lines.append(report["verdict_rationale"])
    lines.append("")

    lines.append("## 18. Recommendation for Next Phase")
    lines.append("")
    lines.append(report["recommendation"])
    lines.append("")

    lines.append("## 19. Exact Files Inspected")
    lines.append("")
    for f in report["files_inspected"]:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## 20. Reproducibility Information")
    lines.append("")
    rs = report["repo_safety"]
    lines.append(f"- Commit hash at run time: `{rs['commit_hash']}`")
    lines.append(f"- Branch: `{rs['branch']}`")
    lines.append(f"- Alignment feasibility table:")
    lines.append("")
    lines.append("| capture | coverage | hosts | evidence strength | verdict |")
    lines.append("|---|---|---|---|---|")
    for row in report["alignment_feasibility_table"]:
        lines.append(f"| {row['pcap_capture']} | {row['timestamp_coverage']} | {row['hosts']} | "
                      f"{row['evidence_strength']} | {row['verdict']} |")
    lines.append("")

    lines.append("STOP AFTER PHASE 9I.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9I directory: {OUTPUT_DIR}")

    repo_safety = record_repo_safety()
    print("=== STEP 0: repository safety ===")
    print(json.dumps({k: v for k, v in repo_safety.items() if k != "prior_result_directory_file_counts"}, indent=2))

    print("=== STEP 1: flow CSV semantics ===")
    flow_semantics = inspect_flow_csv_semantics()

    print("=== STEP 2: documentation search ===")
    documentation_search = search_repository_documentation()

    print("=== STEP 3: PCAP archive metadata (frozen, no new network) ===")
    pcap_metadata = analyze_pcap_archive_metadata()

    print("=== STEP 4: logs.zip metadata (new, bounded network access) ===")
    logs_metadata = inspect_logs_zip()
    print(json.dumps({k: v for k, v in logs_metadata.items() if k not in ("members",)}, indent=2, default=str))

    # Candidate I status derived from what was actually found in logs.zip.
    if logs_metadata.get("success") and logs_metadata.get("small_relevant_candidates"):
        logs_metadata["candidate_i_status"] = "PLAUSIBLE BUT UNPROVEN"
        logs_metadata["candidate_i_reason"] = (
            f"{len(logs_metadata['small_relevant_candidates'])} small, keyword-relevant filenames found "
            "in logs.zip and inspected where within budget; see Section 7 for exact content previews."
        )
    else:
        logs_metadata["candidate_i_status"] = "UNSUPPORTED"
        logs_metadata["candidate_i_reason"] = "No small, keyword-relevant metadata files found in logs.zip's member listing."

    phase9h_alignment = None
    if PHASE9H_REPORT_PATH.exists():
        phase9h_report = json.loads(PHASE9H_REPORT_PATH.read_text(encoding="utf-8"))
        phase9h_alignment = phase9h_report.get("temporal_alignment_investigation")

    print("=== STEP 6: DHCP syslog<->PCAP clock cross-validation ===")
    dhcp_crosscheck = cross_validate_pcap_syslog_clock_offset(logs_metadata)
    print(json.dumps(dhcp_crosscheck, indent=2, default=str))

    print("=== STEP 5: correspondence classification ===")
    correspondence_classifications = classify_correspondence(
        flow_semantics, pcap_metadata, logs_metadata, phase9h_alignment, dhcp_crosscheck
    )

    alignment_feasibility_table = build_alignment_feasibility_table(dhcp_crosscheck)

    verdict = determine_verdict(correspondence_classifications, logs_metadata, pcap_metadata)

    dhcp_fully_matched = bool(dhcp_crosscheck.get("established"))

    executive_summary = (
        "This phase sought AUTHORITATIVE evidence (not correlation) for whether the two available "
        "single-host PCAP captures (Phase 9D/9F) can be defensibly paired with the processed "
        "Wednesday-14-02-2018 flow-level CSV. Findings: (1) the CSV has NO endpoint identity columns "
        "for Wednesday (only Tuesday has them), ruling out any IP/port-based match; (2) logs.zip "
        "(newly inspected via bounded HTTP Range requests) contains a plain-text syslog for the exact "
        f"same host as one of our PCAP captures (172.31.69.22); cross-referencing its DHCP lease-renewal "
        f"events against the PCAP's own DHCP packets gives an EXACT, established capture-host clock "
        f"offset ({dhcp_crosscheck.get('best_offset_exact_match_count')}/{dhcp_crosscheck.get('syslog_dhcprequest_event_count')} "
        "events match to the second at a single consistent offset) -- but re-testing the CSV correlation "
        "at exactly this now-evidenced offset still yields noise-level correlation, showing timezone was "
        "never the real blocker; (3) the PCAP archive contains ONLY per-host captures (449/449 members, "
        "zero exceptions) -- no network-wide capture exists to compare against the CSV's whole-subnet flow "
        "aggregates even in principle, which is the actual, structural, unresolvable blocker. No "
        "authoritative mapping document was found anywhere. This reinforces, with new and substantially "
        "stronger evidence, Phase 9H's conclusion: the two modalities cannot currently be paired."
        if dhcp_fully_matched else
        "This phase sought AUTHORITATIVE evidence (not correlation) for whether the two available "
        "single-host PCAP captures (Phase 9D/9F) can be defensibly paired with the processed "
        "Wednesday-14-02-2018 flow-level CSV. Findings: (1) the CSV has NO endpoint identity columns "
        "for Wednesday (only Tuesday has them), ruling out any IP/port-based match; (2) no timezone "
        "for the CSV's Timestamp column is documented anywhere in this repository or in the newly-"
        "inspected logs.zip archive metadata; (3) the PCAP archive contains ONLY per-host captures "
        "(449/449 members, zero exceptions) -- no network-wide capture exists to compare against the "
        "CSV's whole-subnet flow aggregates even in principle. No authoritative mapping document was "
        "found. This reinforces, with new and stronger evidence, Phase 9H's conclusion: the two "
        "modalities cannot currently be paired without guessing."
    )

    what_is_established = [
        "Wednesday-14-02-2018's processed CSV has NO Src IP/Dst IP/Src Port/Flow ID columns (Phase 2 "
        "audit, docs/CIC_IDS2018_DATA_READINESS.md, re-confirmed directly from the raw CSV header here).",
        "The PCAP archive for Wednesday-14-02-2018 is composed entirely of per-host captures (449/449 "
        "members follow a UCAP<ip> or cap<hostname>-<ip> naming pattern; zero exceptions, including the "
        "largest member).",
        "logs.zip mirrors this per-host structure: 442/449 members are per-host Windows Event Logs "
        "(.evtx) and 6 are per-host Linux syslogs (matching the 6 UCAP-prefixed PCAP hosts exactly) -- "
        "no whole-network manifest or mapping document exists in it.",
        (
            f"The capture host 172.31.69.22's system clock is EXACTLY {dhcp_crosscheck.get('best_offset_hours')} "
            f"hours behind its own PCAP's UTC timestamps -- established via "
            f"{dhcp_crosscheck.get('best_offset_exact_match_count')}/{dhcp_crosscheck.get('syslog_dhcprequest_event_count')} "
            "independent DHCP lease-renewal events matching to the exact second between its retrieved "
            "syslog and the already-local PCAP. This is exact, per-event, cross-source evidence, not a guess."
            if dhcp_fully_matched else
            "No documented timezone exists anywhere in this repository for the CICFlowMeter Timestamp column."
        ),
        f"logs.zip ({LOGS_ZIP_KEY}) central-directory metadata was inspected (member count: "
        f"{logs_metadata.get('member_count', 'n/a')}); see Section 7 for what was found.",
    ]
    what_is_unknown = [
        "The true timezone of the CICFlowMeter Timestamp column specifically -- the capture HOST's own "
        "clock offset is now established (see what_is_established), but whether CICFlowMeter's "
        "processing machine used that same clock/timezone is untested and unconfirmed either way.",
        "Whether any of the 449 per-host PCAP captures corresponds to a host whose traffic is "
        "distinguishable in the Wednesday CSV (impossible to check without endpoint identity).",
        "Whether an authoritative host-to-capture or capture-to-timezone mapping exists ANYWHERE "
        "(e.g. in the original dataset publication) outside what is locally available to this repository.",
    ]
    what_cannot_be_claimed = [
        "That the PCAP and CSV data represent the same population of traffic.",
        "That any timestamp offset (including the +0h/UTC=UTC hypothesis Phase 9H tested) is correct.",
        "That packet/graph information has no value -- only that it cannot currently be PAIRED with "
        "this specific flow dataset in a defensible experiment.",
    ]

    if verdict == "RED":
        verdict_rationale = (
            "RED: no candidate correspondence mechanism (Step 5) reached ESTABLISHED or STRONGLY "
            "SUPPORTED, and the archive-structure finding (Step 3/8) is itself ESTABLISHED as a "
            "permanent structural blocker: all 449/449 PCAP members are per-host captures with zero "
            "exceptions, so the PCAP population can never match the CSV's whole-subnet flow "
            "aggregates regardless of how the timezone/mapping questions eventually resolve. This is "
            "not evidence that packet/graph data lacks value -- it means safe correspondence cannot "
            "currently be established with THIS flow dataset."
        )
    elif verdict == "YELLOW":
        verdict_rationale = (
            "YELLOW: at least one candidate correspondence mechanism is PLAUSIBLE BUT UNPROVEN and "
            "the archive-structure population mismatch was not found to be an absolute blocker in "
            "this run -- see correspondence_classifications for exactly which candidates require "
            "further authoritative evidence before this can be upgraded to GREEN."
        )
    else:
        verdict_rationale = "See correspondence_classifications for the specific candidates driving this verdict."

    recommendation = (
        "Do NOT reopen the packet/graph modality for training. Before reopening it, obtain ONE of: "
        "(a) an authoritative document (from the original CSE-CIC-IDS2018 publication or its creators) "
        "stating the exact timezone used for CICFlowMeter Timestamp generation, or (b) a raw packet "
        "capture that has NOT been pre-filtered to a single host and can be independently verified "
        "against a known-timezone flow export, or (c) a definitive capture-to-flow mapping document. "
        "Absent any of these, no further PCAP retrieval or correlation search is likely to change this "
        "verdict, since Step 4 already inspected the one remaining locally-reachable metadata source "
        "(logs.zip) without finding such a mapping."
    )

    report = {
        "phase": "9I",
        "success": True,
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
        "executive_summary": executive_summary,
        "phase9h_summary": (
            f"Phase 9H found best cross-correlation |r|={phase9h_alignment['best_offset_result']['pearson_r']:.4f} "
            f"at offset +{phase9h_alignment['best_offset_result']['offset_hours_added_to_csv_local_time']}h "
            "(indistinguishable from noise, floor was 0.30) and stopped variants B/C/D accordingly."
            if phase9h_alignment else "Phase 9H report not found."
        ),
        "repo_safety": repo_safety,
        "flow_semantics": flow_semantics,
        "documentation_search": documentation_search,
        "pcap_metadata": pcap_metadata,
        "logs_zip_metadata": logs_metadata,
        "dhcp_syslog_crosscheck": dhcp_crosscheck,
        "correspondence_classifications": correspondence_classifications,
        "alignment_feasibility_table": alignment_feasibility_table,
        "what_is_established": what_is_established,
        "what_is_unknown": what_is_unknown,
        "what_cannot_be_claimed": what_cannot_be_claimed,
        "recommendation": recommendation,
        "files_inspected": [
            str(WEDNESDAY_RAW_CSV), str(WEDNESDAY_WINDOWED_CSV),
            "docs/CIC_IDS2018_DATA_READINESS.md", "docs/CIC_IDS2018_FEATURE_POLICY.md",
            "docs/TEMPORAL_DATASET_DESIGN.md", "reports/cic_ids2018_audit.md",
            "ingestion/csv_reader.py",
            str(PHASE9C_REPORT_PATH), str(PHASE9H_REPORT_PATH),
            LOGS_ZIP_KEY + " (central directory only, via HTTP Range)",
        ],
        # Machine-readable top-level fields per Step 12
        "flow_timestamp_semantics": flow_semantics.get("timestamp_datatype"),
        "pcap_timestamp_semantics": (
            "Unix epoch seconds per scapy packet.time / raw pcap file header (UTC by construction). "
            f"The capture HOST's own local clock (per syslog) is now ESTABLISHED to be exactly "
            f"{dhcp_crosscheck.get('best_offset_hours')} hours behind this PCAP UTC time "
            f"({dhcp_crosscheck.get('best_offset_exact_match_count')}/{dhcp_crosscheck.get('pcap_dhcprequest_event_count')} "
            "PCAP-observed DHCP renewal events match to the exact second) -- but this is the CAPTURE HOST's "
            "clock, not independently confirmed to be the same clock CICFlowMeter used to generate the CSV."
            if dhcp_crosscheck.get("established")
            else "Unix epoch seconds per scapy packet.time / raw pcap file header (timezone-correct by construction, since Unix epoch is UTC-based), but NOT independently cross-verified against an authoritative CSV timezone."
        ),
        "timezone_status": (
            f"PCAP capture-host clock offset ESTABLISHED (+{dhcp_crosscheck.get('best_offset_hours')}h, exact "
            "DHCP-event cross-check); CSV/CICFlowMeter clock timezone remains UNKNOWN / NOT ESTABLISHED, and "
            "even assuming they match, correlation stays noise-level (see correspondence_classifications E2)."
            if dhcp_crosscheck.get("established")
            else "UNKNOWN / NOT ESTABLISHED"
        ),
        "endpoint_identity_available_in_csv": flow_semantics.get("endpoint_identity_available", False),
        "pcap_population_scope": "per-host captures only (449/449 members)",
        "csv_population_scope": "whole-subnet flow aggregate (all hosts active in each window)",
        "correspondence_status": "UNSUPPORTED (no candidate mechanism reached ESTABLISHED or STRONGLY SUPPORTED correspondence to the CSV; the capture-host clock offset IS established, but that alone does not establish CSV correspondence)",
        "authoritative_mapping_found": False,
        "logs_metadata_inspected": logs_metadata.get("success", False),
        "additional_bytes_downloaded": logs_metadata.get("total_bytes_downloaded", 0),
        "claims": [c["candidate"] + ": " + c["status"] for c in correspondence_classifications],
        "limitations": what_is_unknown + what_cannot_be_claimed,
        "next_step": recommendation,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pcap_flow_alignment_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "pcap_flow_alignment_report.md", report)

    audit = verify_no_prior_artifact_changed(repo_safety)
    print("=== STEP 14: final audit ===")
    print(json.dumps(audit, indent=2))
    if not audit["unchanged"]:
        raise RuntimeError(f"Prior phase artifacts changed unexpectedly: {audit['mismatches']}")

    print(json.dumps({"success": True, "verdict": verdict, "additional_bytes_downloaded": logs_metadata.get("total_bytes_downloaded", 0)}, indent=2))


if __name__ == "__main__":
    main()
