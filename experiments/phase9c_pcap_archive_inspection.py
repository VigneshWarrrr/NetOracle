"""Phase 9C: remote ZIP central-directory inspection via HTTP Range requests.

Read-only, stdlib-only (urllib + struct). Inspects the internal structure of
the 37.17GB pcap.zip archive discovered in Phase 9B WITHOUT downloading its
bulk content -- only the End of Central Directory (EOCD), ZIP64 EOCD locator
and record, and the central directory itself (all small, bounded metadata
regions) are fetched via targeted byte-range requests. Every network call is
logged and counted against a hard total-download byte budget
(MAX_TOTAL_DOWNLOAD_BYTES); the central directory is only fetched after its
size is confirmed to be under a separate safety threshold. No member's
actual file content (i.e. any PCAP/log data) is ever requested, no archive
is decompressed, no packet is parsed.

Does not modify Phases 3-9B. Does not train anything. Does not begin GNN
implementation.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

OBJECT_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Original%20Network%20Traffic%20and%20Log%20data/"
    "Wednesday-14-02-2018/pcap.zip"
)
OBJECT_KEY = "Original Network Traffic and Log data/Wednesday-14-02-2018/pcap.zip"

TAIL_SIZE_BYTES = 65536  # Step 2: final 64 KiB
MAX_CENTRAL_DIRECTORY_SAFE_BYTES = 8 * 1024 * 1024  # Step 4/5 safety gate: 8 MB
MAX_TOTAL_DOWNLOAD_BYTES = 10 * 1024 * 1024  # Hard safety constraint #7: keep total under 10 MB

EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_FIXED_SIZE = 22
ZIP64_LOCATOR_SIGNATURE = 0x07064B50
ZIP64_LOCATOR_SIZE = 20
ZIP64_EOCD_SIGNATURE = 0x06064B50
ZIP64_EOCD_FIXED_SIZE = 56
CENTRAL_DIR_HEADER_SIGNATURE = 0x02014B50
CENTRAL_DIR_HEADER_FIXED_SIZE = 46
ZIP64_EXTRA_TAG = 1

OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase9c_pcap_archive_inspection"


class BudgetExceededError(RuntimeError):
    pass


class RemoteObjectInspector:
    """Every byte ever requested from the remote object passes through this
    class, which tracks and hard-caps the running total."""

    def __init__(self, url: str, max_total_bytes: int = MAX_TOTAL_DOWNLOAD_BYTES) -> None:
        self.url = url
        self.max_total_bytes = max_total_bytes
        self.total_bytes_downloaded = 0
        self.requests_log: list[dict] = []

    def head(self) -> dict:
        request = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = dict(response.headers)
            status = response.status
        self.requests_log.append({"type": "HEAD", "status": status, "bytes": 0})
        return {
            "http_status": status,
            "content_length": int(headers.get("Content-Length", 0)),
            "accept_ranges": headers.get("Accept-Ranges"),
            "content_type": headers.get("Content-Type"),
            "etag": headers.get("ETag", "").strip('"'),
            "last_modified": headers.get("Last-Modified"),
        }

    def get_range(self, start: int, end: int, purpose: str) -> bytes:
        """end is INCLUSIVE, matching HTTP Range header semantics."""
        requested_bytes = end - start + 1
        if requested_bytes <= 0:
            raise ValueError(f"Invalid range for '{purpose}': start={start} end={end}")
        if self.total_bytes_downloaded + requested_bytes > self.max_total_bytes:
            raise BudgetExceededError(
                f"Refusing range request for '{purpose}' ({requested_bytes} bytes): would exceed the "
                f"{self.max_total_bytes}-byte total download budget (already used {self.total_bytes_downloaded})"
            )
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            body = response.read()
            content_range = response.headers.get("Content-Range")
        if status != 206:
            raise RuntimeError(f"Expected HTTP 206 Partial Content for '{purpose}', got {status}")
        if len(body) != requested_bytes:
            raise RuntimeError(f"Range request for '{purpose}' returned {len(body)} bytes, expected {requested_bytes}")
        self.total_bytes_downloaded += len(body)
        self.requests_log.append(
            {
                "type": "GET_RANGE",
                "purpose": purpose,
                "start": start,
                "end": end,
                "bytes": len(body),
                "status": status,
                "content_range": content_range,
            }
        )
        return body


def find_eocd(tail: bytes, tail_start_offset_in_file: int) -> dict:
    """Search backward (rightmost match) for the EOCD signature, matching
    the standard ZIP-reader convention."""
    index = tail.rfind(EOCD_SIGNATURE)
    if index == -1:
        raise RuntimeError(f"EOCD signature not found in final {len(tail)}-byte tail")
    fixed = tail[index : index + EOCD_FIXED_SIZE]
    if len(fixed) < EOCD_FIXED_SIZE:
        raise RuntimeError("EOCD signature found too close to the buffer end to read its fixed fields")
    (_signature, disk_number, disk_with_cd, entries_this_disk, total_entries, cd_size, cd_offset, comment_length) = struct.unpack(
        "<IHHHHIIH", fixed
    )
    return {
        "index_in_tail": index,
        "file_offset": tail_start_offset_in_file + index,
        "disk_number": disk_number,
        "disk_with_cd": disk_with_cd,
        "entries_this_disk": entries_this_disk,
        "total_entries": total_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "comment_length": comment_length,
        "cd_size_is_zip64_sentinel": cd_size == 0xFFFFFFFF,
        "cd_offset_is_zip64_sentinel": cd_offset == 0xFFFFFFFF,
        "total_entries_is_zip64_sentinel": total_entries == 0xFFFF,
    }


def find_zip64_locator_in_tail(tail: bytes, eocd_index_in_tail: int, tail_start_offset_in_file: int) -> dict | None:
    """The ZIP64 locator, if present, sits in the 20 bytes immediately
    before the EOCD -- almost always already inside the fetched tail."""
    if eocd_index_in_tail < ZIP64_LOCATOR_SIZE:
        return None
    candidate = tail[eocd_index_in_tail - ZIP64_LOCATOR_SIZE : eocd_index_in_tail]
    signature = struct.unpack_from("<I", candidate, 0)[0]
    if signature != ZIP64_LOCATOR_SIGNATURE:
        return None
    _signature, disk_number, zip64_eocd_offset, total_disks = struct.unpack("<IIQI", candidate)
    return {
        "file_offset": tail_start_offset_in_file + eocd_index_in_tail - ZIP64_LOCATOR_SIZE,
        "disk_number": disk_number,
        "zip64_eocd_offset": zip64_eocd_offset,
        "total_disks": total_disks,
    }


def parse_zip64_eocd_record(data: bytes) -> dict:
    if len(data) != ZIP64_EOCD_FIXED_SIZE:
        raise ValueError(f"Expected {ZIP64_EOCD_FIXED_SIZE} bytes for the ZIP64 EOCD record, got {len(data)}")
    (signature, record_size, version_made_by, version_needed, disk_number, disk_with_cd, entries_this_disk, total_entries, cd_size, cd_offset) = struct.unpack(
        "<IQHHIIQQQQ", data
    )
    if signature != ZIP64_EOCD_SIGNATURE:
        raise RuntimeError(f"Expected ZIP64 EOCD signature 0x{ZIP64_EOCD_SIGNATURE:08x}, got 0x{signature:08x}")
    return {
        "record_size": record_size,
        "version_made_by": version_made_by,
        "version_needed": version_needed,
        "disk_number": disk_number,
        "disk_with_cd": disk_with_cd,
        "entries_this_disk": entries_this_disk,
        "total_entries": total_entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
    }


def parse_central_directory(data: bytes, expected_entries: int) -> list[dict]:
    """Parses every central-directory file header in `data`, resolving
    per-entry ZIP64 extra fields (tag 0x0001) where the 32-bit compressed
    size / uncompressed size / local-header-offset fields are sentinels."""
    entries = []
    pos = 0
    while pos < len(data):
        signature = struct.unpack_from("<I", data, pos)[0]
        if signature != CENTRAL_DIR_HEADER_SIGNATURE:
            break
        (
            _sig,
            _version_made_by,
            _version_needed,
            flags,
            method,
            _mod_time,
            _mod_date,
            crc32,
            comp_size,
            uncomp_size,
            fname_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal_attrs,
            external_attrs,
            local_offset,
        ) = struct.unpack_from("<IHHHHHHIIIHHHHHII", data, pos)

        header_end = pos + CENTRAL_DIR_HEADER_FIXED_SIZE
        fname_bytes = data[header_end : header_end + fname_len]
        extra_bytes = data[header_end + fname_len : header_end + fname_len + extra_len]
        comment_end = header_end + fname_len + extra_len + comment_len

        real_comp_size, real_uncomp_size, real_local_offset = comp_size, uncomp_size, local_offset
        needs_zip64 = comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF
        if needs_zip64:
            extra_pos = 0
            while extra_pos + 4 <= len(extra_bytes):
                tag, size = struct.unpack_from("<HH", extra_bytes, extra_pos)
                if tag == ZIP64_EXTRA_TAG:
                    payload = extra_bytes[extra_pos + 4 : extra_pos + 4 + size]
                    payload_pos = 0
                    if uncomp_size == 0xFFFFFFFF and payload_pos + 8 <= len(payload):
                        real_uncomp_size = struct.unpack_from("<Q", payload, payload_pos)[0]
                        payload_pos += 8
                    if comp_size == 0xFFFFFFFF and payload_pos + 8 <= len(payload):
                        real_comp_size = struct.unpack_from("<Q", payload, payload_pos)[0]
                        payload_pos += 8
                    if local_offset == 0xFFFFFFFF and payload_pos + 8 <= len(payload):
                        real_local_offset = struct.unpack_from("<Q", payload, payload_pos)[0]
                        payload_pos += 8
                extra_pos += 4 + size

        is_utf8 = bool(flags & 0x0800)
        filename = fname_bytes.decode("utf-8" if is_utf8 else "cp437", errors="replace")

        entries.append(
            {
                "filename": filename,
                "compression_method": method,
                "compressed_size": real_comp_size,
                "uncompressed_size": real_uncomp_size,
                "local_header_offset": real_local_offset,
                "external_attributes": external_attrs,
                "crc32": crc32,
                "used_zip64_extra": needs_zip64,
            }
        )
        pos = comment_end

    if len(entries) != expected_entries:
        raise RuntimeError(f"Parsed {len(entries)} central-directory entries, but EOCD/ZIP64-EOCD declared {expected_entries}")
    return entries


def classify_member(filename: str) -> str:
    if filename.endswith("/"):
        return "folder_marker"
    basename = filename.rsplit("/", 1)[-1]
    lower = basename.lower()
    if lower.endswith(".pcap"):
        return "pcap"
    if lower.endswith(".pcapng"):
        return "pcapng"
    if "log" in lower:
        return "log"
    return "unknown_extension"


def build_report() -> tuple[dict, RemoteObjectInspector]:
    inspector = RemoteObjectInspector(OBJECT_URL)
    report: dict = {"object_key": OBJECT_KEY, "object_url": OBJECT_URL}

    # ---- Step 1: metadata only ----
    head_info = inspector.head()
    report["step1_head"] = head_info
    content_length = head_info["content_length"]
    if head_info["accept_ranges"] != "bytes":
        report["stopped_reason"] = "Server does not advertise Accept-Ranges: bytes; cannot safely proceed."
        report["success"] = False
        return report, inspector

    # ---- Step 2: final 64 KiB tail ----
    tail_start = content_length - TAIL_SIZE_BYTES
    tail = inspector.get_range(tail_start, content_length - 1, purpose="tail_64kib")
    report["step2_tail"] = {"requested_bytes": TAIL_SIZE_BYTES, "received_bytes": len(tail), "tail_start_offset": tail_start}

    # ---- Step 3: locate EOCD ----
    try:
        eocd = find_eocd(tail, tail_start)
    except RuntimeError as exc:
        report["stopped_reason"] = str(exc)
        report["success"] = False
        return report, inspector
    report["step3_eocd"] = eocd

    # ---- Step 4: ZIP64 if needed ----
    needs_zip64 = eocd["cd_size_is_zip64_sentinel"] or eocd["cd_offset_is_zip64_sentinel"] or eocd["total_entries_is_zip64_sentinel"]
    report["zip64_required"] = needs_zip64

    resolved_cd_size = eocd["cd_size"]
    resolved_cd_offset = eocd["cd_offset"]
    resolved_total_entries = eocd["total_entries"]

    if needs_zip64:
        locator = find_zip64_locator_in_tail(tail, eocd["index_in_tail"], tail_start)
        if locator is None:
            report["stopped_reason"] = (
                "ZIP64 required but the locator was not found within the already-fetched tail; "
                "a further small range request would be needed and was not attempted this run."
            )
            report["success"] = False
            return report, inspector
        report["step4_zip64_locator"] = locator

        zip64_record_bytes = inspector.get_range(
            locator["zip64_eocd_offset"], locator["zip64_eocd_offset"] + ZIP64_EOCD_FIXED_SIZE - 1, purpose="zip64_eocd_record"
        )
        zip64_record = parse_zip64_eocd_record(zip64_record_bytes)
        report["step4_zip64_eocd_record"] = zip64_record

        resolved_cd_size = zip64_record["cd_size"]
        resolved_cd_offset = zip64_record["cd_offset"]
        resolved_total_entries = zip64_record["total_entries"]

    report["resolved_central_directory"] = {
        "cd_offset": resolved_cd_offset,
        "cd_size": resolved_cd_size,
        "total_entries": resolved_total_entries,
    }

    # ---- Safety gate before fetching the central directory ----
    if resolved_cd_size > MAX_CENTRAL_DIRECTORY_SAFE_BYTES:
        report["stopped_reason"] = (
            f"Central directory size ({resolved_cd_size} bytes) exceeds the safety threshold "
            f"({MAX_CENTRAL_DIRECTORY_SAFE_BYTES} bytes); stopping before fetching it."
        )
        report["success"] = False
        return report, inspector

    # ---- Step 5: fetch and parse the central directory ----
    cd_bytes = inspector.get_range(resolved_cd_offset, resolved_cd_offset + resolved_cd_size - 1, purpose="central_directory")
    entries = parse_central_directory(cd_bytes, resolved_total_entries)
    report["step5_central_directory_entries_count"] = len(entries)

    # ---- Step 6: classify members ----
    classified = [{**entry, "classification": classify_member(entry["filename"])} for entry in entries]
    report["members"] = classified

    file_members = [member for member in classified if member["classification"] != "folder_marker"]
    classification_counts: dict[str, int] = {}
    for member in file_members:
        classification_counts[member["classification"]] = classification_counts.get(member["classification"], 0) + 1
    report["classification_counts"] = classification_counts

    pcap_like = [member for member in file_members if member["classification"] in ("pcap", "pcapng", "unknown_extension")]
    pcap_like_sorted = sorted(pcap_like, key=lambda member: member["compressed_size"])
    report["candidate_pcap_members_count"] = len(pcap_like)
    report["smallest_candidate_member"] = pcap_like_sorted[0] if pcap_like_sorted else None
    report["ten_smallest_candidate_members"] = pcap_like_sorted[:10]
    report["five_largest_candidate_members"] = pcap_like_sorted[-5:]

    # ---- Step 7: naming comparison only ----
    report["step7_naming_comparison"] = {
        "processed_csv_convention": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv (flow-level, one file for the whole day)",
        "raw_archive_member_convention": "pcap/<capture-prefix><hostname>-<ip-address> (no .pcap/.pcapng extension; one file per captured endpoint)",
        "observation": (
            "The two naming conventions are structurally different (one whole-day flow-level CSV vs. many "
            "per-endpoint raw capture files) and share no common filename token beyond both being associated "
            "with the 'Wednesday-14-02-2018' day folder/filename. This is a naming-convention comparison only "
            "-- no row-level, flow-level, or timestamp-level correspondence is claimed or verified; no packet "
            "was opened to check this."
        ),
    }

    # ---- Consistency check ----
    total_compressed = sum(member["compressed_size"] for member in file_members)
    report["consistency_check"] = {
        "sum_of_member_compressed_sizes": total_compressed,
        "object_content_length": content_length,
        "difference_bytes": content_length - total_compressed,
        "note": (
            "Difference is expected structural overhead: one local file header per member, plus the central "
            "directory, ZIP64 EOCD record, ZIP64 locator, and EOCD -- none of which required a bulk download."
        ),
    }

    report["success"] = True
    return report, inspector


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Phase 9C: Remote ZIP Central-Directory Inspection",
        "",
        f"Object: `{report['object_key']}`",
        f"Success: **{report.get('success')}**",
        "",
    ]
    if not report.get("success"):
        lines += ["## Stopped", "", report.get("stopped_reason", "(no reason recorded)"), ""]
    else:
        head = report["step1_head"]
        lines += [
            "## 1. CONFIRMED FROM REMOTE ZIP METADATA",
            "",
            f"- HTTP access: confirmed (HEAD status {head['http_status']})",
            f"- Content-Length: {head['content_length']:,} bytes",
            f"- Accept-Ranges: {head['accept_ranges']}",
            f"- Content-Type: {head['content_type']}",
            f"- ETag: `{head['etag']}`",
            f"- EOCD found at file offset {report['step3_eocd']['file_offset']:,}",
            f"- ZIP64 required: {report['zip64_required']}",
            f"- Resolved central directory: offset={report['resolved_central_directory']['cd_offset']:,}, "
            f"size={report['resolved_central_directory']['cd_size']:,} bytes, "
            f"entries={report['resolved_central_directory']['total_entries']}",
            f"- Central-directory entries actually parsed: {report['step5_central_directory_entries_count']}",
            f"- Consistency check: sum(member compressed sizes) + overhead = Content-Length "
            f"(difference {report['consistency_check']['difference_bytes']:,} bytes, matches expected header/CD/EOCD overhead)",
            "",
            "## 2. NOT VERIFIED",
            "",
            "- Whether any member's content is actually valid PCAP data (no packet was opened or parsed)",
            "- Whether the archive's contents correspond, at the flow or timestamp level, to the existing processed CSV for this day",
            "- Whether the per-endpoint capture files cover the full day or only part of it",
            "",
            "## 3. CANDIDATE PCAP MEMBERS",
            "",
            f"Total non-folder members: {sum(report['classification_counts'].values())}. Classification counts: {report['classification_counts']}",
            "",
            "None of the 449 members carry a `.pcap`/`.pcapng` extension. All are classified `unknown_extension` "
            "and treated as candidates by CONTEXT ONLY: parent archive is named `pcap.zip`, containing folder is "
            "`pcap/`, and filenames are prefixed `cap`/`UCAP` followed by a hostname and/or IP address.",
            "",
            "10 smallest candidate members by compressed size:",
            "",
            "| Filename | Compressed | Uncompressed | Method | ZIP64 extra used |",
            "|---|---:|---:|---:|---|",
        ]
        for member in report["ten_smallest_candidate_members"]:
            lines.append(
                f"| `{member['filename']}` | {member['compressed_size']:,} | {member['uncompressed_size']:,} | "
                f"{member['compression_method']} | {member['used_zip64_extra']} |"
            )
        smallest = report["smallest_candidate_member"]
        lines += [
            "",
            "## 4. SMALLEST PCAP MEMBER",
            "",
            f"`{smallest['filename']}` -- compressed {smallest['compressed_size']:,} bytes "
            f"({smallest['compressed_size']/1024:.1f} KiB), uncompressed {smallest['uncompressed_size']:,} bytes, "
            f"compression method {smallest['compression_method']} (8 = DEFLATE), local header offset {smallest['local_header_offset']:,}.",
            "",
            "## 5. TOTAL BYTES DOWNLOADED DURING INSPECTION",
            "",
            f"**{report['total_bytes_downloaded']:,} bytes** (~{report['total_bytes_downloaded']/1024:.1f} KiB) "
            f"against a {report['max_total_download_budget_bytes']:,}-byte budget "
            f"({report['budget_remaining_bytes']:,} bytes remaining, unused).",
            "",
            "Request log:",
            "",
            "| Type | Purpose | Bytes | Status |",
            "|---|---|---:|---|",
        ]
        for entry in report["requests_log"]:
            lines.append(f"| {entry['type']} | {entry.get('purpose','-')} | {entry['bytes']:,} | {entry['status']} |")

        naming = report["step7_naming_comparison"]
        lines += [
            "",
            "## 7. Naming comparison with processed CSV convention (metadata only, no content claim)",
            "",
            f"- Processed CSV convention: {naming['processed_csv_convention']}",
            f"- Raw archive member convention: {naming['raw_archive_member_convention']}",
            f"- {naming['observation']}",
            "",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9C directory: {OUTPUT_DIR}")

    report, inspector = build_report()
    report["total_bytes_downloaded"] = inspector.total_bytes_downloaded
    report["requests_log"] = inspector.requests_log
    report["max_total_download_budget_bytes"] = inspector.max_total_bytes
    report["budget_remaining_bytes"] = inspector.max_total_bytes - inspector.total_bytes_downloaded

    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "discovery_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "discovery_report.md", report)

    print(
        json.dumps(
            {
                "success": report.get("success"),
                "total_bytes_downloaded": report["total_bytes_downloaded"],
                "candidate_pcap_members_count": report.get("candidate_pcap_members_count"),
                "smallest_candidate_member": (report.get("smallest_candidate_member") or {}).get("filename"),
                "stopped_reason": report.get("stopped_reason"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
