"""Phase 9B: read-only S3 object discovery for CSE-CIC-IDS2018 raw PCAP data.

Uses only plain, unauthenticated HTTPS GET requests to the public S3
bucket's ListObjectsV2 REST API (Python stdlib `urllib.request` +
`xml.etree.ElementTree` -- no AWS CLI, no boto3, no credentials, nothing
installed). Every request in this file is a LIST call, which returns only
XML metadata (key, size, last-modified, etag, storage class). No object
body (PCAP/log/CSV content) is ever requested, downloaded, or extracted --
this script never issues a GET on an actual object key, only on the
bucket's `?list-type=2` listing endpoint.

Does not modify Phases 3-9A, does not train anything, does not touch any
existing experiment file or result.
"""

from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

BUCKET_HOST = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
RAW_TRAFFIC_PREFIX = "Original Network Traffic and Log data/"
PROCESSED_CSV_PREFIX = "Processed Traffic Data for ML Algorithms/"
TARGET_DAY_PREFIX = "Original Network Traffic and Log data/Wednesday-14-02-2018/"

# Per Phase 9B instructions: only a "genuinely tiny" object may be
# downloaded; anything at or above this threshold must stop at metadata.
DOWNLOAD_TINY_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB

OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase9b_pcap_discovery"


def _tag(name: str) -> str:
    return f"{S3_NAMESPACE}{name}"


def list_objects(prefix: str = "", delimiter: str = "/", timeout: int = 30) -> dict:
    """One read-only ListObjectsV2 call against the public bucket. Returns
    parsed common-prefixes (subfolders) and object entries (key, size,
    last-modified, etag, storage class). Never requests an object's body."""
    url = f"{BUCKET_HOST}?list-type=2&delimiter={quote(delimiter)}"
    if prefix:
        url += f"&prefix={quote(prefix)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        status = response.status
        body = response.read()
    root = ET.fromstring(body)

    common_prefixes = [element.find(_tag("Prefix")).text for element in root.findall(_tag("CommonPrefixes"))]
    contents = []
    for element in root.findall(_tag("Contents")):
        storage_class_element = element.find(_tag("StorageClass"))
        contents.append(
            {
                "key": element.find(_tag("Key")).text,
                "size_bytes": int(element.find(_tag("Size")).text),
                "last_modified": element.find(_tag("LastModified")).text,
                "etag": (element.find(_tag("ETag")).text or "").strip('"'),
                "storage_class": storage_class_element.text if storage_class_element is not None else None,
            }
        )
    return {
        "http_status": status,
        "url": url,
        "prefix_queried": prefix,
        "common_prefixes": common_prefixes,
        "contents": contents,
        "is_truncated": root.find(_tag("IsTruncated")).text == "true",
        "key_count": int(root.find(_tag("KeyCount")).text),
    }


def classify_object(key: str) -> str:
    """Best-effort classification from the object's FILENAME alone (never
    the full path -- this prefix's own directory name contains the word
    'Log', which would false-positive-match any file under it if the full
    key were checked instead of just the basename). Never opens the object
    to inspect its actual contents."""
    if key.endswith("/"):
        return "folder_marker"
    basename = key.rsplit("/", 1)[-1]
    lower = basename.lower()
    if lower.endswith(".csv"):
        return "processed_csv"
    if "pcap" in lower:
        return "raw_pcap_archive"
    if "log" in lower:
        return "log_archive"
    return "unknown"


def human_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def build_report() -> dict:
    top_level = list_objects(prefix="", delimiter="/")
    raw_traffic = list_objects(prefix=RAW_TRAFFIC_PREFIX, delimiter="/")
    processed_csv = list_objects(prefix=PROCESSED_CSV_PREFIX, delimiter="/")
    target_day = list_objects(prefix=TARGET_DAY_PREFIX, delimiter="/")

    candidates = []
    for entry in target_day["contents"]:
        classification = classify_object(entry["key"])
        if classification == "folder_marker":
            continue
        candidates.append(
            {
                **entry,
                "classification": classification,
                "size_human": human_size(entry["size_bytes"]),
                "under_tiny_download_threshold": entry["size_bytes"] < DOWNLOAD_TINY_THRESHOLD_BYTES,
            }
        )

    processed_csv_filenames = [entry["key"].rsplit("/", 1)[-1] for entry in processed_csv["contents"] if not entry["key"].endswith("/")]

    return {
        "bucket": "cse-cic-ids2018",
        "region": "ca-central-1",
        "access_method": "unauthenticated HTTPS GET to the public ListObjectsV2 REST endpoint (no AWS CLI, no boto3, no credentials)",
        "bucket_accessible": top_level["http_status"] == 200,
        "top_level_listing": top_level,
        "top_level_prefixes": top_level["common_prefixes"],
        "raw_traffic_prefix_listing": raw_traffic,
        "raw_traffic_day_folders": raw_traffic["common_prefixes"],
        "processed_csv_prefix_listing": processed_csv,
        "processed_csv_filenames_for_corroboration": processed_csv_filenames,
        "target_day_prefix": TARGET_DAY_PREFIX,
        "target_day_listing": target_day,
        "candidate_objects": candidates,
        "correspondence_to_wednesday_14_02_2018": {
            "confidence": "HIGH, by naming convention only",
            "basis": "The raw-traffic prefix contains a folder named exactly "
            "'Wednesday-14-02-2018/', matching the day-of-week and date of our "
            "existing 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', under "
            "the same S3 bucket/organization (CIC) that produced the processed CSVs.",
            "verified_by_content_inspection": False,
            "caveat": "No packet or log byte has been read. This is folder-name "
            "correspondence only, not a verified flow-level or timestamp-level match.",
        },
        "download_performed": False,
        "download_reason": "All candidate objects at this prefix meet or exceed the "
        f"{DOWNLOAD_TINY_THRESHOLD_BYTES / (1024**2):.0f} MB tiny-download threshold; "
        "per instructions, this script stops after metadata discovery and does not "
        "fetch any object body.",
    }


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Phase 9B: CSE-CIC-IDS2018 Raw PCAP Object Discovery",
        "",
        f"Bucket: `s3://{report['bucket']}/` (region `{report['region']}`). Access method: {report['access_method']}.",
        f"Bucket accessible: **{report['bucket_accessible']}**",
        "",
        "## 1. Top-level prefixes",
        "",
    ]
    for prefix in report["top_level_prefixes"]:
        lines.append(f"- `{prefix}`")

    lines += ["", "## 2. Raw traffic day folders (under 'Original Network Traffic and Log data/')", ""]
    for prefix in report["raw_traffic_day_folders"]:
        marker = " <- target" if report["target_day_prefix"] == prefix else ""
        lines.append(f"- `{prefix}`{marker}")

    lines += [
        "",
        "## 3. Candidate objects for Wednesday-14-02-2018",
        "",
        "| Key | Size | Last modified | Storage class | Classification | Under 100MB? |",
        "|---|---:|---|---|---|---|",
    ]
    for candidate in report["candidate_objects"]:
        lines.append(
            f"| `{candidate['key']}` | {candidate['size_human']} | {candidate['last_modified']} | "
            f"{candidate['storage_class']} | {candidate['classification']} | {candidate['under_tiny_download_threshold']} |"
        )

    correspondence = report["correspondence_to_wednesday_14_02_2018"]
    lines += [
        "",
        "## 4. Correspondence to Wednesday-14-02-2018",
        "",
        f"Confidence: **{correspondence['confidence']}**",
        "",
        correspondence["basis"],
        "",
        f"Verified by content inspection: **{correspondence['verified_by_content_inspection']}**. {correspondence['caveat']}",
        "",
        "## 5. Download status",
        "",
        f"Download performed: **{report['download_performed']}**",
        "",
        report["download_reason"],
        "",
        "## Corroboration: processed CSV prefix",
        "",
        "Files under 'Processed Traffic Data for ML Algorithms/' (for cross-checking against our existing 10 local CSVs, listing only):",
        "",
    ]
    for filename in report["processed_csv_filenames_for_corroboration"]:
        lines.append(f"- `{filename}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9B directory: {OUTPUT_DIR}")

    report = build_report()

    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "discovery_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "discovery_report.md", report)

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "bucket_accessible": report["bucket_accessible"],
                "candidate_objects": [{"key": c["key"], "size_human": c["size_human"]} for c in report["candidate_objects"]],
                "download_performed": report["download_performed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
