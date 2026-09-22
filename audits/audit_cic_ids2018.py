"""Audit CIC-IDS-2018 flow CSVs without changing or training the application."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


EXPECTED_FILES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
]

TIMESTAMP_FORMATS = ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S")
NUMERIC_COLUMNS = (
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "RST Flag Cnt",
    "Pkt Len Mean",
    "Pkt Size Avg",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
)


def _parse_number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return math.nan


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, timestamp_format)
        except ValueError:
            continue
    return None


def _new_numeric_stats() -> dict[str, dict[str, float | int | None]]:
    return {
        column: {
            "non_empty": 0,
            "blank": 0,
            "invalid": 0,
            "non_finite": 0,
            "min": None,
            "max": None,
        }
        for column in NUMERIC_COLUMNS
    }


def audit_file(path: Path, chunk_size: int) -> dict:
    csv.field_size_limit(sys.maxsize)
    size_bytes = path.stat().st_size
    row_count = 0
    malformed_rows = 0
    repeated_header_rows = 0
    replacement_characters = 0
    width_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    header: list[str] = []
    blank_by_column: Counter[str] = Counter()
    timestamp_invalid = 0
    timestamp_year_counts: Counter[str] = Counter()
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    numeric_stats = _new_numeric_stats()
    chunks_processed = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "file": path.name,
                "path": str(path),
                "size_bytes": size_bytes,
                "status": "empty",
                "header": [],
                "column_count": 0,
                "row_count": 0,
            }

        header = [column.strip() for column in header]
        expected_width = len(header)
        width_counts[str(expected_width)] += 1
        chunk_rows = 0

        for row in reader:
            row_count += 1
            chunk_rows += 1
            replacement_characters += sum(value.count("\ufffd") for value in row)
            width_counts[str(len(row))] += 1
            if len(row) != expected_width:
                malformed_rows += 1
                if chunk_rows >= chunk_size:
                    chunks_processed += 1
                    chunk_rows = 0
                continue

            values = dict(zip(header, row))
            if values.get("Label", "").strip() == "Label":
                repeated_header_rows += 1
            label = values.get("Label", "").strip()
            if label:
                label_counts[label] += 1

            timestamp_text = values.get("Timestamp", "")
            timestamp = _parse_timestamp(timestamp_text)
            if timestamp is None:
                timestamp_invalid += 1
            else:
                year = str(timestamp.year)
                timestamp_year_counts[year] += 1
                timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
                timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)

            for column in header:
                if not values[column].strip():
                    blank_by_column[column] += 1
            for column, stats in numeric_stats.items():
                if column not in values:
                    continue
                parsed = _parse_number(values[column])
                if parsed is None:
                    stats["blank"] = int(stats["blank"]) + 1
                elif math.isnan(parsed):
                    stats["invalid"] = int(stats["invalid"]) + 1
                elif not math.isfinite(parsed):
                    stats["non_finite"] = int(stats["non_finite"]) + 1
                else:
                    stats["non_empty"] = int(stats["non_empty"]) + 1
                    stats["min"] = parsed if stats["min"] is None else min(float(stats["min"]), parsed)
                    stats["max"] = parsed if stats["max"] is None else max(float(stats["max"]), parsed)

            if chunk_rows >= chunk_size:
                chunks_processed += 1
                chunk_rows = 0

    if chunk_rows:
        chunks_processed += 1
    return {
        "file": path.name,
        "path": str(path),
        "size_bytes": size_bytes,
        "status": "ok",
        "header": header,
        "column_count": len(header),
        "row_count": row_count,
        "chunks_processed": chunks_processed,
        "malformed_rows": malformed_rows,
        "row_width_counts": dict(width_counts),
        "repeated_header_rows": repeated_header_rows,
        "replacement_characters": replacement_characters,
        "blank_cells_by_column": dict(blank_by_column),
        "timestamp": {
            "invalid": timestamp_invalid,
            "year_counts": dict(timestamp_year_counts),
            "min": timestamp_min.isoformat(sep=" ") if timestamp_min else None,
            "max": timestamp_max.isoformat(sep=" ") if timestamp_max else None,
        },
        "label_counts": dict(label_counts),
        "numeric_stats": numeric_stats,
    }


def _markdown_report(report: dict) -> str:
    files = report["files"]
    lines = [
        "# CSE-CIC-IDS2018 CSV Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This report was produced by `audits/audit_cic_ids2018.py`. Each file was read independently in streaming chunks; no data files, training code, or model artifacts were modified.",
        "",
        "## Summary",
        "",
        f"- Requested files: {report['requested_file_count']}",
        f"- Files found: {report['found_file_count']}",
        f"- Files missing: {report['missing_file_count']}",
        f"- Total data rows: {sum(item.get('row_count', 0) for item in files):,}",
        f"- Total input bytes: {sum(item.get('size_bytes', 0) for item in files):,}",
        "",
        "| File | Rows | Columns | Malformed width | Repeated headers | Invalid timestamps | Labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in files:
        lines.append(
            f"| `{item['file']}` | {item.get('row_count', 0):,} | {item.get('column_count', 0)} | "
            f"{item.get('malformed_rows', 0):,} | {item.get('repeated_header_rows', 0):,} | "
            f"{item.get('timestamp', {}).get('invalid', 0):,} | {len(item.get('label_counts', {}))} distinct |"
        )
    lines.extend(["", "## Schema findings", ""])
    for schema, names in report["schema_groups"].items():
        lines.append(f"- {schema} columns: {', '.join(f'`{name}`' for name in names)}")
    lines.extend(["", "## Per-file details", ""])
    for item in files:
        lines.extend(
            [
                f"### `{item['file']}`",
                "",
                f"- Size: {item.get('size_bytes', 0):,} bytes",
                f"- Timestamp range: `{item.get('timestamp', {}).get('min')}` to `{item.get('timestamp', {}).get('max')}`",
                f"- Timestamp years: `{item.get('timestamp', {}).get('year_counts', {})}`",
                f"- Labels: `{item.get('label_counts', {})}`",
                f"- Blank cells: `{item.get('blank_cells_by_column', {})}`",
                f"- Numeric parse/range stats: `{item.get('numeric_stats', {})}`",
                "",
            ]
        )
    return "\n".join(lines)


def run_audit(input_dir: Path, output_dir: Path, files: Iterable[str], chunk_size: int) -> tuple[Path, Path]:
    requested = list(files)
    results = []
    for filename in requested:
        path = input_dir / filename
        if path.exists():
            results.append(audit_file(path, chunk_size))
        else:
            results.append({"file": filename, "path": str(path), "status": "missing"})

    schema_groups: dict[str, list[str]] = {}
    for item in results:
        if item.get("status") == "ok":
            schema_key = str(item["header"])
            schema_groups.setdefault(schema_key, []).append(item["file"])
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_directory": str(input_dir),
        "chunk_size": chunk_size,
        "requested_file_count": len(requested),
        "found_file_count": sum(item.get("status") == "ok" for item in results),
        "missing_file_count": sum(item.get("status") == "missing" for item in results),
        "schema_groups": schema_groups,
        "files": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cic_ids2018_audit.json"
    markdown_path = output_dir / "cic_ids2018_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    json_path, markdown_path = run_audit(args.input_dir, args.output_dir, EXPECTED_FILES, args.chunk_size)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()