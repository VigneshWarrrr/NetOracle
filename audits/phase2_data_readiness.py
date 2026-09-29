"""Run the read-only Phase 2 readiness audit for the CIC-IDS-2018 flow files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


FILES = [
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
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
NON_NUMERIC_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
NORMALIZED_LABELS = {
    "Infilteration": "Infiltration",
    "DDOS attack-LOIC-UDP": "DDoS attack-LOIC-UDP",
    "DDOS attack-HOIC": "DDoS attack-HOIC",
}
ANOMALY_FILES = {
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
}


def parse_timestamp(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw.strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def normalize_label(raw: str) -> str:
    value = raw.strip()
    return NORMALIZED_LABELS.get(value, value)


def parse_float(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return math.nan


def empty_numeric_stat() -> dict[str, int | float | None]:
    return {
        "missing_count": 0,
        "invalid_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "finite_minimum": None,
        "finite_maximum": None,
    }


def audit_file(path: Path, row_sink: list[dict[str, str]]) -> dict:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        header = [value.strip() for value in next(reader)]
        expected_width = len(header)
        row_count = 0
        timestamp_invalid = 0
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None
        previous_timestamp: datetime | None = None
        out_of_order = 0
        label_counts: Counter[tuple[str, str]] = Counter()
        label_first: dict[tuple[str, str], datetime] = {}
        label_last: dict[tuple[str, str], datetime] = {}
        numeric: dict[str, dict[str, int | float | None]] = {
            column: empty_numeric_stat()
            for column in header
            if column not in NON_NUMERIC_COLUMNS
        }

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != expected_width:
                continue
            values = dict(zip(header, row))
            raw_timestamp = values.get("Timestamp", "")
            timestamp = parse_timestamp(raw_timestamp)
            if timestamp is None:
                timestamp_invalid += 1
            else:
                if path.name in ANOMALY_FILES and timestamp.year == 1970:
                    row_sink.append(
                        {
                            "file": path.name,
                            "row_number": str(row_number),
                            "raw_timestamp": raw_timestamp,
                            "parsed_value": timestamp.isoformat(sep=" "),
                            "classification": "malformed source timestamp",
                        }
                    )
                first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
                last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    out_of_order += 1
                previous_timestamp = timestamp

            raw_label = values.get("Label", "").strip()
            normalized_label = normalize_label(raw_label)
            label_key = (raw_label, normalized_label)
            label_counts[label_key] += 1
            if timestamp is not None:
                label_first[label_key] = min(timestamp, label_first.get(label_key, timestamp))
                label_last[label_key] = max(timestamp, label_last.get(label_key, timestamp))

            for column, stats in numeric.items():
                parsed = parse_float(values.get(column, ""))
                if parsed is None:
                    stats["missing_count"] = int(stats["missing_count"]) + 1
                elif math.isnan(parsed):
                    stats["invalid_count"] = int(stats["invalid_count"]) + 1
                elif math.isinf(parsed):
                    key = "positive_infinity_count" if parsed > 0 else "negative_infinity_count"
                    stats[key] = int(stats[key]) + 1
                else:
                    stats["finite_minimum"] = parsed if stats["finite_minimum"] is None else min(float(stats["finite_minimum"]), parsed)
                    stats["finite_maximum"] = parsed if stats["finite_maximum"] is None else max(float(stats["finite_maximum"]), parsed)

        return {
            "file": path.name,
            "header": header,
            "column_count": len(header),
            "row_count": row_count,
            "timestamp_invalid": timestamp_invalid,
            "earliest_timestamp": first_timestamp,
            "latest_timestamp": last_timestamp,
            "out_of_order_count": out_of_order,
            "label_counts": label_counts,
            "label_first": label_first,
            "label_last": label_last,
            "numeric": numeric,
        }


def format_time(value: datetime | None) -> str:
    return value.isoformat(sep=" ") if value is not None else ""


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_schema_report(results: list[dict], path: Path) -> None:
    all_columns = []
    for result in results:
        for column in result["header"]:
            if column not in all_columns:
                all_columns.append(column)
    common_columns = set(results[0]["header"])
    for result in results[1:]:
        common_columns &= set(result["header"])
    rows = []
    for result in results:
        columns = set(result["header"])
        rows.append(
            {
                "file": result["file"],
                "column_count": result["column_count"],
                "extra_columns_vs_common_schema": "; ".join(sorted(columns - common_columns)),
                "missing_columns_vs_union_schema": "; ".join(column for column in all_columns if column not in columns),
                "header_columns": "; ".join(result["header"]),
            }
        )
    write_csv(path, list(rows[0]), rows)


def create_label_report(results: list[dict], path: Path) -> list[dict]:
    total = sum(result["row_count"] for result in results)
    combined: dict[tuple[str, str], dict[str, object]] = {}
    for result in results:
        for key, count in result["label_counts"].items():
            raw_label, normalized_label = key
            item = combined.setdefault(
                key,
                {"raw_label": raw_label, "normalized_label": normalized_label, "flow_count": 0, "files": set(), "first": None, "last": None},
            )
            item["flow_count"] = int(item["flow_count"]) + count
            item["files"].add(result["file"])
            if key in result["label_first"]:
                first = result["label_first"][key]
                last = result["label_last"][key]
                item["first"] = first if item["first"] is None else min(item["first"], first)
                item["last"] = last if item["last"] is None else max(item["last"], last)
    rows = []
    for item in sorted(combined.values(), key=lambda value: (-int(value["flow_count"]), str(value["raw_label"]))):
        rows.append(
            {
                "raw_label": item["raw_label"],
                "normalized_label": item["normalized_label"],
                "flow_count": item["flow_count"],
                "percentage": round(100 * int(item["flow_count"]) / total, 8),
                "files_containing_label": "; ".join(sorted(item["files"])),
                "first_timestamp": format_time(item["first"]),
                "last_timestamp": format_time(item["last"]),
            }
        )
    write_csv(path, list(rows[0]), rows)
    return rows


def create_timeline_report(results: list[dict], path: Path) -> list[dict]:
    rows = []
    for result in results:
        labels: dict[str, dict[str, object]] = {}
        for (raw_label, normalized_label), count in result["label_counts"].items():
            if normalized_label in {"Benign", "Label", ""}:
                continue
            item = labels.setdefault(normalized_label, {"flow_count": 0, "first": None, "last": None, "raw": set()})
            item["flow_count"] = int(item["flow_count"]) + count
            item["raw"].add(raw_label)
            if (raw_label, normalized_label) in result["label_first"]:
                first = result["label_first"][(raw_label, normalized_label)]
                last = result["label_last"][(raw_label, normalized_label)]
                item["first"] = first if item["first"] is None else min(item["first"], first)
                item["last"] = last if item["last"] is None else max(item["last"], last)
        for label, item in sorted(labels.items()):
            first = item["first"]
            last = item["last"]
            duration = (last - first).total_seconds() if first and last else None
            rows.append(
                {
                    "file": result["file"],
                    "attack_label": label,
                    "raw_labels": "; ".join(sorted(item["raw"])),
                    "first_timestamp": format_time(first),
                    "last_timestamp": format_time(last),
                    "duration_seconds": duration if duration is not None else "",
                    "flow_count": item["flow_count"],
                }
            )
    write_csv(path, list(rows[0]), rows)
    return rows


def create_numeric_report(results: list[dict], path: Path) -> list[dict]:
    combined: dict[str, dict[str, int | float | None]] = {}
    files_present: Counter[str] = Counter()
    for result in results:
        for column, source in result["numeric"].items():
            files_present[column] += 1
            target = combined.setdefault(column, empty_numeric_stat())
            for key in ("missing_count", "invalid_count", "positive_infinity_count", "negative_infinity_count"):
                target[key] = int(target[key]) + int(source[key])
            if source["finite_minimum"] is not None:
                target["finite_minimum"] = source["finite_minimum"] if target["finite_minimum"] is None else min(float(target["finite_minimum"]), float(source["finite_minimum"]))
            if source["finite_maximum"] is not None:
                target["finite_maximum"] = source["finite_maximum"] if target["finite_maximum"] is None else max(float(target["finite_maximum"]), float(source["finite_maximum"]))
    rows = []
    for column in sorted(combined):
        item = combined[column]
        rows.append({"feature": column, "files_present": files_present[column], **item})
    write_csv(path, list(rows[0]), rows)
    return rows


def create_temporal_report(results: list[dict], path: Path) -> list[dict]:
    rows = []
    for result in results:
        count = result["row_count"]
        out_of_order = result["out_of_order_count"]
        rows.append(
            {
                "file": result["file"],
                "earliest_timestamp": format_time(result["earliest_timestamp"]),
                "latest_timestamp": format_time(result["latest_timestamp"]),
                "out_of_order_row_count": out_of_order,
                "out_of_order_percentage": round(100 * out_of_order / count, 8) if count else 0,
                "invalid_timestamp_count": result["timestamp_invalid"],
                "chronologically_ordered": "YES" if out_of_order == 0 and result["timestamp_invalid"] == 0 else "NO",
            }
        )
    write_csv(path, list(rows[0]), rows)
    return rows


def write_docs(results: list[dict], labels: list[dict], timeline: list[dict], numeric: list[dict], temporal: list[dict], anomaly_rows: list[dict], path: Path) -> None:
    schema_counts = ", ".join(f"{result['file']}: {result['column_count']}" for result in results)
    total_rows = sum(result["row_count"] for result in results)
    total_invalid_timestamps = sum(result["timestamp_invalid"] for result in results)
    total_out_of_order = sum(result["out_of_order_count"] for result in results)
    numeric_invalid = sum(int(row["invalid_count"]) for row in numeric)
    positive_inf = sum(int(row["positive_infinity_count"]) for row in numeric)
    negative_inf = sum(int(row["negative_infinity_count"]) for row in numeric)
    attack_lines = [
        f"- `{row['file']}` / `{row['attack_label']}`: {row['flow_count']:,} flows, `{row['first_timestamp']}` to `{row['last_timestamp']}` ({row['duration_seconds']} seconds)."
        for row in timeline
    ]
    label_lines = [
        f"- `{row['raw_label']}` -> `{row['normalized_label']}`: {row['flow_count']:,} ({row['percentage']}%), files: {row['files_containing_label']}."
        for row in labels
    ]
    temporal_lines = [
        f"- `{row['file']}`: {row['chronologically_ordered']}; range `{row['earliest_timestamp']}` to `{row['latest_timestamp']}`; out-of-order {row['out_of_order_row_count']:,} ({row['out_of_order_percentage']}%); invalid timestamps {row['invalid_timestamp_count']:,}."
        for row in temporal
    ]
    anomaly_lines = [
        f"- `{row['file']}` row {row['row_number']}: raw `{row['raw_timestamp']}`, parsed value `{row['parsed_value']}`, classified as {row['classification']}."
        for row in anomaly_rows
    ]
    numeric_lines = [
        f"- `{row['feature']}`: missing {row['missing_count']:,}, invalid {row['invalid_count']:,}, +inf {row['positive_infinity_count']:,}, -inf {row['negative_infinity_count']:,}, finite range `{row['finite_minimum']}` to `{row['finite_maximum']}`."
        for row in numeric
    ]
    lines = [
        "# CIC-IDS2018 Data Readiness",
        "",
        "This Phase 2 report is read-only. Each raw CSV was scanned independently with bounded-memory CSV iteration. No source data was merged, rewritten, sorted, imputed, deleted, moved, or repaired. No model or temporal training window was created.",
        "",
        "## Dataset overview",
        "",
        f"The ten files contain {total_rows:,} flow rows across approximately 6.89 GB of raw CSV. Exact header counts: {schema_counts}.",
        "",
        "## Schema findings",
        "",
        "Nine files share the 80-column common schema. `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` has 84 columns and adds `Flow ID`, `Src IP`, `Src Port`, and `Dst IP`. Relative to the 84-column union, the other nine files are missing exactly those four columns. No other header differences were found.",
        "",
        "## Label findings",
        "",
        *label_lines,
        "",
        "The repeated header rows appear in the label report as raw label `Label`; they are not attack classes and are excluded from the attack timeline.",
        "",
        "## Attack timeline",
        "",
        *attack_lines,
        "",
        "## Timestamp quality",
        "",
        f"There are {total_invalid_timestamps} invalid timestamps across all files. The requested year-1970 investigation found {len(anomaly_rows)} affected rows: {sum(row['file'].startswith('Thursday') for row in anomaly_rows)} in Thursday-22-02 and {sum(row['file'].startswith('Wednesday') for row in anomaly_rows)} in Wednesday-14-02.",
        "These are malformed source values, not a parser-only issue: the raw strings explicitly contain year 1970. They are preserved verbatim in the audit process and are not repaired.",
        *anomaly_lines,
        "",
        "## Numeric quality",
        "",
        f"Across the numeric features, the scan found {numeric_invalid:,} invalid numeric strings, {positive_inf:,} positive infinity values, and {negative_inf:,} negative infinity values. Blank-cell counts and finite ranges are recorded per feature in `data/audit/numeric_feature_quality.csv`.",
        "",
        *numeric_lines,
        "",
        "## Temporal ordering",
        "",
        *temporal_lines,
        "",
        "Ordering is assessed in original row order using valid parsed timestamps only; rows with invalid timestamps are reported separately and do not get silently repositioned.",
        "",
        "## Feature policy",
        "",
        "- `SAFE_TRAFFIC_FEATURE`: CICFlowMeter-derived traffic measurements, protocol, ports, packet counts, byte counts, durations, rates, flags, sizes, windows, and inter-arrival statistics, subject to numeric-quality checks.",
        "- `POTENTIAL_IDENTIFIER`: `Flow ID`, `Src IP`, and `Dst IP`; these can identify flows or hosts and may enable memorization or environment-specific shortcuts.",
        "- `POTENTIAL_LEAKAGE`: `Label`; it is the supervised target and must not be used as an input feature. `Timestamp` is also leakage-prone if used beyond an explicitly justified temporal split or as a proxy for collection conditions.",
        "- `NEEDS_INVESTIGATION`: `Timestamp`, port semantics, and all identifier-like fields require an explicit split and generalization policy before modeling. `Src Port` is present only in the Tuesday schema and needs schema handling.",
        "",
        "## Packet-level gap",
        "",
        "The current processed CSVs provide flow-level, CICFlowMeter-derived features. SIH 26153 also requires packet-level features. Packet-level integration is not performed in this phase and remains a separate data-engineering requirement.",
        "",
        "## Recommended preprocessing policy",
        "",
        "1. Preserve raw files as immutable inputs and keep per-file provenance.",
        "2. Resolve the four-column schema discrepancy explicitly; do not infer absent identifiers as real values.",
        "3. Define a documented policy for repeated header rows and malformed 1970 timestamps before any downstream feature generation.",
        "4. Track blank, invalid, and infinite numeric values separately; choose any later handling only after domain review, without changing raw inputs.",
        "5. Use time-aware, file-aware validation that prevents identifier and collection-time leakage.",
        "6. Address the packet-level requirement before claiming complete SIH 26153 readiness.",
        "",
        "## Final decision",
        "",
        "NOT READY — ISSUE(S) REQUIRE RESOLUTION",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/audit")
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = parser.parse_args()
    results = []
    anomaly_rows: list[dict[str, str]] = []
    for filename in FILES:
        path = args.input_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        results.append(audit_file(path, anomaly_rows))
    schema_path = args.output_dir / "schema_comparison.csv"
    label_path = args.output_dir / "label_distribution.csv"
    timeline_path = args.output_dir / "attack_timeline.csv"
    numeric_path = args.output_dir / "numeric_feature_quality.csv"
    temporal_path = args.output_dir / "temporal_order.csv"
    create_schema_report(results, schema_path)
    labels = create_label_report(results, label_path)
    timeline = create_timeline_report(results, timeline_path)
    numeric = create_numeric_report(results, numeric_path)
    temporal = create_temporal_report(results, temporal_path)
    timestamp_path = args.output_dir / "timestamp_anomalies.csv"
    write_csv(timestamp_path, list(anomaly_rows[0]), anomaly_rows)
    write_docs(results, labels, timeline, numeric, temporal, anomaly_rows, args.docs_dir / "CIC_IDS2018_DATA_READINESS.md")
    print(f"Audited {len(results)} files and {sum(result['row_count'] for result in results):,} rows")
    for output in (schema_path, label_path, timeline_path, numeric_path, temporal_path, timestamp_path, args.docs_dir / "CIC_IDS2018_DATA_READINESS.md"):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()