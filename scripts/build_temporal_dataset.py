"""Build per-source-day temporal forecasting states without training a model."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


WINDOW_SECONDS = 10
HISTORY_WINDOWS = 6
FORECAST_HORIZON_WINDOWS = 6
CHUNK_ROWS = 100_000
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
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
IDENTIFIER_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
KNOWN_1970_ROWS = {
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246435),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246436),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246437),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246438),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246439),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246440),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246441),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 246717),
    ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", 248316),
    ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 410958),
    ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 410959),
    ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 410960),
    ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 410961),
    ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 412186),
}
LABEL_NORMALIZATION = {
    "Infilteration": "Infiltration",
    "DDOS attack-HOIC": "DDoS attack-HOIC",
    "DDOS attack-LOIC-UDP": "DDoS attack-LOIC-UDP",
}


def parse_timestamp(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw.strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def normalize_label(raw: str) -> str:
    value = raw.strip()
    return LABEL_NORMALIZATION.get(value, value)


def parse_number(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return math.nan
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def window_start(timestamp: datetime) -> datetime:
    return timestamp.replace(second=timestamp.second - timestamp.second % WINDOW_SECONDS, microsecond=0)


def write_sorted_chunk(path: Path, header: list[str], rows: list[tuple[datetime, list[str]]]) -> None:
    rows.sort(key=lambda item: item[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for timestamp, row in rows:
            writer.writerow([timestamp.isoformat(sep=" "), *row])


def merge_sorted_chunks(paths: list[Path], header: list[str]):
    streams = [path.open("r", encoding="utf-8", newline="") for path in paths]
    readers = [csv.reader(stream) for stream in streams]
    heap: list[tuple[datetime, int, list[str]]] = []
    try:
        for index, reader in enumerate(readers):
            row = next(reader, None)
            if row:
                heapq.heappush(heap, (datetime.fromisoformat(row[0]), index, row[1:]))
        while heap:
            timestamp, index, row = heapq.heappop(heap)
            yield timestamp, row
            next_row = next(readers[index], None)
            if next_row:
                heapq.heappush(heap, (datetime.fromisoformat(next_row[0]), index, next_row[1:]))
    finally:
        for stream in streams:
            stream.close()


def numeric_columns(header: list[str]) -> list[str]:
    return [column for column in header if column not in IDENTIFIER_COLUMNS and column != "Src Port"]


def aggregate_window(
    window: datetime,
    rows: list[tuple[list[str], int, str]],
    columns: list[str],
    positions: dict[str, int],
) -> dict[str, object]:
    values: dict[str, list[float]] = defaultdict(list)
    label_counts: dict[str, int] = defaultdict(int)
    for row, attack, label in rows:
        label_counts[label] += 1
        for column in columns:
            parsed = parse_number(row[positions[column]])
            if parsed is not None and not math.isnan(parsed):
                values[column].append(parsed)

    state: dict[str, object] = {
        "window_start": window.isoformat(sep=" "),
        "flow_count": len(rows),
        "current_attack": int(any(attack for _, attack, _ in rows)),
        "current_attack_types": ";".join(sorted(label for label in label_counts if label not in {"", "Benign", "Label"})),
    }
    for column in columns:
        column_key = column.lower().replace(" ", "_").replace("/", "_per_")
        column_values = values[column]
        state[f"{column_key}__sum"] = sum(column_values) if column_values else 0.0
        state[f"{column_key}__mean"] = sum(column_values) / len(column_values) if column_values else 0.0
    return state


def build_file(path: Path, output_dir: Path, audit_rows: list[dict[str, object]]) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        header = [value.strip() for value in next(reader)]
        positions = {column: index for index, column in enumerate(header)}
        common_columns = numeric_columns(header)
        if "Timestamp" not in positions or "Label" not in positions:
            raise ValueError(f"{path.name} lacks Timestamp or Label")

        valid_chunks: list[Path] = []
        chunk: list[tuple[datetime, list[str]]] = []
        excluded_counts: defaultdict[str, int] = defaultdict(int)
        numeric_affected = 0
        first_valid: datetime | None = None
        last_valid: datetime | None = None
        with tempfile.TemporaryDirectory(prefix="netoracle-sort-") as temp_dir:
            temp_path = Path(temp_dir)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    excluded_counts["malformed_width"] += 1
                    audit_rows.append({"source_file": path.name, "row_number": row_number, "reason": "malformed_width", "raw_timestamp": row[positions["Timestamp"]] if len(row) > positions["Timestamp"] else "", "raw_label": row[positions["Label"]] if len(row) > positions["Label"] else ""})
                    continue
                raw_timestamp = row[positions["Timestamp"]].strip()
                raw_label = row[positions["Label"]].strip()
                if raw_label == "Label":
                    excluded_counts["repeated_header"] += 1
                    audit_rows.append({"source_file": path.name, "row_number": row_number, "reason": "repeated_header", "raw_timestamp": raw_timestamp, "raw_label": raw_label})
                    continue
                timestamp = parse_timestamp(raw_timestamp)
                if timestamp is None:
                    excluded_counts["invalid_timestamp"] += 1
                    audit_rows.append({"source_file": path.name, "row_number": row_number, "reason": "invalid_timestamp", "raw_timestamp": raw_timestamp, "raw_label": raw_label})
                    continue
                if (path.name, row_number) in KNOWN_1970_ROWS:
                    excluded_counts["known_1970_timestamp"] += 1
                    audit_rows.append({"source_file": path.name, "row_number": row_number, "reason": "known_1970_timestamp", "raw_timestamp": raw_timestamp, "raw_label": raw_label})
                    continue
                first_valid = timestamp if first_valid is None else min(first_valid, timestamp)
                last_valid = timestamp if last_valid is None else max(last_valid, timestamp)
                for column in common_columns:
                    parsed = parse_number(row[positions[column]])
                    if parsed is None or math.isnan(parsed):
                        numeric_affected += 1
                chunk.append((timestamp, row))
                if len(chunk) >= CHUNK_ROWS:
                    chunk_path = temp_path / f"chunk-{len(valid_chunks):05d}.csv"
                    write_sorted_chunk(chunk_path, header, chunk)
                    valid_chunks.append(chunk_path)
                    chunk = []
            if chunk:
                chunk_path = temp_path / f"chunk-{len(valid_chunks):05d}.csv"
                write_sorted_chunk(chunk_path, header, chunk)
                valid_chunks.append(chunk_path)

            states_by_window: dict[datetime, list[tuple[list[str], int, str]]] = defaultdict(list)
            for timestamp, row in merge_sorted_chunks(valid_chunks, header):
                label = normalize_label(row[positions["Label"]])
                attack = int(label not in {"", "Benign", "Label"})
                states_by_window[window_start(timestamp)].append((row, attack, label))

        if not states_by_window:
            raise ValueError(f"No valid rows remain in {path.name}")
        first_window = min(states_by_window)
        last_window = max(states_by_window)
        all_windows: list[datetime] = []
        current = first_window
        while current <= last_window:
            all_windows.append(current)
            current += timedelta(seconds=WINDOW_SECONDS)

        states = []
        for window in all_windows:
            state = aggregate_window(window, states_by_window.get(window, []), common_columns, positions)
            states.append(state)
        for index, state in enumerate(states):
            future_states = states[index + 1:index + 1 + FORECAST_HORIZON_WINDOWS]
            future_attacks = [int(item["current_attack"]) for item in future_states]
            state["future_attack_within_horizon"] = int(any(future_attacks))
            state["future_attack_types"] = ";".join(sorted({label for item in future_states for label in str(item["current_attack_types"]).split(";") if label}))
            state["history_available"] = int(index >= HISTORY_WINDOWS - 1)
            state["history_window_count"] = min(index + 1, HISTORY_WINDOWS)
            state["source_file"] = path.name
            state["split"] = _split_for_index(index, len(states))
            state["forecast_sample_eligible"] = int(
                index >= HISTORY_WINDOWS - 1
                and index + FORECAST_HORIZON_WINDOWS < len(states)
                and len({_split_for_index(position, len(states)) for position in range(index - HISTORY_WINDOWS + 1, index + FORECAST_HORIZON_WINDOWS + 1)}) == 1
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{path.stem}.csv"
        write_window_partition(output_path, states)
        feature_columns = [column for column in states[0] if column not in {"window_start", "source_file", "split", "current_attack", "current_attack_types", "future_attack_within_horizon", "future_attack_types", "history_available", "history_window_count", "forecast_sample_eligible"}]
        eligible_states = [state for state in states if state["forecast_sample_eligible"]]
        incomplete_horizon_windows = min(FORECAST_HORIZON_WINDOWS, len(states))
        return {
            "source_file": path.name,
            "window_count": len(states),
            "benign_current_windows": sum(int(state["current_attack"] == 0) for state in states),
            "attack_current_windows": sum(int(state["current_attack"] == 1) for state in states),
            "future_attack_positive_windows": sum(int(state["future_attack_within_horizon"]) for state in states),
            "future_attack_negative_windows": sum(int(not state["future_attack_within_horizon"]) for state in states),
            "earliest_window": first_window.isoformat(sep=" "),
            "latest_window": last_window.isoformat(sep=" "),
            "excluded_rows": sum(excluded_counts.values()),
            "excluded_repeated_headers": excluded_counts["repeated_header"],
            "excluded_known_1970": excluded_counts["known_1970_timestamp"],
            "numeric_affected_values": numeric_affected,
            "feature_count": len(feature_columns),
            "incomplete_future_horizon_windows": incomplete_horizon_windows,
            "valid_forecasting_samples": len(eligible_states),
            "train_windows": sum(state["forecast_sample_eligible"] and state["split"] == "train" for state in states),
            "validation_windows": sum(state["forecast_sample_eligible"] and state["split"] == "validation" for state in states),
            "test_windows": sum(state["forecast_sample_eligible"] and state["split"] == "test" for state in states),
        }


def _split_for_index(index: int, window_count: int) -> str:
    if index < window_count * 0.70:
        return "train"
    if index < window_count * 0.85:
        return "validation"
    return "test"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_log(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["source_file", "row_number", "reason", "raw_timestamp", "raw_label"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_window_partition(path: Path, states: list[dict[str, object]]) -> None:
    fieldnames = list(states[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(states)


def write_design_doc(path: Path, summary: list[dict[str, object]], feature_count: int) -> None:
    total_windows = sum(int(row["window_count"]) for row in summary)
    total_positive = sum(int(row["future_attack_positive_windows"]) for row in summary)
    total_negative = sum(int(row["future_attack_negative_windows"]) for row in summary)
    total_train = sum(int(row["train_windows"]) for row in summary)
    total_validation = sum(int(row["validation_windows"]) for row in summary)
    total_test = sum(int(row["test_windows"]) for row in summary)
    total_valid_samples = sum(int(row["valid_forecasting_samples"]) for row in summary)
    total_incomplete = sum(int(row["incomplete_future_horizon_windows"]) for row in summary)
    feature_names = "`<source-column>__sum` and `<source-column>__mean` for every common numeric CICFlowMeter column, excluding `Label`, raw `Timestamp`, `Flow ID`, `Src IP`, `Dst IP`, and Tuesday-only `Src Port`."
    text = f"""# Temporal Dataset Design

This artifact describes the read-only Phase 3 dataset build. Raw CSV files remain unchanged, source days remain separate, and no model training or model-weight generation occurs here. Each source-day partition is written as a compact CSV because the selected environment does not provide a Parquet writer; no new dependency is added.

## 1. State representation

Each source file is reconstructed independently in timestamp order. Every 10-second state contains `flow_count`, current-window attack metadata, and sum/mean aggregates for the common numeric CICFlowMeter columns. The resulting state vector has {feature_count} numeric features. The final feature policy is {feature_names}

## 2. Cleaning policy

Invalid numeric strings and `+inf`/`-inf` are represented as missing during aggregation. Invalid values do not remove an otherwise valid flow. Aggregates use only finite observations; an empty aggregate is represented as 0.0 as a structural no-observation value, not as learned imputation. Affected-value counts are recorded in the summary. No training statistics are learned in this phase.

## 3. Timestamp policy

Repeated header rows, the 14 known year-1970 rows, other unparseable timestamps, and malformed-width rows are excluded from modeling and logged in `data/audit/temporal_preprocessing_log.csv`. Valid flows are externally sorted within each source file. Raw timestamps are not model inputs.

## 4. Label policy

`Infilteration` is normalized to `Infiltration`; known capitalization variants of DDoS labels are normalized. Raw labels remain available in the preprocessing audit. `Benign` maps to current attack indicator 0. Any other known non-header label maps to 1. Repeated `Label` rows are excluded and never become attack classes.

## 5. Window size

`WINDOW_SECONDS = {WINDOW_SECONDS}`. Windows are anchored to ten-second boundaries and are created independently per source day, including empty traffic windows between the valid minimum and maximum timestamps.

## 6. History length

`HISTORY_WINDOWS = {HISTORY_WINDOWS}`. `history_available` marks rows with enough preceding states for a later sequence consumer; no sequences are trained or materialized here.

## 7. Forecast horizon

`FORECAST_HORIZON_WINDOWS = {FORECAST_HORIZON_WINDOWS}`, representing the next 60 seconds.

## 8. Future-label definition

For state S(t), `future_attack_within_horizon` is 1 when any current attack indicator in S(t+1) through S(t+6) is 1. S(t) is explicitly excluded. Future attack types are retained where available. The final {total_incomplete} terminal windows with fewer than six future states are retained for state continuity but marked `forecast_sample_eligible = 0` and excluded from forecasting samples.

## 9. Split strategy

Each source-day partition is split chronologically by window position: first 70% train, next 15% validation, final 15% test. A forecasting sample is eligible only when its six history states, current state, and six future states all belong to the same split. This removes split guard windows and prevents histories or targets from crossing boundaries. No individual flows are randomly split and no source-day sequence crosses a partition boundary. Totals: {total_valid_samples} valid samples; train {total_train}, validation {total_validation}, test {total_test}.

## 10. Leakage prevention

`Label`, raw `Timestamp`, `Flow ID`, `Src IP`, and `Dst IP` are excluded from numeric state features. Current attack metadata is retained only for audit/evaluation and is not part of the feature vector. Future labels use only strictly later windows. Tuesday-only identifiers are not required by the common feature schema.

## 11. Final feature list

The exact generated feature columns are the common numeric source columns with `__sum` and `__mean` suffixes, plus `flow_count`. The state metadata columns are `current_attack`, `current_attack_types`, `future_attack_within_horizon`, `future_attack_types`, `history_available`, `history_window_count`, `forecast_sample_eligible`, `source_file`, `window_start`, and `split`; metadata is not a model input.

## 12. Known limitations

- The source CSVs are flow-level CICFlowMeter data, not packet-level SIH 26153 data.
- The source files are heavily out of order and are reconstructed with external chunk sorting.
- Empty-window zero values are structural aggregation values, not imputation.
- The final six windows of each source partition have fewer than six future observations and should be filtered or treated explicitly by a later training consumer.
- Samples near chronological split boundaries are intentionally excluded when their history or future span would cross a split.
- Imputation/scaling parameters must be fit on eligible training samples only, then applied unchanged to validation and test.
- No learned imputation, scaling, threshold tuning, model training, or temporal-window model integration is performed.

## Output totals

- Total temporal windows: {total_windows}
- Valid forecasting samples: {total_valid_samples}
- Incomplete future-horizon windows removed from forecasting samples: {total_incomplete}
- Future-positive windows: {total_positive}
- Future-negative windows: {total_negative}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/audit")
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = parser.parse_args()
    args.windows_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for filename in FILES:
        summary.append(build_file(args.input_dir / filename, args.windows_dir, audit_rows))
    write_summary(args.audit_dir / "temporal_dataset_summary.csv", summary)
    write_log(args.audit_dir / "temporal_preprocessing_log.csv", audit_rows)
    write_design_doc(args.docs_dir / "TEMPORAL_DATASET_DESIGN.md", summary, int(summary[0]["feature_count"]))
    print(json.dumps({
        "files": len(summary),
        "windows": sum(int(row["window_count"]) for row in summary),
        "features": int(summary[0]["feature_count"]),
        "future_positive": sum(int(row["future_attack_positive_windows"]) for row in summary),
        "future_negative": sum(int(row["future_attack_negative_windows"]) for row in summary),
        "excluded_rows": len(audit_rows),
        "train": sum(int(row["train_windows"]) for row in summary),
        "validation": sum(int(row["validation_windows"]) for row in summary),
        "test": sum(int(row["test_windows"]) for row in summary),
    }, indent=2))


if __name__ == "__main__":
    main()