"""Phase 6B vector World Model data interface.

Read-only interface over the existing Phase 3.5 canonical state CSVs
(`data/windows/*.csv`). For every existing `forecast_sample_eligible`
row t, builds a rollout pair:

    X = S(t-5) ... S(t)       shape [6, 157]   (history, including current)
    Y = S(t+1) ... S(t+6)     shape [6, 157]   (future target states)

using the exact eligibility and indexing rules already used by
`phase4_baseline.read_samples`. This module does not modify, rebuild,
or regenerate Phase 3.5, does not touch Phase 4/5 code or results, and
does not train anything -- it only reads `data/windows/*.csv` and
exposes/validates (X, Y) tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from phase4_baseline import FORECAST_HORIZON_WINDOWS, HISTORY_WINDOWS, META_COLUMNS

EXPECTED_FEATURE_COUNT = 157
EXPECTED_SPLIT_COUNTS = {"train": 29315, "validation": 6195, "test": 6195}
WINDOW_SECONDS = 10
SPAN_LENGTH = HISTORY_WINDOWS + FORECAST_HORIZON_WINDOWS  # 12 timestamps: t-5..t, t+1..t+6
CURRENT_INDEX_IN_SPAN = HISTORY_WINDOWS - 1  # index of t within the 12-timestamp span


@dataclass
class WorldModelSampleSet:
    """Model-ready tensors plus audit-only metadata (never fed to a model)."""

    X: np.ndarray  # [N, 6, 157] float32 -- S(t-5)...S(t)
    Y: np.ndarray  # [N, 6, 157] float32 -- S(t+1)...S(t+6)
    label: np.ndarray  # [N] int64 -- future_attack_within_horizon at row t (same Phase 4/5 target)
    future_attack_step: np.ndarray  # [N] int64 -- earliest future step (1-6) with current_attack==1, else 0
    source_file: np.ndarray  # [N] object/str -- source-day file of row t
    window_start: np.ndarray  # [N] object/str -- window_start of row t
    window_start_sequence: np.ndarray  # [N, 12] object/str -- t-5..t, t+1..t+6 (audit only)


def read_world_model_samples(
    windows_dir: Path,
) -> tuple[dict[str, WorldModelSampleSet], list[str], list[str]]:
    """Build (X, Y) rollout samples for every eligible row in data/windows/*.csv."""
    partitions = sorted(windows_dir.glob("*.csv"))
    if len(partitions) != 10:
        raise ValueError(f"Expected 10 temporal partitions, found {len(partitions)}")

    collected: dict[str, list[tuple]] = {"train": [], "validation": [], "test": []}
    feature_columns: list[str] | None = None
    source_files: list[str] = []

    for partition in partitions:
        with partition.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            current_features = [column for column in columns if column not in META_COLUMNS]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise ValueError(f"Feature schema differs in {partition.name}")
            rows = list(reader)
        source_files.append(partition.name)

        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] != "1":
                continue
            if index < HISTORY_WINDOWS - 1 or index + FORECAST_HORIZON_WINDOWS >= len(rows):
                raise ValueError(f"Invalid eligible index {index} in {partition.name}")

            split = row["split"]
            if split not in collected:
                raise ValueError(f"Unexpected split {split!r} in {partition.name}")

            history_rows = rows[index - HISTORY_WINDOWS + 1 : index + 1]
            future_rows = rows[index + 1 : index + 1 + FORECAST_HORIZON_WINDOWS]
            span_rows = history_rows + future_rows

            span_splits = {span_row["split"] for span_row in span_rows}
            if span_splits != {split}:
                raise ValueError(
                    f"Split boundary crossed for eligible sample at "
                    f"{partition.name}:{index} (splits found: {sorted(span_splits)})"
                )

            X = np.asarray(
                [[float(history_row[column]) for column in feature_columns] for history_row in history_rows],
                dtype=np.float32,
            )
            Y = np.asarray(
                [[float(future_row[column]) for column in feature_columns] for future_row in future_rows],
                dtype=np.float32,
            )
            window_start_sequence = [span_row["window_start"] for span_row in span_rows]
            label = int(row["future_attack_within_horizon"])
            future_attack_step = 0
            for step, future_row in enumerate(future_rows, start=1):
                if future_row["current_attack"] == "1":
                    future_attack_step = step
                    break

            collected[split].append(
                (X, Y, label, future_attack_step, partition.name, row["window_start"], window_start_sequence)
            )

    if feature_columns is None or len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} model features, found {len(feature_columns or [])}")

    output: dict[str, WorldModelSampleSet] = {}
    for split, items in collected.items():
        if items:
            X = np.stack([item[0] for item in items])
            Y = np.stack([item[1] for item in items])
        else:
            X = np.zeros((0, HISTORY_WINDOWS, EXPECTED_FEATURE_COUNT), dtype=np.float32)
            Y = np.zeros((0, FORECAST_HORIZON_WINDOWS, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        output[split] = WorldModelSampleSet(
            X=X,
            Y=Y,
            label=np.asarray([item[2] for item in items], dtype=np.int64),
            future_attack_step=np.asarray([item[3] for item in items], dtype=np.int64),
            source_file=np.asarray([item[4] for item in items], dtype=object),
            window_start=np.asarray([item[5] for item in items], dtype=object),
            window_start_sequence=np.asarray([item[6] for item in items], dtype=object),
        )

    return output, feature_columns, source_files


def validate_world_model_samples(
    samples: dict[str, WorldModelSampleSet],
    feature_columns: list[str],
) -> dict[str, object]:
    """Independent validation pass: counts, shapes, finiteness, temporal alignment, leakage."""
    issues: list[str] = []
    report: dict[str, object] = {"splits": {}}

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        issues.append(f"feature_count={len(feature_columns)} != {EXPECTED_FEATURE_COUNT}")
    if any(column in META_COLUMNS for column in feature_columns):
        issues.append("feature_columns contains a metadata/label column")

    seen_keys: dict[tuple[str, str], str] = {}
    cross_split_leakage_pairs = 0
    total_samples = 0

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        sample_set = samples[split]
        n = sample_set.X.shape[0]
        total_samples += n
        split_report: dict[str, object] = {"count": n}

        if n != expected_count:
            issues.append(f"{split}: count={n} != expected {expected_count}")

        if sample_set.X.shape != (n, HISTORY_WINDOWS, EXPECTED_FEATURE_COUNT):
            issues.append(f"{split}: X shape {sample_set.X.shape} != ({n}, {HISTORY_WINDOWS}, {EXPECTED_FEATURE_COUNT})")
        if sample_set.Y.shape != (n, FORECAST_HORIZON_WINDOWS, EXPECTED_FEATURE_COUNT):
            issues.append(f"{split}: Y shape {sample_set.Y.shape} != ({n}, {FORECAST_HORIZON_WINDOWS}, {EXPECTED_FEATURE_COUNT})")
        split_report["X_shape"] = list(sample_set.X.shape)
        split_report["Y_shape"] = list(sample_set.Y.shape)

        x_finite = bool(np.isfinite(sample_set.X).all()) if n else True
        y_finite = bool(np.isfinite(sample_set.Y).all()) if n else True
        split_report["X_finite"] = x_finite
        split_report["Y_finite"] = y_finite
        if not x_finite:
            issues.append(f"{split}: X contains non-finite values")
        if not y_finite:
            issues.append(f"{split}: Y contains non-finite values")

        alignment_failures = 0
        for row_index in range(n):
            sequence = sample_set.window_start_sequence[row_index]
            if len(sequence) != SPAN_LENGTH:
                alignment_failures += 1
                continue
            if sequence[CURRENT_INDEX_IN_SPAN] != sample_set.window_start[row_index]:
                alignment_failures += 1
                continue
            try:
                timestamps = [datetime.fromisoformat(value) for value in sequence]
            except ValueError:
                alignment_failures += 1
                continue
            gaps_ok = all(
                (timestamps[i + 1] - timestamps[i]) == timedelta(seconds=WINDOW_SECONDS)
                for i in range(SPAN_LENGTH - 1)
            )
            if not gaps_ok:
                alignment_failures += 1

            key = (str(sample_set.source_file[row_index]), str(sample_set.window_start[row_index]))
            if key in seen_keys and seen_keys[key] != split:
                cross_split_leakage_pairs += 1
                issues.append(f"leakage: {key} appears in both {seen_keys[key]} and {split}")
            seen_keys[key] = split

        split_report["temporal_alignment_failures"] = alignment_failures
        if alignment_failures:
            issues.append(f"{split}: {alignment_failures} samples failed 10s contiguous alignment check")

        duplicate_count = 0
        if n:
            flat = sample_set.X.reshape(n, -1)
            _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
            duplicate_count = int(n - counts.shape[0])
        split_report["duplicate_X_sequences_present"] = duplicate_count

        if n:
            recomputed_label = (sample_set.future_attack_step > 0).astype(np.int64)
            label_mismatches = int((recomputed_label != sample_set.label).sum())
        else:
            label_mismatches = 0
        split_report["label_recomputation_mismatches"] = label_mismatches
        if label_mismatches:
            issues.append(
                f"{split}: {label_mismatches} samples where future_attack_within_horizon disagrees with "
                "independently recomputed any-current_attack-in-future-rows label"
            )
        split_report["positive"] = int(sample_set.label.sum())
        split_report["negative"] = int((sample_set.label == 0).sum())

        report["splits"][split] = split_report

    if total_samples != sum(EXPECTED_SPLIT_COUNTS.values()):
        issues.append(f"total_samples={total_samples} != {sum(EXPECTED_SPLIT_COUNTS.values())}")

    report["total_samples"] = total_samples
    report["cross_split_leakage_pairs"] = cross_split_leakage_pairs
    report["issues"] = issues
    report["status"] = "PASS" if not issues else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/windows",
    )
    args = parser.parse_args()

    samples, feature_columns, source_files = read_world_model_samples(args.windows_dir)
    report = validate_world_model_samples(samples, feature_columns)
    report["source_files"] = source_files
    report["feature_count"] = len(feature_columns)
    report["history_windows"] = HISTORY_WINDOWS
    report["forecast_horizon_windows"] = FORECAST_HORIZON_WINDOWS

    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
