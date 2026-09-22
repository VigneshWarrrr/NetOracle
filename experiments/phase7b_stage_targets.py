"""Phase 7B: independent MITRE stage target construction.

Reads `data/windows/*.csv` directly (read-only). Deliberately does NOT
import or modify `world_model_dataset.py` -- this file is fully standalone,
replicating only the canonical eligibility/indexing rule (imported read-only
from `phase4_baseline.py`, the same source `world_model_dataset.py` itself
uses) so that its per-sample ordering is independently derivable and
verifiable, not copy-pasted.

Produces, per eligible Phase 3.5 sample t:
  - six PER-STEP MITRE stage targets for t+1..t+6, each derived from that
    future row's OWN `current_attack_types` column
  - one AGGREGATE "headline" target, derived from `future_attack_types`
    (the union over t+1..t+6, already computed by build_temporal_dataset.py)

Both paths route through the corrected, Phase-7A-tested
`forecasting.mitre_mapping.MitreMapper.from_label_set()`. This module
produces TARGET/LABEL arrays only. Nothing here is ever passed to a model's
forward() -- see phase7b_train_stage_head.py's leakage-audit tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # NetOracle/, for `forecasting`

from phase4_baseline import FORECAST_HORIZON_WINDOWS, HISTORY_WINDOWS
from forecasting.mitre_mapping import MitreMapper, MitreStage

# The six MitreStage values with real coverage in this dataset, per the
# Phase 7B inspection report (the other 8 MitreStage values never occur for
# any real CIC-IDS2018 label and must never be produced as a target). Order
# is fixed by MitreStage.value for a stable, deterministic class-index space.
COVERED_STAGES: list[MitreStage] = sorted(
    [
        MitreStage.BENIGN,
        MitreStage.INITIAL_ACCESS,
        MitreStage.CREDENTIAL_ACCESS,
        MitreStage.LATERAL_MOVEMENT,
        MitreStage.COMMAND_AND_CONTROL,
        MitreStage.IMPACT,
    ],
    key=lambda stage: stage.value,
)
NUM_STAGE_CLASSES = len(COVERED_STAGES)
STAGE_TO_CLASS_INDEX = {stage: index for index, stage in enumerate(COVERED_STAGES)}
CLASS_INDEX_TO_STAGE = {index: stage for stage, index in STAGE_TO_CLASS_INDEX.items()}

ABSENT_STAGES = [stage for stage in MitreStage if stage not in STAGE_TO_CLASS_INDEX]

EXPECTED_SPLIT_COUNTS = {"train": 29315, "validation": 6195, "test": 6195}

# Aggregate stage distribution exactly as computed and reported in the
# Phase 7B inspection report (section 5). Any deviation is a hard-stop
# condition per the Phase 7B implementation instructions.
EXPECTED_AGGREGATE_DISTRIBUTION = {
    "train": {
        "BENIGN": 25744, "COMMAND_AND_CONTROL": 1044, "IMPACT": 262,
        "INITIAL_ACCESS": 766, "CREDENTIAL_ACCESS": 559, "LATERAL_MOVEMENT": 940,
    },
    "validation": {
        "BENIGN": 3603, "COMMAND_AND_CONTROL": 352, "IMPACT": 810,
        "CREDENTIAL_ACCESS": 945, "LATERAL_MOVEMENT": 485,
    },
    "test": {
        "BENIGN": 4432, "COMMAND_AND_CONTROL": 637, "IMPACT": 406,
        "CREDENTIAL_ACCESS": 408, "LATERAL_MOVEMENT": 312,
    },
}
# INITIAL_ACCESS per-step counts by split, exactly as computed and reported
# in the Phase 7B inspection report. validation/test coverage is (near-)zero
# by construction of the real data, not a bug in this reader.
EXPECTED_INITIAL_ACCESS_PER_STEP = {
    "train": [174, 173, 172, 172, 172, 172],
    "validation": [1, 1, 1, 1, 1, 1],
    "test": [0, 0, 0, 0, 0, 0],
}


def stage_to_class_index(stage: MitreStage) -> int:
    if stage not in STAGE_TO_CLASS_INDEX:
        raise ValueError(
            f"{stage.name} is not one of the {NUM_STAGE_CLASSES} MitreStage values with real "
            "coverage in this dataset; it must never be produced as a stage target."
        )
    return STAGE_TO_CLASS_INDEX[stage]


@dataclass
class StageTargetSet:
    per_step_class_index: np.ndarray   # [N, 6] int64, values in 0..NUM_STAGE_CLASSES-1
    per_step_stage_name: np.ndarray    # [N, 6] object/str
    aggregate_class_index: np.ndarray  # [N] int64
    aggregate_stage_name: np.ndarray   # [N] object/str
    source_file: np.ndarray            # [N] object/str -- row t's source file
    window_start: np.ndarray           # [N] object/str -- row t's window_start


def read_stage_targets(windows_dir: Path) -> dict[str, StageTargetSet]:
    partitions = sorted(windows_dir.glob("*.csv"))
    if len(partitions) != 10:
        raise ValueError(f"Expected 10 temporal partitions, found {len(partitions)}")

    collected: dict[str, list[tuple]] = {"train": [], "validation": [], "test": []}

    for partition in partitions:
        with partition.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))

        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] != "1":
                continue
            if index < HISTORY_WINDOWS - 1 or index + FORECAST_HORIZON_WINDOWS >= len(rows):
                raise ValueError(f"Invalid eligible index {index} in {partition.name}")

            split = row["split"]
            if split not in collected:
                raise ValueError(f"Unexpected split {split!r} in {partition.name}")

            future_rows = rows[index + 1 : index + 1 + FORECAST_HORIZON_WINDOWS]

            span_splits = {future_row["split"] for future_row in future_rows} | {split}
            if span_splits != {split}:
                raise ValueError(
                    f"Split boundary crossed while building stage targets for "
                    f"{partition.name}:{index} (splits found: {sorted(span_splits)})"
                )

            per_step_stages = [
                MitreMapper.from_label_set(future_row["current_attack_types"]).stage
                if future_row["current_attack_types"]
                else MitreStage.BENIGN
                for future_row in future_rows
            ]
            per_step_indices = [stage_to_class_index(stage) for stage in per_step_stages]

            aggregate_stage = MitreMapper.from_label_set(row["future_attack_types"]).stage
            aggregate_index = stage_to_class_index(aggregate_stage)

            collected[split].append(
                (
                    per_step_indices,
                    [stage.name for stage in per_step_stages],
                    aggregate_index,
                    aggregate_stage.name,
                    partition.name,
                    row["window_start"],
                )
            )

    output: dict[str, StageTargetSet] = {}
    for split, items in collected.items():
        if items:
            output[split] = StageTargetSet(
                per_step_class_index=np.asarray([item[0] for item in items], dtype=np.int64),
                per_step_stage_name=np.asarray([item[1] for item in items], dtype=object),
                aggregate_class_index=np.asarray([item[2] for item in items], dtype=np.int64),
                aggregate_stage_name=np.asarray([item[3] for item in items], dtype=object),
                source_file=np.asarray([item[4] for item in items], dtype=object),
                window_start=np.asarray([item[5] for item in items], dtype=object),
            )
        else:
            output[split] = StageTargetSet(
                per_step_class_index=np.zeros((0, FORECAST_HORIZON_WINDOWS), dtype=np.int64),
                per_step_stage_name=np.zeros((0, FORECAST_HORIZON_WINDOWS), dtype=object),
                aggregate_class_index=np.zeros((0,), dtype=np.int64),
                aggregate_stage_name=np.zeros((0,), dtype=object),
                source_file=np.zeros((0,), dtype=object),
                window_start=np.zeros((0,), dtype=object),
            )
    return output


def validate_stage_targets(stage_targets: dict[str, StageTargetSet]) -> dict[str, object]:
    """Hard gate: counts and the aggregate stage distribution must match the
    Phase 7B inspection report exactly. Also checks INITIAL_ACCESS per-step
    coverage matches the previously-reported near-zero validation/test
    coverage, so any silent change in the dataset or the mapping is caught."""
    issues: list[str] = []
    report: dict[str, object] = {"splits": {}}

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        target_set = stage_targets[split]
        n = target_set.aggregate_class_index.shape[0]
        if n != expected_count:
            issues.append(f"{split}: count={n} != expected {expected_count}")

        aggregate_distribution = dict(Counter(target_set.aggregate_stage_name.tolist()))
        expected_distribution = EXPECTED_AGGREGATE_DISTRIBUTION[split]
        if aggregate_distribution != expected_distribution:
            issues.append(
                f"{split}: aggregate stage distribution {aggregate_distribution} != "
                f"expected {expected_distribution}"
            )

        initial_access_per_step = [
            int((target_set.per_step_stage_name[:, step] == "INITIAL_ACCESS").sum())
            for step in range(FORECAST_HORIZON_WINDOWS)
        ]
        expected_initial_access = EXPECTED_INITIAL_ACCESS_PER_STEP[split]
        if initial_access_per_step != expected_initial_access:
            issues.append(
                f"{split}: INITIAL_ACCESS per-step counts {initial_access_per_step} != "
                f"expected {expected_initial_access}"
            )

        report["splits"][split] = {
            "count": n,
            "aggregate_distribution": aggregate_distribution,
            "initial_access_per_step": initial_access_per_step,
        }

    report["covered_stages"] = [stage.name for stage in COVERED_STAGES]
    report["absent_stages"] = [stage.name for stage in ABSENT_STAGES]
    report["issues"] = issues
    report["status"] = "PASS" if not issues else "FAIL"
    return report


def train_only_class_weights(stage_targets: dict[str, StageTargetSet]) -> np.ndarray:
    """Standard inverse-frequency class weights (N / (num_classes * count_c)),
    computed from TRAIN per-step target labels ONLY (flattened across all 6
    steps). Never touches validation/test labels."""
    flat_train = stage_targets["train"].per_step_class_index.reshape(-1)
    counts = np.bincount(flat_train, minlength=NUM_STAGE_CLASSES).astype(np.float64)
    total = counts.sum()
    weights = np.where(counts > 0, total / (NUM_STAGE_CLASSES * np.maximum(counts, 1)), 0.0)
    return weights.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    args = parser.parse_args()

    stage_targets = read_stage_targets(args.windows_dir)
    report = validate_stage_targets(stage_targets)
    weights = train_only_class_weights(stage_targets)
    report["train_only_class_weights"] = {
        CLASS_INDEX_TO_STAGE[i].name: float(weights[i]) for i in range(NUM_STAGE_CLASSES)
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
