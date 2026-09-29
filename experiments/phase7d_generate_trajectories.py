"""Phase 7D: generate trajectory records for a small, deterministic sample set.

Reuses the exact same deterministic sample-selection rule as Phase 7C
(`phase7c_explain_samples.select_deterministic_samples`, imported unmodified
-- not reimplemented) so this run's sample set is identical to Phase 7C's,
enabling a direct cross-check against Phase 7C's already-persisted
explanation records in addition to Phase 7B's persisted predictions.

Writes JSON + CSV trajectory records and a human-readable validation report
under experiments/results/phase7d_trajectory/ -- never into any Phase
6B/7B/7C results directory. Ground-truth labels are used ONLY in this
orchestration/validation script (to pick a benign/attack mix and to cross-
check predictions), never passed into `build_trajectory_record`'s model-
input path.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase7c_explain_samples import load_persisted_predictions, select_deterministic_samples
from phase7c_explainability import load_trained_phase7b_model
from phase7d_trajectory_narrative import build_trajectory_record
from world_model_dataset import read_world_model_samples

WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data/windows"
RUN1_DIR = Path(__file__).resolve().parent / "results/phase6b_vector_world_model_ablation/run1_existing_scaling"
RUN1_SCALER = RUN1_DIR / "model/scaler.joblib"
PHASE7B_CHECKPOINT = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/model/best_stage_head.pt"
PHASE7B_PREDICTIONS_TEST = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/predictions_test.csv"
PHASE7C_RECORDS_DIR = Path(__file__).resolve().parent / "results/phase7c_explainability/records"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase7d_trajectory"

PREDICTION_MATCH_TOLERANCE = 1e-3

INTEGRITY_FILES = [
    RUN1_DIR / "model/best_model.pt",
    RUN1_DIR / "model/scaler.joblib",
    PHASE7B_CHECKPOINT,
    Path(__file__).resolve().parent / "phase6b_vector_world_model.py",
    Path(__file__).resolve().parent / "phase7b_mitre_stage_head.py",
    Path(__file__).resolve().parent / "phase7c_explainability.py",
    Path(__file__).resolve().parents[1] / "forecasting/mitre_mapping.py",
    Path(__file__).resolve().parents[1] / "explainability/shap_explainer.py",
]


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _flatten_row(record: dict[str, object]) -> dict[str, object]:
    row = {
        "source_file": record["source_file"],
        "window_start": record["window_start"],
        "attack_probability": record["attack_probability"],
        "overall_predicted_stage": record["overall_predicted_stage"],
        "overall_confidence": record["overall_confidence"],
        "overall_stage_step": record["overall_stage_step"],
        "disagreement_detected": record["disagreement"]["detected"],
    }
    for step in record["steps"]:
        n = step["step"]
        row[f"step_{n}_predicted_stage"] = step["predicted_stage"]
        row[f"step_{n}_stage_confidence"] = step["stage_confidence"]
        row[f"step_{n}_top2_margin"] = step["top2_confidence_margin"]
        row[f"step_{n}_non_benign_probability"] = step["non_benign_probability"]
        row[f"step_{n}_top_feature"] = step["top_k_driving_features"][0]["feature"] if step["top_k_driving_features"] else ""
    return row


def main() -> None:
    if DEFAULT_OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 7D directory: {DEFAULT_OUTPUT_DIR}")

    for path in INTEGRITY_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    integrity_before = {str(path): _file_signature(path) for path in INTEGRITY_FILES}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_trained_phase7b_model(PHASE7B_CHECKPOINT, device)
    checkpoint_state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    scaler = joblib.load(RUN1_SCALER)
    samples, feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
    test_samples = samples["test"]

    persisted_predictions = load_persisted_predictions(PHASE7B_PREDICTIONS_TEST)
    selected_indices = select_deterministic_samples(test_samples, persisted_predictions)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True)
    records_dir = DEFAULT_OUTPUT_DIR / "records"
    records_dir.mkdir()

    validation_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for order, index in enumerate(selected_indices, start=1):
        source_file = str(test_samples.source_file[index])
        window_start = str(test_samples.window_start[index])
        x_scaled = scaler.transform(test_samples.X[index].reshape(-1, 157)).reshape(6, 157).astype(np.float32)

        record = build_trajectory_record(
            model=model,
            feature_names=feature_columns,
            x_scaled=x_scaled,
            scaler=scaler,
            source_file=source_file,
            window_start=window_start,
            device=device,
        )

        record_path = records_dir / f"trajectory_{order:04d}.json"
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        csv_rows.append(_flatten_row(record))

        # ---- cross-check against Phase 7B's persisted predictions (ground
        # truth label used HERE only, for validation reporting, never fed
        # into build_trajectory_record above) ----
        persisted = persisted_predictions.get((source_file, window_start))
        row_status = {"order": order, "source_file": source_file, "window_start": window_start, "record_file": record_path.name}
        if persisted is None:
            row_status["status"] = "FAIL"
            row_status["reason"] = "no matching persisted Phase 7B prediction row"
            validation_rows.append(row_status)
            continue

        probability_diff = abs(record["attack_probability"] - float(persisted["attack_probability"]))
        recomputed_steps = [step["predicted_stage"] for step in record["steps"]]
        persisted_steps = [persisted[f"mitre_stage_step_{step}"] for step in range(1, 7)]
        steps_match = recomputed_steps == persisted_steps
        overall_match = record["overall_predicted_stage"] == persisted["mitre_stage_overall"]

        attributions_finite = all(
            all(np.isfinite(feature["importance"]) for feature in step["top_k_driving_features"])
            for step in record["steps"]
        )
        six_steps_present = len(record["steps"]) == 6
        ordering_correct = [step["step"] for step in record["steps"]] == [1, 2, 3, 4, 5, 6]

        row_status.update(
            {
                "true_label": int(test_samples.label[index]),
                "probability_diff": probability_diff,
                "probability_match": probability_diff <= PREDICTION_MATCH_TOLERANCE,
                "per_step_stage_match": steps_match,
                "overall_stage_match": overall_match,
                "six_steps_present": six_steps_present,
                "step_ordering_correct": ordering_correct,
                "attribution_values_finite": attributions_finite,
                "status": "PASS"
                if (probability_diff <= PREDICTION_MATCH_TOLERANCE and steps_match and overall_match and six_steps_present and ordering_correct and attributions_finite)
                else "FAIL",
            }
        )
        validation_rows.append(row_status)

    duration_seconds = time.perf_counter() - started

    checkpoint_state_after = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    checkpoint_unchanged = set(checkpoint_state_before.keys()) == set(checkpoint_state_after.keys()) and all(
        torch.equal(checkpoint_state_before[key], checkpoint_state_after[key]) for key in checkpoint_state_before
    )
    integrity_after = {str(path): _file_signature(path) for path in INTEGRITY_FILES}
    files_unchanged = integrity_before == integrity_after

    # ---- determinism check: regenerate the first sample and compare ----
    first_index = selected_indices[0]
    x_scaled_repeat = scaler.transform(test_samples.X[first_index].reshape(-1, 157)).reshape(6, 157).astype(np.float32)
    repeat_record = build_trajectory_record(
        model=model,
        feature_names=feature_columns,
        x_scaled=x_scaled_repeat,
        scaler=scaler,
        source_file=str(test_samples.source_file[first_index]),
        window_start=str(test_samples.window_start[first_index]),
        device=device,
    )
    first_original = json.loads((records_dir / "trajectory_0001.json").read_text(encoding="utf-8"))
    deterministic = (
        repeat_record["attack_probability"] == first_original["attack_probability"]
        and repeat_record["steps"] == first_original["steps"]
        and repeat_record["narrative"] == first_original["narrative"]
    )

    with (DEFAULT_OUTPUT_DIR / "trajectories.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    all_passed = all(row.get("status") == "PASS" for row in validation_rows)

    report_lines = [
        "# Phase 7D Trajectory Generation -- Validation Report",
        "",
        f"Samples generated: {len(selected_indices)}",
        f"Duration: {duration_seconds:.3f}s on {device.type}",
        f"All samples passed cross-check against Phase 7B persisted predictions: {all_passed}",
        f"Checkpoint state unchanged during run: {checkpoint_unchanged}",
        f"Tracked frozen files unchanged: {files_unchanged}",
        f"Deterministic regeneration of sample 1 matches: {deterministic}",
        "",
        "## Per-sample results",
        "",
        "| # | source_file | window_start | label | prob diff | stages match | overall match | disagreement |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for row in validation_rows:
        report_lines.append(
            f"| {row['order']} | {row['source_file'][:30]} | {row['window_start']} | {row.get('true_label','?')} | "
            f"{row.get('probability_diff', float('nan')):.2e} | {row.get('per_step_stage_match','?')} | "
            f"{row.get('overall_stage_match','?')} | {json.loads((records_dir / row['record_file']).read_text())['disagreement']['detected']} |"
        )
    report_lines += [
        "",
        "## Example narrative (sample 1)",
        "",
        first_original["narrative"],
        "",
    ]
    (DEFAULT_OUTPUT_DIR / "VALIDATION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    validation_json = {
        "n_samples": len(selected_indices),
        "duration_seconds": duration_seconds,
        "device": device.type,
        "checkpoint_path": str(PHASE7B_CHECKPOINT),
        "all_samples_passed_cross_check": all_passed,
        "checkpoint_state_unchanged_during_run": checkpoint_unchanged,
        "tracked_frozen_files_unchanged": files_unchanged,
        "deterministic_regeneration_match": deterministic,
        "samples": validation_rows,
    }
    (DEFAULT_OUTPUT_DIR / "validation_report.json").write_text(json.dumps(validation_json, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "n_samples": len(selected_indices),
                "all_samples_passed_cross_check": all_passed,
                "checkpoint_state_unchanged_during_run": checkpoint_unchanged,
                "tracked_frozen_files_unchanged": files_unchanged,
                "deterministic_regeneration_match": deterministic,
                "duration_seconds": duration_seconds,
            },
            indent=2,
        )
    )

    if not (all_passed and checkpoint_unchanged and files_unchanged and deterministic):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
