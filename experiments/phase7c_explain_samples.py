"""Phase 7C: generate explanation records for a small, deterministic sample set.

Uses the real, already-trained Phase 7B checkpoint
(results/phase7b_mitre_stage_head/model/best_stage_head.pt) and the exact
Run-1 scaler. Cross-checks every explained sample's recomputed
attack_probability and per-step MITRE stages/confidences against the
already-persisted results/phase7b_mitre_stage_head/predictions_test.csv row
for the same (source_file, window_start) -- proving this is genuinely the
same frozen model, not a re-trained or re-thresholded one.

Sample selection is deterministic (first N in file order, split by
predicted label) -- no randomness, no test-label-driven tuning, and this
starts small (5 samples) rather than explaining the full test set, per
instructions.

Writes everything under experiments/results/phase7c_explainability/ --
never into any Phase 6B/7A/7B results directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # NetOracle/, for `forecasting`/`explainability`

from phase7c_explainability import explain_sample, load_trained_phase7b_model
from world_model_dataset import read_world_model_samples

WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data/windows"
RUN1_DIR = Path(__file__).resolve().parent / "results/phase6b_vector_world_model_ablation/run1_existing_scaling"
RUN1_SCALER = RUN1_DIR / "model/scaler.joblib"
PHASE7B_CHECKPOINT = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/model/best_stage_head.pt"
PHASE7B_PREDICTIONS_TEST = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/predictions_test.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase7c_explainability"

N_BENIGN_SAMPLES = 3
N_ATTACK_SAMPLES = 2
PREDICTION_MATCH_TOLERANCE = 1e-3  # persisted CSV stores attack_probability to 12 significant figures via %.12g

# Files whose integrity must be proven unchanged by this run.
INTEGRITY_FILES = [
    RUN1_DIR / "model/best_model.pt",
    RUN1_DIR / "model/scaler.joblib",
    PHASE7B_CHECKPOINT,
    Path(__file__).resolve().parent / "phase6b_vector_world_model.py",
    Path(__file__).resolve().parent / "phase7b_mitre_stage_head.py",
    Path(__file__).resolve().parents[1] / "forecasting/mitre_mapping.py",
    Path(__file__).resolve().parents[1] / "explainability/shap_explainer.py",
]


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_deterministic_samples(test_samples, predictions_csv_rows: list[dict[str, str]]) -> list[int]:
    """First N_BENIGN_SAMPLES with label==0 and first N_ATTACK_SAMPLES with
    label==1, in file order (deterministic, no randomness, no test-label
    tuning -- selection uses the label only to pick a mix, never to alter
    any prediction or threshold)."""
    benign_indices: list[int] = []
    attack_indices: list[int] = []
    for index in range(len(test_samples.label)):
        label = int(test_samples.label[index])
        if label == 0 and len(benign_indices) < N_BENIGN_SAMPLES:
            benign_indices.append(index)
        elif label == 1 and len(attack_indices) < N_ATTACK_SAMPLES:
            attack_indices.append(index)
        if len(benign_indices) >= N_BENIGN_SAMPLES and len(attack_indices) >= N_ATTACK_SAMPLES:
            break
    return sorted(benign_indices + attack_indices)


def load_persisted_predictions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["source_file"], row["window_start"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 7C directory: {args.output_dir}")

    for path in INTEGRITY_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    integrity_before = {str(path): _file_signature(path) for path in INTEGRITY_FILES}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_trained_phase7b_model(PHASE7B_CHECKPOINT, device)
    checkpoint_state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    scaler = joblib.load(RUN1_SCALER)
    samples, feature_columns, _ = read_world_model_samples(args.windows_dir)
    test_samples = samples["test"]

    persisted_predictions = load_persisted_predictions(PHASE7B_PREDICTIONS_TEST)
    selected_indices = select_deterministic_samples(test_samples, persisted_predictions)

    args.output_dir.mkdir(parents=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()

    validation_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for order, index in enumerate(selected_indices, start=1):
        source_file = str(test_samples.source_file[index])
        window_start = str(test_samples.window_start[index])
        x_scaled = scaler.transform(test_samples.X[index].reshape(-1, 157)).reshape(6, 157).astype(np.float32)

        explanation = explain_sample(
            model=model,
            feature_names=feature_columns,
            x_scaled=x_scaled,
            scaler=scaler,
            source_file=source_file,
            window_start=window_start,
            device=device,
            top_k=args.top_k,
        )

        record_path = records_dir / f"sample_{order:04d}.json"
        record_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

        persisted = persisted_predictions.get((source_file, window_start))
        if persisted is None:
            validation_rows.append(
                {
                    "order": order,
                    "source_file": source_file,
                    "window_start": window_start,
                    "status": "FAIL",
                    "reason": "no matching persisted Phase 7B prediction row found",
                }
            )
            continue

        recomputed_probability = explanation["future_attack_risk_prediction"]["attack_probability"]
        persisted_probability = float(persisted["attack_probability"])
        probability_diff = abs(recomputed_probability - persisted_probability)

        recomputed_steps = explanation["mitre_stage_prediction"]["per_step_stage"]
        persisted_steps = [persisted[f"mitre_stage_step_{step}"] for step in range(1, 7)]
        stages_match = recomputed_steps == persisted_steps

        recomputed_overall = explanation["mitre_stage_prediction"]["overall_stage"]
        persisted_overall = persisted["mitre_stage_overall"]

        validation_rows.append(
            {
                "order": order,
                "source_file": source_file,
                "window_start": window_start,
                "true_label": int(test_samples.label[index]),
                "recomputed_attack_probability": recomputed_probability,
                "persisted_attack_probability": persisted_probability,
                "probability_diff": probability_diff,
                "probability_match": probability_diff <= PREDICTION_MATCH_TOLERANCE,
                "recomputed_per_step_stage": recomputed_steps,
                "persisted_per_step_stage": persisted_steps,
                "per_step_stage_match": stages_match,
                "recomputed_overall_stage": recomputed_overall,
                "persisted_overall_stage": persisted_overall,
                "overall_stage_match": recomputed_overall == persisted_overall,
                "status": "PASS" if (probability_diff <= PREDICTION_MATCH_TOLERANCE and stages_match and recomputed_overall == persisted_overall) else "FAIL",
                "attack_attribution_shape": list(np.asarray(explanation["temporal_evidence"]["attack_risk_attribution"]).shape),
                "stage_attribution_shape": list(np.asarray(explanation["temporal_evidence"]["stage_attribution"]).shape),
                "attack_attribution_all_finite": bool(np.isfinite(np.asarray(explanation["temporal_evidence"]["attack_risk_attribution"])).all()),
                "stage_attribution_all_finite": bool(np.isfinite(np.asarray(explanation["temporal_evidence"]["stage_attribution"])).all()),
                "attack_explanation_method": explanation["future_attack_risk_prediction"]["explanation_method"],
                "stage_explanation_method": explanation["mitre_stage_prediction"]["explanation_method"],
                "record_file": str(record_path.name),
            }
        )

    duration_seconds = time.perf_counter() - started

    checkpoint_state_after = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    checkpoint_unchanged = set(checkpoint_state_before.keys()) == set(checkpoint_state_after.keys()) and all(
        torch.equal(checkpoint_state_before[key], checkpoint_state_after[key]) for key in checkpoint_state_before
    )

    integrity_after = {str(path): _file_signature(path) for path in INTEGRITY_FILES}
    files_unchanged = integrity_before == integrity_after

    all_passed = all(row.get("status") == "PASS" for row in validation_rows)

    report = {
        "n_samples_explained": len(selected_indices),
        "duration_seconds": duration_seconds,
        "device": device.type,
        "checkpoint_path": str(PHASE7B_CHECKPOINT),
        "scaler_path": str(RUN1_SCALER),
        "feature_count": len(feature_columns),
        "checkpoint_state_unchanged_during_run": checkpoint_unchanged,
        "tracked_frozen_files_unchanged": files_unchanged,
        "tracked_frozen_files": [str(path) for path in INTEGRITY_FILES],
        "all_samples_passed_cross_check": all_passed,
        "samples": validation_rows,
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_samples": len(selected_indices),
                "all_samples_passed_cross_check": all_passed,
                "checkpoint_state_unchanged_during_run": checkpoint_unchanged,
                "tracked_frozen_files_unchanged": files_unchanged,
                "duration_seconds": duration_seconds,
            },
            indent=2,
        )
    )

    if not all_passed or not checkpoint_unchanged or not files_unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
