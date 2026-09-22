"""Phase 7B: train ONLY a new MitreStageHead against a FROZEN Phase 6B Run-1 backbone.

Order of operations (all hard-stop on failure, per the Phase 7B protocol):
  1. Load the frozen Run-1 checkpoint (results/phase6b_vector_world_model_ablation/run1_existing_scaling).
  2. HARD PRE-FLIGHT: reproduce Run-1's frozen attack-forecasting test metrics
     using the exact Run-1 threshold-selection protocol. STOP if they don't match.
  3. Build stage targets independently (phase7b_stage_targets.py) and validate
     them against the exact numbers from the Phase 7B inspection report. STOP on mismatch.
  4. Verify stage targets are row-for-row aligned with the Phase 3.5 samples. STOP on mismatch.
  5. Freeze every backbone parameter; train ONLY MitreStageHead.
  6. Verify the backbone is byte-identical before/after training. STOP if not.
  7. Re-run the frozen attack head after training; verify metrics are unchanged. STOP if not.
  8. Evaluate the stage head (6 covered classes only) and persist everything under
     results/phase7b_mitre_stage_head/ -- never under any Phase 6B results directory.

Nothing in this file ever passes a future label into model.forward(). Stage
and state/attack targets are used only as loss/evaluation arguments.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # NetOracle/, for `forecasting` -- explicit, not relying on import order
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from phase4_baseline import calculate_metrics
from phase6b_ablation import TARGET_FPR, choose_threshold_recall_at_fpr
from phase6b_vector_world_model import BATCH_SIZE, LEARNING_RATE, SEED, scale_array, set_seed
from phase7b_mitre_stage_head import (
    NUM_STAGE_CLASSES,
    VectorWorldModelWithStageHead,
    assert_only_stage_head_trainable,
    backbone_state_dict,
    backbone_unchanged,
    freeze_backbone,
    load_frozen_backbone,
    predict_all,
    set_frozen_backbone_training_mode,
)
from phase7b_stage_targets import (
    ABSENT_STAGES,
    CLASS_INDEX_TO_STAGE,
    COVERED_STAGES,
    read_stage_targets,
    train_only_class_weights,
    validate_stage_targets,
)
from forecasting.mitre_mapping import MitreStage
from world_model_dataset import EXPECTED_SPLIT_COUNTS, read_world_model_samples, validate_world_model_samples

WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data/windows"
RUN1_DIR = Path(__file__).resolve().parent / "results/phase6b_vector_world_model_ablation/run1_existing_scaling"
RUN1_CHECKPOINT = RUN1_DIR / "model/best_model.pt"
RUN1_SCALER = RUN1_DIR / "model/scaler.joblib"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head"

EXPECTED_RUN1_TEST_METRICS = {"pr_auc": 0.849, "roc_auc": 0.872, "f1": 0.747, "false_positive_rate": 0.096}
REPRODUCTION_TOLERANCE = 0.01
EXACT_MATCH_TOLERANCE = 1e-6  # for the post-training re-verification (same weights, same protocol)

STAGE_MAX_EPOCHS = 30
STAGE_PATIENCE = 8
STATE_LOSS_WEIGHT = 1.0
ATTACK_LOSS_WEIGHT = 1.0
STAGE_LOSS_WEIGHT = 1.0


def _hard_stop(output_dir: Path, reason: str, details: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "HARD_STOP", "reason": reason, "details": details}
    (output_dir / "HARD_STOP.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_preflight_check(
    model: VectorWorldModelWithStageHead,
    samples,
    scaled_x: dict[str, np.ndarray],
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    """Reproduces Run-1's attack-forecasting test metrics using the exact
    Run-1 protocol (recall-at-FPR<=5% threshold, selected on validation)."""
    _, validation_attack_prob, _ = predict_all(model, scaled_x["validation"], device, use_amp)
    threshold_info = choose_threshold_recall_at_fpr(samples["validation"].label, validation_attack_prob, TARGET_FPR)
    threshold = threshold_info["threshold"]

    _, test_attack_prob, _ = predict_all(model, scaled_x["test"], device, use_amp)
    test_metrics = calculate_metrics(samples["test"].label, test_attack_prob, threshold)

    reproduced = {
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "f1": test_metrics["f1"],
        "false_positive_rate": test_metrics["false_positive_rate"],
    }
    mismatches = {
        key: {"reproduced": reproduced[key], "expected": EXPECTED_RUN1_TEST_METRICS[key]}
        for key in EXPECTED_RUN1_TEST_METRICS
        if abs(reproduced[key] - EXPECTED_RUN1_TEST_METRICS[key]) > REPRODUCTION_TOLERANCE
    }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "reproduced_test_metrics": reproduced,
        "expected_test_metrics": EXPECTED_RUN1_TEST_METRICS,
        "mismatches": mismatches,
        "full_test_metrics": test_metrics,
    }


def aggregate_predicted_stage(step_stage_names: list[str], step_confidences: list[float]) -> tuple[str, float]:
    """Predicted-side analog of MitreMapper.from_label_set(): pick the
    highest-value predicted stage among the 6 steps; ties broken
    alphabetically by stage name (there is no raw label string to break
    ties by at inference time, unlike at training-target construction)."""
    max_value = max(MitreStage[name].value for name in step_stage_names)
    candidates = [
        (name, confidence)
        for name, confidence in zip(step_stage_names, step_confidences)
        if MitreStage[name].value == max_value
    ]
    candidates.sort(key=lambda item: item[0])
    winning_name = candidates[0][0]
    winning_confidence = max(confidence for name, confidence in candidates if name == winning_name)
    return winning_name, winning_confidence


def save_predictions(
    path: Path,
    source_file: np.ndarray,
    window_start: np.ndarray,
    label: np.ndarray,
    attack_probability: np.ndarray,
    stage_probability: np.ndarray,
) -> None:
    rows = []
    for i in range(len(label)):
        step_indices = stage_probability[i].argmax(axis=-1)
        step_names = [CLASS_INDEX_TO_STAGE[int(idx)].name for idx in step_indices]
        step_confidences = [float(stage_probability[i, k, step_indices[k]]) for k in range(6)]
        overall_name, overall_confidence = aggregate_predicted_stage(step_names, step_confidences)
        row = {
            "source_file": str(source_file[i]),
            "window_start": str(window_start[i]),
            "label": int(label[i]),
            "attack_probability": f"{attack_probability[i]:.12g}",
        }
        for step in range(6):
            row[f"mitre_stage_step_{step + 1}"] = step_names[step]
            row[f"mitre_confidence_step_{step + 1}"] = f"{step_confidences[step]:.6f}"
        row["mitre_stage_overall"] = overall_name
        row["overall_confidence"] = f"{overall_confidence:.6f}"
        rows.append(row)
    write_csv(path, rows)


def evaluate_stage_head(stage_targets, stage_probability: np.ndarray, split_name: str) -> dict[str, object]:
    """Flattened-per-step-slot evaluation over the 6 covered classes only."""
    y_true = stage_targets.per_step_class_index.reshape(-1)
    y_pred = stage_probability.reshape(-1, NUM_STAGE_CLASSES).argmax(axis=-1)

    labels = list(range(NUM_STAGE_CLASSES))
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    confusion = sk_confusion_matrix(y_true, y_pred, labels=labels)

    initial_access_index = None
    for index, stage in CLASS_INDEX_TO_STAGE.items():
        if stage == MitreStage.INITIAL_ACCESS:
            initial_access_index = index
            break

    per_class = {}
    for index in labels:
        stage_name = CLASS_INDEX_TO_STAGE[index].name
        test_support = int(support[index])
        if stage_name == "INITIAL_ACCESS" and test_support == 0:
            per_class[stage_name] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": test_support,
                "note": "not evaluable: zero examples of this stage in this split -- no fabricated score reported",
            }
        elif stage_name == "INITIAL_ACCESS" and test_support <= 6:
            per_class[stage_name] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": test_support,
                "note": f"support={test_support} (at most one window per horizon step) -- not statistically meaningful",
            }
        else:
            per_class[stage_name] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": test_support,
                "note": None,
            }

    per_step_macro_f1 = [
        float(
            f1_score(
                stage_targets.per_step_class_index[:, step],
                stage_probability[:, step, :].argmax(axis=-1),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        for step in range(6)
    ]

    return {
        "split": split_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": [CLASS_INDEX_TO_STAGE[i].name for i in labels],
        "per_step_macro_f1": per_step_macro_f1,
        "absent_stages_not_evaluated": [stage.name for stage in ABSENT_STAGES],
    }


def calibration_table(stage_probability: np.ndarray, stage_targets) -> tuple[list[dict[str, object]], float]:
    max_probability = stage_probability.max(axis=-1).reshape(-1)
    predicted_class = stage_probability.argmax(axis=-1).reshape(-1)
    true_class = stage_targets.per_step_class_index.reshape(-1)
    correct = (predicted_class == true_class).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, 11)
    rows = []
    ece = 0.0
    n = len(correct)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (max_probability >= lo) & (max_probability < hi) if hi < 1.0 else (max_probability >= lo) & (max_probability <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_confidence = float(max_probability[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / n) * abs(mean_confidence - accuracy)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return rows, ece


def progression_consistency(stage_probability: np.ndarray) -> dict[str, object]:
    predicted_class = stage_probability.argmax(axis=-1)  # [N, 6]
    predicted_values = np.vectorize(lambda idx: CLASS_INDEX_TO_STAGE[int(idx)].value)(predicted_class)
    non_decreasing = np.all(predicted_values[:, :-1] <= predicted_values[:, 1:], axis=1)
    return {
        "definition": "fraction of samples whose 6 raw per-step predicted MitreStage values form a "
        "non-decreasing sequence over t+1..t+6 (no unfiltered backward progression)",
        "consistent_fraction": float(non_decreasing.mean()),
        "sample_count": int(len(non_decreasing)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 7B directory: {args.output_dir}")

    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ---- Phase 3.5 interface (unchanged, read-only) + Run-1 scaler ----
    samples, feature_columns, source_files = read_world_model_samples(args.windows_dir)
    dataset_validation = validate_world_model_samples(samples, feature_columns)
    if dataset_validation["status"] != "PASS":
        _hard_stop(args.output_dir, "phase3_5_interface_validation_failed", dataset_validation)
        raise RuntimeError(f"world_model_dataset validation FAILED: {dataset_validation['issues']}")
    counts = tuple(samples[split].X.shape[0] for split in ("train", "validation", "test"))
    if counts != tuple(EXPECTED_SPLIT_COUNTS[split] for split in ("train", "validation", "test")):
        _hard_stop(args.output_dir, "phase3_5_interface_count_mismatch", {"counts": counts})
        raise RuntimeError(f"Phase 3.5 interface mismatch: counts={counts}")

    if not RUN1_SCALER.exists():
        _hard_stop(args.output_dir, "run1_scaler_not_found", {"path": str(RUN1_SCALER)})
        raise FileNotFoundError(f"Run-1 scaler not found: {RUN1_SCALER}")
    scaler = joblib.load(RUN1_SCALER)
    scaled_x = {split: scale_array(samples[split].X, scaler) for split in ("train", "validation", "test")}

    # ---- Step 1+2: load frozen checkpoint, HARD PRE-FLIGHT reproduction ----
    try:
        model = load_frozen_backbone(RUN1_CHECKPOINT, device)
    except Exception as exc:  # noqa: BLE001 -- intentional hard stop on ANY load failure
        _hard_stop(args.output_dir, "checkpoint_load_failed", {"error": str(exc), "path": str(RUN1_CHECKPOINT)})
        raise

    preflight = run_preflight_check(model, samples, scaled_x, device, use_amp)
    if preflight["status"] != "PASS":
        _hard_stop(args.output_dir, "run1_reproduction_failed", preflight)
        raise RuntimeError(f"HARD STOP: Run-1 attack-metric reproduction FAILED: {preflight['mismatches']}")

    # ---- Step 3: independent stage targets + hard validation gate ----
    stage_targets = read_stage_targets(args.windows_dir)
    stage_validation = validate_stage_targets(stage_targets)
    if stage_validation["status"] != "PASS":
        _hard_stop(args.output_dir, "stage_target_validation_failed", stage_validation)
        raise RuntimeError(f"HARD STOP: stage target validation FAILED: {stage_validation['issues']}")

    # ---- Step 4: alignment between world_model_dataset samples and independently-read targets ----
    alignment_issues = []
    for split in ("train", "validation", "test"):
        if not np.array_equal(samples[split].source_file, stage_targets[split].source_file):
            alignment_issues.append(f"{split}: source_file mismatch")
        if not np.array_equal(samples[split].window_start, stage_targets[split].window_start):
            alignment_issues.append(f"{split}: window_start mismatch")
    if alignment_issues:
        _hard_stop(args.output_dir, "target_alignment_failed", {"issues": alignment_issues})
        raise RuntimeError(f"HARD STOP: target alignment FAILED: {alignment_issues}")

    # ---- Step 5: freeze backbone, train ONLY stage_head ----
    freeze_backbone(model)
    assert_only_stage_head_trainable(model)
    backbone_before = backbone_state_dict(model)

    scaled_y = {split: scale_array(samples[split].Y, scaler) for split in ("train", "validation", "test")}
    class_weights = train_only_class_weights(stage_targets)
    class_weight_tensor = torch.tensor(class_weights, device=device)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled_x["train"]),
            torch.from_numpy(scaled_y["train"]),
            torch.from_numpy(samples["train"].label.astype(np.float32)),
            torch.from_numpy(stage_targets["train"].per_step_class_index),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    train_labels = samples["train"].label
    pos_weight = torch.tensor(float((train_labels == 0).sum()) / float(train_labels.sum()), device=device)
    attack_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stage_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = torch.optim.Adam(model.stage_head.parameters(), lr=LEARNING_RATE)
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir()
    best_path = model_dir / "best_stage_head.pt"

    history: list[dict[str, object]] = []
    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    grad_isolation_verified = False

    started = time.perf_counter()
    for epoch in range(1, STAGE_MAX_EPOCHS + 1):
        set_frozen_backbone_training_mode(model)
        state_losses, attack_losses, stage_losses = [], [], []
        for batch_x, batch_y, batch_label, batch_stage in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            batch_stage = batch_stage.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                y_hat, attack_logit, stage_logits, z_future = model(batch_x)
                state_loss = nn.functional.mse_loss(y_hat, batch_y)
                attack_loss = attack_criterion(attack_logit, batch_label)
                stage_loss = stage_criterion(stage_logits.permute(0, 2, 1), batch_stage)

            if not grad_isolation_verified:
                # One-time runtime proof that the frozen backbone truly blocks
                # gradient flow: y_hat/attack_logit/z_future must NOT require
                # grad (pure functions of frozen parameters + fresh input),
                # only stage_logits (via stage_head's trainable weights) may.
                if y_hat.requires_grad or attack_logit.requires_grad or z_future.requires_grad:
                    _hard_stop(
                        args.output_dir,
                        "backbone_not_isolated_from_autograd",
                        {"y_hat_requires_grad": y_hat.requires_grad, "attack_logit_requires_grad": attack_logit.requires_grad, "z_future_requires_grad": z_future.requires_grad},
                    )
                    raise RuntimeError("HARD STOP: backbone outputs unexpectedly require grad -- freeze invariant violated.")
                if not stage_logits.requires_grad:
                    _hard_stop(args.output_dir, "stage_head_not_trainable", {})
                    raise RuntimeError("HARD STOP: stage_logits does not require grad -- stage_head is not trainable.")
                grad_isolation_verified = True

            # Only stage_loss drives the optimizer step -- state_loss/attack_loss are
            # computed purely from frozen-backbone outputs (requires_grad=False, see
            # above) and contribute exactly zero gradient to stage_head; they are
            # computed here only for monitoring/logging, matching the requested loss
            # specification, and are intentionally excluded from the backward() call.
            trainable_loss = STAGE_LOSS_WEIGHT * stage_loss
            grad_scaler.scale(trainable_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            state_losses.append(float(state_loss.item()))
            attack_losses.append(float(attack_loss.item()))
            stage_losses.append(float(stage_loss.item()))

        _, val_attack_prob, val_stage_prob = predict_all(model, scaled_x["validation"], device, use_amp)
        val_pred = val_stage_prob.argmax(axis=-1)
        val_macro_f1_per_step = [
            float(
                f1_score(
                    stage_targets["validation"].per_step_class_index[:, step],
                    val_pred[:, step],
                    labels=list(range(NUM_STAGE_CLASSES)),
                    average="macro",
                    zero_division=0,
                )
            )
            for step in range(6)
        ]
        val_macro_f1 = float(np.mean(val_macro_f1_per_step))

        history.append(
            {
                "epoch": epoch,
                "train_state_loss_monitor_only": float(np.mean(state_losses)),
                "train_attack_loss_monitor_only": float(np.mean(attack_losses)),
                "train_stage_loss": float(np.mean(stage_losses)),
                "validation_macro_f1_mean_over_steps": val_macro_f1,
            }
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= STAGE_PATIENCE:
                break

    duration_seconds = time.perf_counter() - started

    # ---- Step 6: verify backbone truly unchanged ----
    backbone_after_training = backbone_state_dict(model)
    if not backbone_unchanged(backbone_before, backbone_after_training):
        _hard_stop(args.output_dir, "backbone_changed_during_training", {})
        raise RuntimeError("HARD STOP: backbone parameters changed during stage-head training.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    backbone_after_load = backbone_state_dict(model)
    if not backbone_unchanged(backbone_before, backbone_after_load):
        _hard_stop(args.output_dir, "backbone_changed_after_checkpoint_load", {})
        raise RuntimeError("HARD STOP: backbone parameters differ after loading the best stage-head checkpoint.")

    # ---- Step 7: rerun frozen attack head, verify identical to preflight ----
    _, posttrain_test_attack_prob, posttrain_test_stage_prob = predict_all(model, scaled_x["test"], device, use_amp)
    posttrain_test_metrics = calculate_metrics(samples["test"].label, posttrain_test_attack_prob, preflight["threshold"])
    attack_reproduction_after_training = {
        key: {"preflight": preflight["full_test_metrics"][key], "post_training": posttrain_test_metrics[key]}
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate")
    }
    attack_metrics_identical = all(
        abs(v["preflight"] - v["post_training"]) < EXACT_MATCH_TOLERANCE for v in attack_reproduction_after_training.values()
    )
    if not attack_metrics_identical:
        _hard_stop(args.output_dir, "attack_metrics_changed_after_stage_training", attack_reproduction_after_training)
        raise RuntimeError("HARD STOP: frozen attack-head metrics changed after stage-head training.")

    # ---- Step 8: evaluate stage head, persist everything ----
    _, val_attack_prob_final, val_stage_prob_final = predict_all(model, scaled_x["validation"], device, use_amp)
    _, test_attack_prob_final, test_stage_prob_final = predict_all(model, scaled_x["test"], device, use_amp)

    validation_stage_eval = evaluate_stage_head(stage_targets["validation"], val_stage_prob_final, "validation")
    test_stage_eval = evaluate_stage_head(stage_targets["test"], test_stage_prob_final, "test")
    calibration_rows, ece = calibration_table(test_stage_prob_final, stage_targets["test"])
    progression = progression_consistency(test_stage_prob_final)

    write_csv(args.output_dir / "training_log.csv", history)
    write_csv(
        args.output_dir / "confusion_matrix_test.csv",
        [
            {"true_stage": test_stage_eval["confusion_matrix_labels"][i], **{f"predicted_{name}": test_stage_eval["confusion_matrix"][i][j] for j, name in enumerate(test_stage_eval["confusion_matrix_labels"])}}
            for i in range(NUM_STAGE_CLASSES)
        ],
    )
    write_csv(args.output_dir / "calibration_test.csv", calibration_rows)
    save_predictions(
        args.output_dir / "predictions_validation.csv",
        samples["validation"].source_file,
        samples["validation"].window_start,
        samples["validation"].label,
        val_attack_prob_final,
        val_stage_prob_final,
    )
    save_predictions(
        args.output_dir / "predictions_test.csv",
        samples["test"].source_file,
        samples["test"].window_start,
        samples["test"].label,
        test_attack_prob_final,
        test_stage_prob_final,
    )

    parameter_count_total = sum(p.numel() for p in model.parameters())
    parameter_count_trainable = sum(p.numel() for p in model.stage_head.parameters())

    config = {
        "experiment": "phase7b_mitre_stage_head",
        "seed": SEED,
        "run1_checkpoint": str(RUN1_CHECKPOINT),
        "run1_scaler": str(RUN1_SCALER),
        "backbone_frozen": True,
        "trainable_parameters": "stage_head only",
        "parameter_count_total": parameter_count_total,
        "parameter_count_trainable": parameter_count_trainable,
        "num_stage_classes": NUM_STAGE_CLASSES,
        "covered_stages": [stage.name for stage in COVERED_STAGES],
        "absent_stages_not_represented_in_dataset": [stage.name for stage in ABSENT_STAGES],
        "loss": {
            "state_loss_weight": STATE_LOSS_WEIGHT,
            "attack_loss_weight": ATTACK_LOSS_WEIGHT,
            "stage_loss_weight": STAGE_LOSS_WEIGHT,
            "note": "state_loss and attack_loss are computed and logged each step for monitoring only; "
            "since the backbone is frozen (requires_grad=False), their outputs (y_hat, attack_logit) "
            "have requires_grad=False and contribute exactly zero gradient. Only stage_loss is passed "
            "to .backward(), and the optimizer is scoped to model.stage_head.parameters() only.",
            "stage_class_weights_train_only": {CLASS_INDEX_TO_STAGE[i].name: float(class_weights[i]) for i in range(NUM_STAGE_CLASSES)},
        },
        "training": {
            "max_epochs": STAGE_MAX_EPOCHS,
            "patience": STAGE_PATIENCE,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "duration_seconds": duration_seconds,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "mixed_precision": use_amp,
            "device": device.type,
        },
        "preflight_reproduction": preflight,
        "post_training_attack_reproduction": {
            "attack_metrics_identical_to_preflight": attack_metrics_identical,
            "comparison": attack_reproduction_after_training,
        },
        "backbone_unchanged_after_training": True,
        "backbone_unchanged_after_checkpoint_load": True,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    metrics = {
        "validation_stage": validation_stage_eval,
        "test_stage": test_stage_eval,
        "calibration_test": {"expected_calibration_error": ece, "bins": calibration_rows},
        "progression_consistency_test": progression,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    write_results_doc(args.output_dir, config, metrics)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preflight_status": preflight["status"],
                "backbone_unchanged": True,
                "best_epoch": best_epoch,
                "duration_seconds": duration_seconds,
                "test_stage_accuracy": test_stage_eval["accuracy"],
                "test_stage_macro_f1": test_stage_eval["macro_f1"],
                "attack_metrics_identical_to_preflight": attack_metrics_identical,
            },
            indent=2,
        )
    )


def write_results_doc(output_dir: Path, config: dict, metrics: dict) -> None:
    lines = [
        "# Phase 7B: MITRE Stage Head (frozen backbone)",
        "",
        "## Architecture",
        "",
        "`X [B,6,157] -> frozen Phase 6B Run-1 backbone -> z_future [B,6,128] -> "
        "(existing AttackForecastHead -> attack risk, unchanged) + (new MitreStageHead -> [B,6,6] stage logits)`.",
        "",
        "`VectorWorldModelWithStageHead` (experiments/phase7b_mitre_stage_head.py) subclasses "
        "`VectorWorldModel` (experiments/phase6b_vector_world_model.py, not modified) and reuses its "
        "`state_encoder`, `temporal_context`, `transition`, `state_decoder`, `attack_head` submodules "
        "unchanged. `forward()` returns `(y_hat, attack_logit, stage_logits, z_future)`.",
        "",
        "## Target construction",
        "",
        "`experiments/phase7b_stage_targets.py` independently re-reads `data/windows/*.csv` (does not "
        "import or modify `world_model_dataset.py`). Per-step targets come from each future row's own "
        "`current_attack_types`; the aggregate headline target comes from `future_attack_types`. Both "
        "route through the corrected, Phase-7A-tested `MitreMapper.from_label_set()`.",
        "",
        "## Leakage prevention",
        "",
        "`forward()` takes only `x` (observed history). Stage/state/attack targets never appear inside "
        "`forward()` -- verified at runtime (see tests.py: test_backbone_outputs_require_no_grad, "
        "test_forward_signature_has_no_label_arguments) that `y_hat`/`attack_logit`/`z_future` have "
        "`requires_grad=False` when the backbone is frozen, and only `stage_logits` carries a gradient.",
        "",
        "## Class coverage limitation",
        "",
        f"Only {config['num_stage_classes']} of 14 MitreStage values occur in this dataset: "
        f"{', '.join(config['covered_stages'])}. The following are NOT represented in CIC-IDS2018 "
        f"training data and are NOT evaluated: {', '.join(config['absent_stages_not_represented_in_dataset'])}.",
        "",
        "## Frozen-backbone protocol",
        "",
        f"Pre-flight reproduction status: **{config['preflight_reproduction']['status']}**. "
        f"Post-training attack-metric identity to pre-flight: "
        f"**{config['post_training_attack_reproduction']['attack_metrics_identical_to_preflight']}**. "
        f"Backbone byte-identical before/after training: **{config['backbone_unchanged_after_training']}**.",
        "",
        "## Metrics",
        "",
        f"Test stage accuracy: {metrics['test_stage']['accuracy']:.4f}, macro-F1: {metrics['test_stage']['macro_f1']:.4f}.",
        f"Progression consistency (test): {metrics['progression_consistency_test']['consistent_fraction']:.4f}.",
        f"Expected calibration error (test): {metrics['calibration_test']['expected_calibration_error']:.4f}.",
        "",
        "See metrics.json for full per-class precision/recall/support (including the explicit "
        "INITIAL_ACCESS zero-test-coverage note) and the confusion matrix.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
