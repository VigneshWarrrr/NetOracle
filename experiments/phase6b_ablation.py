"""Phase 6B controlled improvement ablation.

Two controlled runs, architecture held fixed (VectorWorldModel is imported
unmodified from phase6b_vector_world_model.py -- not redefined here):

  Run 1: existing StandardScaler-on-everything preprocessing + improved
         threshold selection (maximize recall subject to validation FPR <= 5%).
  Run 2: same, but the 4 diagnosed pathological __mean features use a
         log1p + train-only-standardize transform instead of raw StandardScaler;
         the other 153 features keep the existing StandardScaler policy.

Both runs use identical seed, batch size, learning rate, loss weights, and
model hyperparameters -- only preprocessing differs between them. Early
stopping is relaxed (more epochs, more patience) so epoch-1 is not an
automatic stop; the best checkpoint is still selected by validation PR-AUC.

No architecture change. No test-label use in threshold selection. No test
sweep to pick an operating point -- the validation-selected threshold is
applied to test exactly once. Phase 3-5 artifacts, the canonical dataset,
and the existing (non-ablation) Phase 6B experiment are read-only inputs
and are not modified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.amp import GradScaler, autocast

from phase4_baseline import calculate_metrics, choose_threshold
from phase6b_vector_world_model import (
    ATTACK_LOSS_WEIGHT,
    BATCH_SIZE,
    FORECAST_HORIZON_WINDOWS,
    HISTORY_WINDOWS,
    INPUT_SIZE,
    LEARNING_RATE,
    SEED,
    STATE_LOSS_WEIGHT,
    VectorWorldModel,
    make_loader,
    per_horizon_state_error,
    predict,
    save_predictions,
    set_seed,
    write_csv,
)
from world_model_dataset import EXPECTED_SPLIT_COUNTS, read_world_model_samples, validate_world_model_samples

WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data/windows"
OUTPUT_ROOT = Path(__file__).resolve().parent / "results/phase6b_vector_world_model_ablation"
PHASE5_METRICS_PATH = Path(__file__).resolve().parent / "results/phase5_temporal_transformer/metrics.json"

MAX_EPOCHS = 60      # substantially more than the original 25 (and the 6 actually run)
PATIENCE = 10         # will not stop immediately after epoch 1
TARGET_FPR = 0.05

PATHOLOGICAL_FEATURES = [
    "fwd_act_data_pkts__mean",
    "tot_fwd_pkts__mean",
    "subflow_fwd_pkts__mean",
    "fwd_header_len__mean",
]


# ============================================================
# A) Threshold selection: maximize recall subject to validation FPR <= target
# ============================================================


def choose_threshold_recall_at_fpr(labels: np.ndarray, probabilities: np.ndarray, max_fpr: float = TARGET_FPR) -> dict[str, float]:
    """Validation-only threshold: among cutoffs with FPR <= max_fpr, pick the one with highest recall."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = fpr <= max_fpr
    if not valid.any():
        idx = int(np.argmin(fpr))
    else:
        candidate_idx = np.where(valid)[0]
        idx = int(candidate_idx[np.argmax(tpr[candidate_idx])])
    threshold = float(thresholds[idx])
    return {"threshold": threshold, "achieved_validation_fpr": float(fpr[idx]), "achieved_validation_recall": float(tpr[idx])}


# ============================================================
# B) Preprocessing: existing StandardScaler vs hybrid log1p+standardize
#    for the 4 pathological features only. All statistics fit on TRAIN ONLY.
# ============================================================


class HybridFeatureScaler:
    """StandardScaler for all 157 features, overridden by log1p + train-only
    standardization for the 4 diagnosed non-negative heavy-tailed __mean
    features. Labels are never transformed."""

    def __init__(self, feature_columns: list[str], pathological_features: list[str]) -> None:
        self.feature_columns = list(feature_columns)
        self.pathological_features = list(pathological_features)
        self.pathological_indices = [self.feature_columns.index(name) for name in pathological_features]
        self.base_scaler = StandardScaler()
        self.log_mean_: np.ndarray | None = None
        self.log_std_: np.ndarray | None = None

    def fit(self, train_x: np.ndarray) -> "HybridFeatureScaler":
        flat = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64)
        self.base_scaler.fit(flat)
        log_values = np.log1p(np.clip(flat[:, self.pathological_indices], a_min=0.0, a_max=None))
        self.log_mean_ = log_values.mean(axis=0)
        self.log_std_ = log_values.std(axis=0)
        self.log_std_[self.log_std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).astype(np.float64)
        transformed = self.base_scaler.transform(flat)
        log_values = np.log1p(np.clip(flat[:, self.pathological_indices], a_min=0.0, a_max=None))
        transformed[:, self.pathological_indices] = (log_values - self.log_mean_) / self.log_std_
        return transformed.reshape(shape).astype(np.float32)


def fit_standard_scaler(train_x: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    return scaler


def apply_standard_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = x.shape
    return scaler.transform(x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


# ============================================================
# Training (identical procedure for both runs; only scaled_x/scaled_y differ)
# ============================================================


def train_one_run(
    run_name: str,
    samples: dict,
    scaled_x: dict[str, np.ndarray],
    scaled_y: dict[str, np.ndarray],
    scaler: object,
    output_dir: Path,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "model"
    model_dir.mkdir()

    set_seed(SEED)
    model = VectorWorldModel().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    train_labels = samples["train"].label
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    pos_weight = torch.tensor(negative / positive, device=device)
    attack_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    train_loader = make_loader(scaled_x["train"], scaled_y["train"], samples["train"].label, BATCH_SIZE, shuffle=True)

    history: list[dict[str, float]] = []
    best_val_pr_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    first_batch_device: str | None = None
    best_path = model_dir / "best_model.pt"

    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        state_losses, attack_losses, total_losses = [], [], []
        for batch_x, batch_y, batch_label in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            if first_batch_device is None:
                first_batch_device = str(batch_x.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                y_hat, attack_logit = model(batch_x)
                state_loss = nn.functional.mse_loss(y_hat, batch_y)
                attack_loss = attack_criterion(attack_logit, batch_label)
                loss = STATE_LOSS_WEIGHT * state_loss + ATTACK_LOSS_WEIGHT * attack_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            state_losses.append(float(state_loss.item()))
            attack_losses.append(float(attack_loss.item()))
            total_losses.append(float(loss.item()))

        validation_y_hat, validation_probabilities = predict(model, scaled_x["validation"], device, use_amp)
        validation_pr_auc = float(average_precision_score(samples["validation"].label, validation_probabilities))
        validation_state_mse = float(np.mean((validation_y_hat - scaled_y["validation"]) ** 2))
        history.append(
            {
                "epoch": float(epoch),
                "train_total_loss": float(np.mean(total_losses)),
                "train_state_loss": float(np.mean(state_losses)),
                "train_attack_loss": float(np.mean(attack_losses)),
                "validation_pr_auc": validation_pr_auc,
                "validation_state_loss_mse": validation_state_mse,
            }
        )
        if validation_pr_auc > best_val_pr_auc:
            best_val_pr_auc = validation_pr_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    duration_seconds = time.perf_counter() - started

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    joblib.dump(scaler, model_dir / "scaler.joblib")

    validation_y_hat, validation_probabilities = predict(model, scaled_x["validation"], device, use_amp)
    test_y_hat, test_probabilities = predict(model, scaled_x["test"], device, use_amp)

    # -- A) threshold selection: primary = recall @ FPR<=5% on validation only --
    primary_threshold_info = choose_threshold_recall_at_fpr(samples["validation"].label, validation_probabilities, TARGET_FPR)
    primary_threshold = primary_threshold_info["threshold"]
    # existing F1-max rule, reported for comparison only -- NOT the operating point
    reference_f1_threshold = choose_threshold(samples["validation"].label, validation_probabilities)

    validation_metrics_primary = calculate_metrics(samples["validation"].label, validation_probabilities, primary_threshold)
    test_metrics_primary = calculate_metrics(samples["test"].label, test_probabilities, primary_threshold)
    validation_metrics_reference = calculate_metrics(samples["validation"].label, validation_probabilities, reference_f1_threshold)
    test_metrics_reference = calculate_metrics(samples["test"].label, test_probabilities, reference_f1_threshold)

    validation_ranking = {
        "pr_auc": float(average_precision_score(samples["validation"].label, validation_probabilities)),
        "roc_auc": float(roc_auc_score(samples["validation"].label, validation_probabilities)),
    }
    test_ranking = {
        "pr_auc": float(average_precision_score(samples["test"].label, test_probabilities)),
        "roc_auc": float(roc_auc_score(samples["test"].label, test_probabilities)),
    }

    validation_horizon_error = per_horizon_state_error(validation_y_hat, scaled_y["validation"])
    test_horizon_error = per_horizon_state_error(test_y_hat, scaled_y["test"])

    best_epoch_row = next((row for row in history if int(row["epoch"]) == best_epoch), history[-1])

    write_csv(output_dir / "training_log.csv", history)
    write_csv(output_dir / "per_horizon_state_error_validation.csv", validation_horizon_error)
    write_csv(output_dir / "per_horizon_state_error_test.csv", test_horizon_error)
    save_predictions(output_dir / "predictions_validation.csv", samples["validation"], validation_probabilities)
    save_predictions(output_dir / "predictions_test.csv", samples["test"], test_probabilities)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"model": run_name, "split": "validation", "threshold_rule": "recall_at_fpr<=5pct", **validation_metrics_primary["confusion_matrix"]},
            {"model": run_name, "split": "test", "threshold_rule": "recall_at_fpr<=5pct", **test_metrics_primary["confusion_matrix"]},
            {"model": run_name, "split": "validation", "threshold_rule": "f1_max_reference_only", **validation_metrics_reference["confusion_matrix"]},
            {"model": run_name, "split": "test", "threshold_rule": "f1_max_reference_only", **test_metrics_reference["confusion_matrix"]},
        ],
    )

    result = {
        "run_name": run_name,
        "parameter_count": parameter_count,
        "device": device.type,
        "mixed_precision": use_amp,
        "first_training_batch_device": first_batch_device,
        "epochs_run": len(history),
        "max_epochs_allowed": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "training_duration_seconds": duration_seconds,
        "best_epoch_train_state_loss": best_epoch_row["train_state_loss"],
        "best_epoch_train_attack_loss": best_epoch_row["train_attack_loss"],
        "best_epoch_train_total_loss": best_epoch_row["train_total_loss"],
        "best_epoch_validation_state_loss_mse": best_epoch_row["validation_state_loss_mse"],
        "primary_threshold_selection": {
            "rule": f"maximize recall subject to validation FPR <= {TARGET_FPR}",
            **primary_threshold_info,
        },
        "reference_f1_threshold": {
            "rule": "F1-maximizing threshold on validation (existing Phase 4/5/6B rule) -- reported for comparison only, NOT the operating point",
            "threshold": reference_f1_threshold,
        },
        "validation_ranking": validation_ranking,
        "test_ranking": test_ranking,
        "validation_metrics_primary_threshold": validation_metrics_primary,
        "test_metrics_primary_threshold": test_metrics_primary,
        "validation_metrics_reference_f1_threshold": validation_metrics_reference,
        "test_metrics_reference_f1_threshold": test_metrics_reference,
        "per_horizon_state_error": {"validation": validation_horizon_error, "test": test_horizon_error},
        "pos_weight": float(negative / positive),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "seed": SEED,
                "input_shape": [HISTORY_WINDOWS, INPUT_SIZE],
                "output_shape": [FORECAST_HORIZON_WINDOWS, INPUT_SIZE],
                "architecture": "VectorWorldModel (imported unmodified from phase6b_vector_world_model.py)",
                "loss": f"{STATE_LOSS_WEIGHT} * MSE(Y_hat, Y) + {ATTACK_LOSS_WEIGHT} * BCEWithLogits(attack_logit, label)",
                "pos_weight_source": "computed from TRAIN split labels only",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "mixed_precision": use_amp,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def load_phase5_metrics() -> dict[str, object] | None:
    if not PHASE5_METRICS_PATH.exists():
        return None
    return json.loads(PHASE5_METRICS_PATH.read_text(encoding="utf-8"))


def comparison_row(name: str, metrics: dict[str, object]) -> str:
    cm = metrics["confusion_matrix"] if "confusion_matrix" in metrics else {key: metrics[key] for key in ("tp", "tn", "fp", "fn")}
    fpr = metrics.get("false_positive_rate", metrics.get("fpr", 0.0))
    return (
        f"| {name} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | "
        f"{metrics['pr_auc']:.6f} | {metrics['roc_auc']:.6f} | {fpr:.6f} | "
        f"{cm['tp']} | {cm['tn']} | {cm['fp']} | {cm['fn']} |"
    )


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing ablation directory: {OUTPUT_ROOT}")

    set_seed(SEED)
    samples, feature_columns, source_files = read_world_model_samples(WINDOWS_DIR)
    dataset_validation = validate_world_model_samples(samples, feature_columns)
    if dataset_validation["status"] != "PASS":
        raise ValueError(f"world_model_dataset validation FAILED, refusing to train: {dataset_validation['issues']}")
    counts = tuple(samples[split].X.shape[0] for split in ("train", "validation", "test"))
    expected_counts = tuple(EXPECTED_SPLIT_COUNTS[split] for split in ("train", "validation", "test"))
    if counts != expected_counts:
        raise ValueError(f"Phase 3.5 interface mismatch: counts={counts}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    OUTPUT_ROOT.mkdir(parents=True)

    # ---------------------------------------------------------------
    # Run 1: existing StandardScaler-on-everything preprocessing
    # ---------------------------------------------------------------
    standard_scaler = fit_standard_scaler(samples["train"].X)
    scaled_x_run1 = {s: apply_standard_scaler(samples[s].X, standard_scaler) for s in ("train", "validation", "test")}
    scaled_y_run1 = {s: apply_standard_scaler(samples[s].Y, standard_scaler) for s in ("train", "validation", "test")}
    result_run1 = train_one_run(
        "run1_existing_scaling_improved_threshold",
        samples,
        scaled_x_run1,
        scaled_y_run1,
        standard_scaler,
        OUTPUT_ROOT / "run1_existing_scaling",
        device,
        use_amp,
    )

    # ---------------------------------------------------------------
    # Run 2: hybrid scaling -- log1p + train-only-standardize for the
    # 4 pathological __mean features; existing StandardScaler for the rest.
    # ---------------------------------------------------------------
    hybrid_scaler = HybridFeatureScaler(feature_columns, PATHOLOGICAL_FEATURES).fit(samples["train"].X)
    scaled_x_run2 = {s: hybrid_scaler.transform(samples[s].X) for s in ("train", "validation", "test")}
    scaled_y_run2 = {s: hybrid_scaler.transform(samples[s].Y) for s in ("train", "validation", "test")}
    result_run2 = train_one_run(
        "run2_robust_pathological_scaling_improved_threshold",
        samples,
        scaled_x_run2,
        scaled_y_run2,
        hybrid_scaler,
        OUTPUT_ROOT / "run2_robust_scaling",
        device,
        use_amp,
    )

    phase5_metrics = load_phase5_metrics()

    summary = {
        "dataset_validation": dataset_validation,
        "counts": {
            split: {
                "samples": int(samples[split].X.shape[0]),
                "positive": int(samples[split].label.sum()),
                "negative": int((samples[split].label == 0).sum()),
            }
            for split in ("train", "validation", "test")
        },
        "pathological_features": PATHOLOGICAL_FEATURES,
        "run1_existing_scaling": result_run1,
        "run2_robust_scaling": result_run2,
        "phase5_temporal_transformer_test": phase5_metrics["test"] if phase5_metrics else None,
    }
    (OUTPUT_ROOT / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 6B Controlled Improvement Ablation",
        "",
        "Architecture is unchanged (VectorWorldModel imported unmodified). Two controlled runs differ ONLY in "
        "feature preprocessing; both use the same improved threshold-selection rule (maximize recall subject to "
        "validation FPR <= 5%, validation-only) and the same relaxed early-stopping budget "
        f"(max_epochs={MAX_EPOCHS}, patience={PATIENCE}). The existing F1-max validation threshold is reported "
        "for comparison only and is not the operating point. Test labels were never used for threshold selection; "
        "the selected threshold is applied to test exactly once (no test sweep).",
        "",
        "## Run summary",
        "",
        "| Run | Parameters | Best epoch | Epochs run | Training time (s) | Primary threshold | Val FPR @ threshold | Val recall @ threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (result_run1, result_run2):
        pt = result["primary_threshold_selection"]
        lines.append(
            f"| {result['run_name']} | {result['parameter_count']} | {result['best_epoch']} | {result['epochs_run']} | "
            f"{result['training_duration_seconds']:.3f} | {pt['threshold']:.6f} | {pt['achieved_validation_fpr']:.6f} | "
            f"{pt['achieved_validation_recall']:.6f} |"
        )

    lines += [
        "",
        "## Validation / test metrics at the PRIMARY threshold (recall @ FPR<=5%)",
        "",
        "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (result_run1, result_run2):
        lines.append(comparison_row(f"{result['run_name']} (validation)", result["validation_metrics_primary_threshold"]))
        lines.append(comparison_row(f"{result['run_name']} (test)", result["test_metrics_primary_threshold"]))
    if phase5_metrics:
        lines.append(comparison_row("phase5_temporal_transformer (test, its own selected threshold)", phase5_metrics["test"]))

    lines += [
        "",
        "## Reference only: existing F1-max validation threshold (NOT the operating point)",
        "",
        "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (result_run1, result_run2):
        lines.append(comparison_row(f"{result['run_name']} (validation)", result["validation_metrics_reference_f1_threshold"]))
        lines.append(comparison_row(f"{result['run_name']} (test)", result["test_metrics_reference_f1_threshold"]))

    lines += [
        "",
        "## Per-horizon state prediction error (test, standardized feature scale -- NOTE: run 2's scale for the "
        "4 pathological features differs from run 1's by construction, so raw MSE magnitudes are not directly "
        "comparable feature-for-feature between runs; the reduction in total test MSE is the intended effect being measured)",
        "",
        "| Horizon | Run 1 MSE | Run 1 MAE | Run 2 MSE | Run 2 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row1, row2 in zip(result_run1["per_horizon_state_error"]["test"], result_run2["per_horizon_state_error"]["test"]):
        lines.append(
            f"| t+{row1['horizon_step']} | {row1['mse_standardized']:.6f} | {row1['mae_standardized']:.6f} | "
            f"{row2['mse_standardized']:.6f} | {row2['mae_standardized']:.6f} |"
        )

    lines += [
        "",
        "## Training loss components at the best (checkpointed) epoch",
        "",
        "| Run | Best epoch | Train state loss (MSE) | Train attack loss (BCE) | Val state loss (MSE) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in (result_run1, result_run2):
        lines.append(
            f"| {result['run_name']} | {result['best_epoch']} | {result['best_epoch_train_state_loss']:.6f} | "
            f"{result['best_epoch_train_attack_loss']:.6f} | {result['best_epoch_validation_state_loss_mse']:.6f} |"
        )

    lines += [
        "",
        "No test-label information was used to select any threshold. No architecture change was made in either run "
        "(both import VectorWorldModel unmodified from phase6b_vector_world_model.py). Improvement claims should be "
        "read from the primary-threshold table and the ranking metrics (PR-AUC/ROC-AUC), not from any test sweep.",
        "",
    ]
    (OUTPUT_ROOT / "ablation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_ROOT),
                "run1": {
                    "best_epoch": result_run1["best_epoch"],
                    "primary_threshold": result_run1["primary_threshold_selection"],
                    "validation_primary": result_run1["validation_metrics_primary_threshold"],
                    "test_primary": result_run1["test_metrics_primary_threshold"],
                },
                "run2": {
                    "best_epoch": result_run2["best_epoch"],
                    "primary_threshold": result_run2["primary_threshold_selection"],
                    "validation_primary": result_run2["validation_metrics_primary_threshold"],
                    "test_primary": result_run2["test_metrics_primary_threshold"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
