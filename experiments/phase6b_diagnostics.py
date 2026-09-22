"""Phase 6B diagnostics -- read-only investigation of the already-trained Vector World Model.

This script does NOT train, does NOT modify the model architecture or
hyperparameters, and does NOT touch Phase 3-5 artifacts. It:

  - loads the already-saved checkpoint `results/phase6b_vector_world_model/model/best_model.pt`
    and scaler in eval mode and runs forward passes only (no backward, no
    optimizer, no weight updates) to recover per-sample / per-feature
    rollout predictions that were not persisted by the training run, and
  - reads the already-saved `training_log.csv`, `predictions_*.csv`,
    `config.json`, `metrics.json` produced by that run.

All outputs are written under `results/phase6b_vector_world_model/diagnostics/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from phase4_baseline import calculate_metrics
from phase6b_vector_world_model import VectorWorldModel, predict, scale_array
from world_model_dataset import read_world_model_samples

RESULTS_DIR = Path(__file__).resolve().parent / "results/phase6b_vector_world_model"
WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data/windows"
DIAG_DIR = RESULTS_DIR / "diagnostics"
FIXED_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = [], []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            labels.append(int(row["label"]))
            probabilities.append(float(row["world_model_probability"]))
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    return {
        "count": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Phase 6B results directory not found: {RESULTS_DIR}")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((RESULTS_DIR / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    selected_threshold = float(config["training"]["selected_validation_threshold"])

    # -------------------------------------------------------
    # Load unchanged Phase 3.5 interface (read-only) + saved scaler/checkpoint
    # -------------------------------------------------------
    samples, feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
    scaler = joblib.load(RESULTS_DIR / "model" / "scaler.joblib")
    scaled_x = {split: scale_array(samples[split].X, scaler) for split in ("train", "validation", "test")}
    scaled_y = {split: scale_array(samples[split].Y, scaler) for split in ("train", "validation", "test")}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = VectorWorldModel().to(device)
    checkpoint = torch.load(RESULTS_DIR / "model" / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    report: dict[str, object] = {"reproduction_check": {}}

    # -------------------------------------------------------
    # Reproduce validation/test forward pass (eval only) and sanity-check
    # against the metrics already saved by the training run.
    # -------------------------------------------------------
    val_y_hat, val_prob = predict(model, scaled_x["validation"], device, use_amp)
    test_y_hat, test_prob = predict(model, scaled_x["test"], device, use_amp)

    reproduced_test_metrics = calculate_metrics(samples["test"].label, test_prob, selected_threshold)
    saved_test_metrics = metrics["test"]
    reproduction_matches = all(
        abs(reproduced_test_metrics[key] - saved_test_metrics[key]) < 1e-6
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc")
    )
    report["reproduction_check"] = {
        "matches_saved_metrics.json": reproduction_matches,
        "reproduced_test_pr_auc": reproduced_test_metrics["pr_auc"],
        "saved_test_pr_auc": saved_test_metrics["pr_auc"],
    }

    # =========================================================
    # 1 + per-feature contribution: test vs validation state MSE
    # =========================================================
    diff_test = test_y_hat - scaled_y["test"]          # [N,6,157] standardized residuals
    diff_val = val_y_hat - scaled_y["validation"]

    per_feature_mse_test = np.mean(diff_test ** 2, axis=(0, 1))   # [157]
    per_feature_mae_test = np.mean(np.abs(diff_test), axis=(0, 1))
    per_feature_mse_val = np.mean(diff_val ** 2, axis=(0, 1))
    per_feature_mae_val = np.mean(np.abs(diff_val), axis=(0, 1))

    order = np.argsort(-per_feature_mse_test)
    top10_idx = order[:10]

    feature_rows = []
    for rank, j in enumerate(order, start=1):
        feature_rows.append(
            {
                "rank": rank,
                "feature": feature_columns[j],
                "test_mse_standardized": float(per_feature_mse_test[j]),
                "test_mae_standardized": float(per_feature_mae_test[j]),
                "validation_mse_standardized": float(per_feature_mse_val[j]),
                "validation_mae_standardized": float(per_feature_mae_val[j]),
                "test_mse_share_of_total_pct": float(per_feature_mse_test[j] / per_feature_mse_test.sum() * 100.0),
            }
        )
    write_csv(DIAG_DIR / "per_feature_test_rollout_error.csv", feature_rows)
    top10_rows = feature_rows[:10]

    total_feature_mse_test = float(per_feature_mse_test.sum())
    total_feature_mse_val = float(per_feature_mse_val.sum())
    top10_share_of_total = float(sum(r["test_mse_share_of_total_pct"] for r in top10_rows))

    # =========================================================
    # Per-sample contribution (top 1% / 5% / 10% worst samples)
    # =========================================================
    sample_sse_test = np.sum(diff_test.reshape(diff_test.shape[0], -1) ** 2, axis=1)  # [N]
    total_sse_test = float(sample_sse_test.sum())
    n_test = sample_sse_test.shape[0]
    sorted_idx = np.argsort(-sample_sse_test)

    worst_share = {}
    for pct in (1, 5, 10):
        k = max(1, int(np.ceil(n_test * pct / 100.0)))
        worst_share[f"top_{pct}pct_samples"] = {
            "sample_count": k,
            "sse_share_of_total_pct": float(sample_sse_test[sorted_idx[:k]].sum() / total_sse_test * 100.0),
        }

    worst_rows = []
    for rank, i in enumerate(sorted_idx[:25], start=1):
        worst_rows.append(
            {
                "rank": rank,
                "source_file": str(samples["test"].source_file[i]),
                "window_start": str(samples["test"].window_start[i]),
                "label": int(samples["test"].label[i]),
                "future_attack_step": int(samples["test"].future_attack_step[i]),
                "predicted_probability": float(test_prob[i]),
                "sample_sse_standardized": float(sample_sse_test[i]),
                "sample_share_of_total_mse_pct": float(sample_sse_test[i] / total_sse_test * 100.0),
            }
        )
    write_csv(DIAG_DIR / "worst_test_samples.csv", worst_rows)

    # =========================================================
    # Predicted-vs-true ranges + train/val/test distributions for top-10 features
    # (both standardized scale and inverse-transformed raw scale)
    # =========================================================
    feature_distribution_rows = []
    for j in top10_idx:
        name = feature_columns[j]
        mean_j = float(scaler.mean_[j])
        scale_j = float(scaler.scale_[j])

        true_std_test = scaled_y["test"][:, :, j].reshape(-1)
        pred_std_test = test_y_hat[:, :, j].reshape(-1)
        true_raw_test = samples["test"].Y[:, :, j].reshape(-1)
        pred_raw_test = pred_std_test * scale_j + mean_j

        feature_distribution_rows.append(
            {
                "feature": name,
                "scaler_mean": mean_j,
                "scaler_scale": scale_j,
                "true_standardized": percentiles(true_std_test),
                "predicted_standardized": percentiles(pred_std_test),
                "true_raw": percentiles(true_raw_test),
                "predicted_raw": percentiles(pred_raw_test),
                "train_raw_distribution": percentiles(samples["train"].Y[:, :, j].reshape(-1)),
                "validation_raw_distribution": percentiles(samples["validation"].Y[:, :, j].reshape(-1)),
                "test_raw_distribution": percentiles(samples["test"].Y[:, :, j].reshape(-1)),
            }
        )
    (DIAG_DIR / "top10_feature_ranges_and_distributions.json").write_text(
        json.dumps(feature_distribution_rows, indent=2), encoding="utf-8"
    )

    # =========================================================
    # Training curve components (already logged by the training run; re-presented)
    # =========================================================
    training_log_path = RESULTS_DIR / "training_log.csv"
    with training_log_path.open("r", encoding="utf-8", newline="") as stream:
        training_log = list(csv.DictReader(stream))

    # =========================================================
    # Fixed-threshold stability of attack-probability ranking
    # =========================================================
    val_labels, val_probs_saved = read_predictions_csv(RESULTS_DIR / "predictions_validation.csv")
    test_labels, test_probs_saved = read_predictions_csv(RESULTS_DIR / "predictions_test.csv")

    fixed_threshold_rows = []
    for split_name, labels, probs in (("validation", val_labels, val_probs_saved), ("test", test_labels, test_probs_saved)):
        for threshold in FIXED_THRESHOLDS:
            result = calculate_metrics(labels, probs, threshold)
            fixed_threshold_rows.append(
                {
                    "split": split_name,
                    "threshold": threshold,
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "fpr": result["false_positive_rate"],
                    "tp": result["confusion_matrix"]["tp"],
                    "tn": result["confusion_matrix"]["tn"],
                    "fp": result["confusion_matrix"]["fp"],
                    "fn": result["confusion_matrix"]["fn"],
                }
            )
    write_csv(DIAG_DIR / "fixed_threshold_sweep.csv", fixed_threshold_rows)

    # Ranking metrics are threshold-independent -- report once per split for reference
    ranking_only = {
        "validation": {
            "pr_auc": float(average_precision_score(val_labels, val_probs_saved)),
            "roc_auc": float(roc_auc_score(val_labels, val_probs_saved)),
        },
        "test": {
            "pr_auc": float(average_precision_score(test_labels, test_probs_saved)),
            "roc_auc": float(roc_auc_score(test_labels, test_probs_saved)),
        },
    }

    # =========================================================
    # Checkpoint availability
    # =========================================================
    checkpoint_note = (
        "Only one checkpoint was persisted (best_model.pt, saved at the epoch with the highest "
        "validation PR-AUC). Training stopped after 6 epochs with best_epoch=1 and no later epoch "
        "improved on it, so no other epoch's weights were ever written to disk. Per-epoch validation "
        "PR-AUC was logged live during training (see training_log.csv / epoch curve below) and is "
        "reported in full. Per-epoch ROC-AUC was NOT logged during training and cannot be recovered "
        "for epochs 2-6 without re-running training with checkpointing enabled at every epoch, which "
        "would require retraining -- out of scope for this diagnostic-only pass."
    )

    # =========================================================
    # Calibration / reliability
    # =========================================================
    def calibration_table(labels: np.ndarray, probs: np.ndarray) -> tuple[list[dict[str, object]], float, float]:
        bin_edges = np.linspace(0.0, 1.0, 11)
        rows = []
        ece = 0.0
        n = labels.shape[0]
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
            count = int(mask.sum())
            if count == 0:
                rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0, "mean_predicted": None, "observed_positive_rate": None, "gap": None})
                continue
            mean_predicted = float(probs[mask].mean())
            observed = float(labels[mask].mean())
            gap = abs(mean_predicted - observed)
            ece += (count / n) * gap
            rows.append(
                {
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "observed_positive_rate": observed,
                    "gap": gap,
                }
            )
        brier = float(np.mean((probs - labels) ** 2))
        return rows, ece, brier

    val_calibration, val_ece, val_brier = calibration_table(val_labels.astype(np.float64), val_probs_saved)
    test_calibration, test_ece, test_brier = calibration_table(test_labels.astype(np.float64), test_probs_saved)
    write_csv(DIAG_DIR / "calibration_validation.csv", val_calibration)
    write_csv(DIAG_DIR / "calibration_test.csv", test_calibration)

    # =========================================================
    # Assemble full diagnostics.json
    # =========================================================
    diagnostics = {
        "reproduction_check": report["reproduction_check"],
        "1_test_vs_validation_state_mse": {
            "test_total_feature_mse_standardized": total_feature_mse_test,
            "validation_total_feature_mse_standardized": total_feature_mse_val,
            "ratio_test_over_validation": total_feature_mse_test / total_feature_mse_val,
            "top10_features_share_of_total_test_mse_pct": top10_share_of_total,
            "top10_features": top10_rows,
        },
        "worst_sample_concentration": worst_share,
        "worst_samples_file": "worst_test_samples.csv",
        "top10_feature_distributions_file": "top10_feature_ranges_and_distributions.json",
        "2_training_curve": {
            "epochs_run": len(training_log),
            "best_epoch": int(config["training"]["best_epoch"]),
            "patience": int(config["training"]["patience"]),
            "epoch_log": training_log,
            "checkpoint_note": checkpoint_note,
        },
        "3_threshold_analysis": {
            "selected_validation_threshold": selected_threshold,
            "selection_rule": "F1-maximizing threshold on VALIDATION predictions only (choose_threshold in phase4_baseline.py)",
            "validation_positive_rate": float(samples["validation"].label.mean()),
            "test_positive_rate": float(samples["test"].label.mean()),
            "train_positive_rate": float(samples["train"].label.mean()),
            "ranking_metrics": ranking_only,
            "fixed_threshold_sweep_file": "fixed_threshold_sweep.csv",
        },
        "calibration": {
            "validation_brier_score": val_brier,
            "test_brier_score": test_brier,
            "validation_expected_calibration_error": val_ece,
            "test_expected_calibration_error": test_ece,
            "validation_calibration_file": "calibration_validation.csv",
            "test_calibration_file": "calibration_test.csv",
        },
    }
    (DIAG_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
