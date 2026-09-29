"""Train the single Phase 4.3 unweighted-LSTM class-weight ablation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase4_baseline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATIENCE,
    ForecastLSTM,
    HISTORY_WINDOWS,
    choose_threshold,
    read_samples,
    scale_samples,
)


THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_PARAMETER_COUNT = 279169


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "fpr": float(fp / (fp + tn)) if tn + fp else 0.0,
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def predict(model: torch.nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(sequences)), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            values.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
    return np.concatenate(values)


def metadata_for_samples(windows_dir: Path) -> dict[str, list[dict[str, str]]]:
    metadata = {"train": [], "validation": [], "test": []}
    for partition in sorted(windows_dir.glob("*.csv")):
        with partition.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] == "1":
                metadata[row["split"]].append({
                    "sample_id": f"{partition.name}:{index}",
                    "source_file": partition.name,
                    "split": row["split"],
                    "window_start": row["window_start"],
                    "target": row["future_attack_within_horizon"],
                })
    return metadata


def save_predictions(path: Path, metadata: list[dict[str, str]], labels: np.ndarray, probabilities: np.ndarray) -> None:
    fields = ["sample_id", "source_file", "split", "window_start", "target", "lstm_probability"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item, label, probability in zip(metadata, labels, probabilities):
            writer.writerow({**item, "target": int(label), "lstm_probability": f"{probability:.12g}"})


def threshold_rows(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, object]]:
    return [{"threshold": threshold, **metrics(labels, probabilities, threshold)} for threshold in THRESHOLDS]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--baseline-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_baseline")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_3_class_weight_ablation")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing ablation directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the ablation is intentionally stopped")

    torch.manual_seed(42)
    np.random.seed(42)
    sample_sets, feature_columns, _ = read_samples(args.windows_dir)
    if len(feature_columns) != 157 or sample_sets["train"].sequences.shape[1:] != (6, 157):
        raise ValueError("Unexpected Phase 4 input shape or feature count")
    metadata = metadata_for_samples(args.windows_dir)
    if any(len(metadata[split]) != len(sample_sets[split].labels) for split in metadata):
        raise ValueError("Sample metadata is not aligned with Phase 4 reconstruction")
    if (len(sample_sets["train"].labels), len(sample_sets["validation"].labels), len(sample_sets["test"].labels)) != (29315, 6195, 6195):
        raise ValueError("Unexpected Phase 4 split counts")

    baseline_scaler = joblib.load(args.baseline_dir / "lstm/scaler.joblib")
    scaled = {split: scale_samples(sample_sets[split], baseline_scaler) for split in ("train", "validation", "test")}
    device = torch.device("cuda:0")
    model = ForecastLSTM(157, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Ablation model verification failed")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(scaled["train"].sequences), torch.from_numpy(scaled["train"].labels.astype(np.float32))), batch_size=DEFAULT_BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(TensorDataset(torch.from_numpy(scaled["validation"].sequences), torch.from_numpy(scaled["validation"].labels.astype(np.float32))), batch_size=DEFAULT_BATCH_SIZE, shuffle=False, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    args.output_dir.mkdir(parents=True)
    output_lstm = args.output_dir / "lstm"
    output_lstm.mkdir()
    best_path = output_lstm / "best_model.pt"
    history: list[dict[str, float]] = []
    best_pr_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    first_batch_device: str | None = None
    started = time.perf_counter()
    for epoch in range(1, DEFAULT_EPOCHS + 1):
        model.train()
        losses = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if first_batch_device is None:
                first_batch_device = str(inputs.device)
                if inputs.device.type != "cuda":
                    raise RuntimeError("First ablation training batch was not processed on CUDA")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_probabilities = predict(model, scaled["validation"].sequences, device)
        validation_pr_auc = average_precision_score(sample_sets["validation"].labels, validation_probabilities)
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_pr_auc": float(validation_pr_auc)})
        if validation_pr_auc > best_pr_auc:
            best_pr_auc = float(validation_pr_auc)
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= DEFAULT_PATIENCE:
                break
    duration_seconds = time.perf_counter() - started
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_probabilities = predict(model, scaled["validation"].sequences, device)
    test_probabilities = predict(model, scaled["test"].sequences, device)
    selected_threshold = choose_threshold(sample_sets["validation"].labels, validation_probabilities)
    validation_metrics = metrics(sample_sets["validation"].labels, validation_probabilities, selected_threshold)
    test_metrics = metrics(sample_sets["test"].labels, test_probabilities, selected_threshold)

    joblib.dump(baseline_scaler, output_lstm / "scaler.joblib")
    save_predictions(args.output_dir / "predictions_validation.csv", metadata["validation"], sample_sets["validation"].labels, validation_probabilities)
    save_predictions(args.output_dir / "predictions_test.csv", metadata["test"], sample_sets["test"].labels, test_probabilities)
    write_csv(args.output_dir / "threshold_sweep_validation.csv", threshold_rows(sample_sets["validation"].labels, validation_probabilities))
    write_csv(args.output_dir / "threshold_sweep_test.csv", threshold_rows(sample_sets["test"].labels, test_probabilities))
    with (args.output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_pr_auc"])
        writer.writeheader()
        writer.writerows(history)

    weighted_metrics = json.loads((args.baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    weighted_predictions = args.baseline_dir / "diagnostics_phase4_2"
    weighted_validation = np.genfromtxt(weighted_predictions / "predictions_validation.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    weighted_test = np.genfromtxt(weighted_predictions / "predictions_test.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    weighted_val_probabilities = weighted_validation["lstm_probability"].astype(float)
    weighted_test_probabilities = weighted_test["lstm_probability"].astype(float)
    weighted_val_labels = weighted_validation["target"].astype(int)
    weighted_test_labels = weighted_test["target"].astype(int)
    weighted_threshold = weighted_metrics["lstm"]["threshold"]
    weighted_validation_metrics = metrics(weighted_val_labels, weighted_val_probabilities, weighted_threshold)
    weighted_test_metrics = metrics(weighted_test_labels, weighted_test_probabilities, weighted_threshold)
    comparison = {
        "weighted_lstm": {"validation": weighted_validation_metrics, "test": weighted_test_metrics, "best_epoch": 9, "training_duration_seconds": None, "pos_weight": 7.209185102212265},
        "unweighted_lstm": {"validation": validation_metrics, "test": test_metrics, "best_epoch": best_epoch, "training_duration_seconds": duration_seconds, "pos_weight": 1.0},
    }
    config = {
        "experiment": "phase4.3_class_weight_ablation",
        "only_changed_variable": "pos_weight",
        "pos_weight": 1.0,
        "input_shape": [6, 157],
        "feature_count": 157,
        "target": "future_attack_within_horizon",
        "counts": {split: {"samples": len(sample_sets[split].labels), "positive": int(sample_sets[split].labels.sum()), "negative": int((sample_sets[split].labels == 0).sum())} for split in ("train", "validation", "test")},
        "hidden_size": DEFAULT_HIDDEN_SIZE, "num_layers": DEFAULT_NUM_LAYERS, "dropout": DEFAULT_DROPOUT,
        "batch_size": DEFAULT_BATCH_SIZE, "learning_rate": 1e-3, "max_epochs": DEFAULT_EPOCHS, "patience": DEFAULT_PATIENCE,
        "parameter_count": parameter_count, "gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__, "model_device": str(next(model.parameters()).device), "first_batch_device": first_batch_device,
        "training_duration_seconds": duration_seconds, "best_epoch": best_epoch, "selected_validation_threshold": selected_threshold,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps({"validation": validation_metrics, "test": test_metrics, "comparison": comparison}, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "confusion_matrix.csv", [{"model": "unweighted_lstm", **{key: test_metrics[key] for key in ("tp", "tn", "fp", "fn")}}, {"model": "weighted_lstm", **{key: weighted_test_metrics[key] for key in ("tp", "tn", "fp", "fn")}}])
    report = ["# Phase 4.3 Class-Weight Ablation", "", "Only `pos_weight` changed from the Phase 4 weighted LSTM. No raw data, temporal data, split, architecture, scaler methodology, optimizer, learning rate, batch size, or existing artifact was modified.", "", "## Comparison", "", "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN | Best epoch | Duration (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in comparison.items():
        for split in ("validation", "test"):
            row = result[split]
            report.append(f"| {name} | {split} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {row['fpr']:.6f} | {row['threshold']:.12g} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | {result['best_epoch']} | {result['training_duration_seconds'] if result['training_duration_seconds'] is not None else 'not recorded'} |")
    report += ["", "## Diagnostic interpretation", "", f"- Removing class weighting changed the test false-positive count from {weighted_test_metrics['fp']} to {test_metrics['fp']} at each model's validation-selected threshold.", f"- Test recall changed from {weighted_test_metrics['recall']:.6f} to {test_metrics['recall']:.6f}; test F1 changed from {weighted_test_metrics['f1']:.6f} to {test_metrics['f1']:.6f}.", f"- Test PR-AUC changed from {weighted_test_metrics['pr_auc']:.6f} to {test_metrics['pr_auc']:.6f}; test ROC-AUC changed from {weighted_test_metrics['roc_auc']:.6f} to {test_metrics['roc_auc']:.6f}.", "- The probability distribution and threshold behavior should be interpreted with the saved prediction files and diagnostic threshold sweeps; no test threshold was selected.", "- No single metric alone determines the better practical forecasting model.", "", "## Conclusion", "", "The class-weight ablation isolates whether the original training weight materially contributed to false-positive behavior. The unweighted model's validation-selected threshold is the only threshold used for its final test evaluation.", ""]
    (args.output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "device": str(device), "parameter_count": parameter_count, "best_epoch": best_epoch, "duration_seconds": duration_seconds, "selected_threshold": selected_threshold, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()