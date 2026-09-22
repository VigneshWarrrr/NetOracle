"""Train and evaluate the Phase 4 forecasting baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HISTORY_WINDOWS = 6
FORECAST_HORIZON_WINDOWS = 6
DEFAULT_BATCH_SIZE = 128
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 25
DEFAULT_PATIENCE = 5
SEED = 42
META_COLUMNS = {
    "window_start", "source_file", "split", "current_attack", "current_attack_types",
    "future_attack_within_horizon", "future_attack_types", "history_available",
    "history_window_count", "forecast_sample_eligible",
}


@dataclass
class SampleSet:
    sequences: np.ndarray
    labels: np.ndarray


class ForecastLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(inputs)
        return self.output(self.dropout(states[:, -1, :])).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_samples(windows_dir: Path) -> tuple[dict[str, SampleSet], list[str], list[str]]:
    partitions = sorted(windows_dir.glob("*.csv"))
    if len(partitions) != 10:
        raise ValueError(f"Expected 10 temporal partitions, found {len(partitions)}")

    samples: dict[str, list[tuple[np.ndarray, int]]] = {"train": [], "validation": [], "test": []}
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
            if split not in samples:
                raise ValueError(f"Unexpected split {split!r} in {partition.name}")
            history_rows = rows[index - HISTORY_WINDOWS + 1:index + 1]
            sequence = np.asarray(
                [[float(history_row[column]) for column in feature_columns] for history_row in history_rows],
                dtype=np.float32,
            )
            label = int(row["future_attack_within_horizon"])
            samples[split].append((sequence, label))

    if feature_columns is None or len(feature_columns) != 157:
        raise ValueError(f"Expected 157 model features, found {len(feature_columns or [])}")
    output = {
        split: SampleSet(
            sequences=np.stack([item[0] for item in values]),
            labels=np.asarray([item[1] for item in values], dtype=np.int64),
        )
        for split, values in samples.items()
    }
    return output, feature_columns, source_files


def fit_scaler(train: SampleSet) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train.sequences.reshape(-1, train.sequences.shape[-1]))
    return scaler


def scale_samples(samples: SampleSet, scaler: StandardScaler) -> SampleSet:
    scaled = scaler.transform(samples.sequences.reshape(-1, samples.sequences.shape[-1]))
    return SampleSet(scaled.reshape(samples.sequences.shape).astype(np.float32), samples.labels)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    best = (float("-inf"), float("inf"))
    for threshold in candidates:
        score = f1_score(labels, probabilities >= threshold, zero_division=0)
        candidate = (float(score), -float(threshold))
        if candidate > best:
            best = candidate
    return float(-best[1])


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "threshold": float(threshold),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def predict_lstm(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            values.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
    return np.concatenate(values)


def train_lstm(
    train: SampleSet,
    validation: SampleSet,
    model_dir: Path,
    config: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, float]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; LSTM training is intentionally stopped")
    device = torch.device("cuda:0")
    model = ForecastLSTM(157, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT).to(device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("LSTM model did not move to CUDA")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train.sequences), torch.from_numpy(train.labels.astype(np.float32))),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.sequences), torch.from_numpy(validation.labels.astype(np.float32))),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    positive = int(train.labels.sum())
    negative = int(len(train.labels) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    memory_input_mib = DEFAULT_BATCH_SIZE * HISTORY_WINDOWS * 157 * 4 / (1024 ** 2)
    config["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "available_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model_device": str(next(model.parameters()).device),
        "parameter_count": parameter_count,
        "batch_input_memory_mib_fp32": memory_input_mib,
        "input_shape": [DEFAULT_BATCH_SIZE, HISTORY_WINDOWS, 157],
    }
    first_batch_device: str | None = None
    best_pr_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    best_path = model_dir / "best_model.pt"
    for epoch in range(1, DEFAULT_EPOCHS + 1):
        model.train()
        losses = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if first_batch_device is None:
                first_batch_device = str(inputs.device)
                if inputs.device.type != "cuda":
                    raise RuntimeError("First LSTM training batch was not processed on CUDA")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_probabilities = predict_lstm(model, validation_loader, device)
        validation_pr_auc = average_precision_score(validation.labels, validation_probabilities)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_pr_auc": float(validation_pr_auc),
        }
        history.append(epoch_record)
        if validation_pr_auc > best_pr_auc:
            best_pr_auc = float(validation_pr_auc)
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= DEFAULT_PATIENCE:
                break
    config["gpu"]["first_training_batch_device"] = first_batch_device
    config["best_epoch"] = best_epoch
    config["epochs_completed"] = len(history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_model_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.sequences), torch.from_numpy(validation.labels.astype(np.float32))),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
    )
    validation_probabilities = predict_lstm(model, test_model_loader, device)
    threshold = choose_threshold(validation.labels, validation_probabilities)
    return {"model": model, "threshold": threshold, "best_validation_pr_auc": best_pr_auc, "best_epoch": best_epoch}, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_baseline")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {args.output_dir}")
    set_seed(SEED)
    sample_sets, feature_columns, source_files = read_samples(args.windows_dir)
    train, validation, test = (sample_sets[split] for split in ("train", "validation", "test"))
    scaler = fit_scaler(train)
    scaled_train, scaled_validation, scaled_test = (scale_samples(item, scaler) for item in (train, validation, test))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "logistic_regression").mkdir()
    (args.output_dir / "lstm").mkdir()
    joblib.dump(scaler, args.output_dir / "logistic_regression/scaler.joblib")
    joblib.dump(scaler, args.output_dir / "lstm/scaler.joblib")
    config: dict[str, object] = {
        "seed": SEED,
        "source_files": source_files,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "history_windows": HISTORY_WINDOWS,
        "forecast_horizon_windows": FORECAST_HORIZON_WINDOWS,
        "input_shape": [HISTORY_WINDOWS, len(feature_columns)],
        "target": "future_attack_within_horizon",
        "target_definition": "any current_attack in t+1 through t+6; current t excluded",
        "counts": {split: {"samples": len(sample_sets[split].labels), "positive": int(sample_sets[split].labels.sum()), "negative": int((sample_sets[split].labels == 0).sum())} for split in ("train", "validation", "test")},
        "scaling": "StandardScaler fitted on six-state training histories only",
        "logistic": {"representation": "flattened six-state history", "class_weight": "training balanced weights", "max_iter": 2000},
        "lstm": {"hidden_size": DEFAULT_HIDDEN_SIZE, "num_layers": DEFAULT_NUM_LAYERS, "dropout": DEFAULT_DROPOUT, "batch_size": DEFAULT_BATCH_SIZE, "learning_rate": DEFAULT_LEARNING_RATE, "max_epochs": DEFAULT_EPOCHS, "patience": DEFAULT_PATIENCE, "loss": "BCEWithLogitsLoss", "pos_weight": float((len(train.labels) - train.labels.sum()) / train.labels.sum())},
    }
    write_json(args.output_dir / "config.json", config)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    logistic.fit(scaled_train.sequences.reshape(len(scaled_train.labels), -1), scaled_train.labels)
    joblib.dump(logistic, args.output_dir / "logistic_regression/model.joblib")
    validation_probabilities = logistic.predict_proba(scaled_validation.sequences.reshape(len(scaled_validation.labels), -1))[:, 1]
    logistic_threshold = choose_threshold(validation.labels, validation_probabilities)
    logistic_probabilities = logistic.predict_proba(scaled_test.sequences.reshape(len(scaled_test.labels), -1))[:, 1]
    logistic_metrics = calculate_metrics(test.labels, logistic_probabilities, logistic_threshold)
    lstm_result, training_history = train_lstm(scaled_train, scaled_validation, args.output_dir / "lstm", config)
    with (args.output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_pr_auc"])
        writer.writeheader()
        writer.writerows(training_history)
    lstm_model = lstm_result["model"]
    test_loader = DataLoader(TensorDataset(torch.from_numpy(scaled_test.sequences), torch.from_numpy(scaled_test.labels.astype(np.float32))), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    lstm_probabilities = predict_lstm(lstm_model, test_loader, torch.device("cuda"))
    lstm_metrics = calculate_metrics(test.labels, lstm_probabilities, float(lstm_result["threshold"]))
    metrics = {"logistic_regression": logistic_metrics, "lstm": lstm_metrics}
    write_json(args.output_dir / "metrics.json", metrics)
    with (args.output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "tn", "fp", "fn", "tp"])
        writer.writeheader()
        for name, result in metrics.items():
            writer.writerow({"model": name, **result["confusion_matrix"]})
    report = ["# Phase 4 Baseline Comparison", "", "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in metrics.items():
        cm = result["confusion_matrix"]
        report.append(f"| {name} | {result['precision']:.6f} | {result['recall']:.6f} | {result['f1']:.6f} | {result['pr_auc']:.6f} | {result['roc_auc']:.6f} | {result['false_positive_rate']:.6f} | {result['threshold']:.6f} | {cm['tp']} | {cm['tn']} | {cm['fp']} | {cm['fn']} |")
    report.extend(["", "Thresholds were selected using validation predictions only. The test set was used once for final evaluation.", "", "The better model should be determined from the reported forecasting metrics, with PR-AUC prioritized because the target is imbalanced."])
    (args.output_dir / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.output_dir / "config.json", config)
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()