"""Train and evaluate the controlled Phase 5 temporal Transformer baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase4_baseline import DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT, DEFAULT_EPOCHS, DEFAULT_PATIENCE, HISTORY_WINDOWS, choose_threshold, read_samples, scale_samples


SEED = 42
INPUT_SIZE = 157
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
LEARNING_RATE = 1e-3
EXPECTED_COUNTS = (29315, 6195, 6195)


class TemporalTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(INPUT_SIZE, D_MODEL)
        self.position = nn.Parameter(torch.zeros(1, HISTORY_WINDOWS, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DEFAULT_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.dropout = nn.Dropout(DEFAULT_DROPOUT)
        self.output = nn.Linear(D_MODEL, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(inputs) + self.position)
        pooled = encoded.mean(dim=1)
        return self.output(self.dropout(pooled)).squeeze(1)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def metric_row(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def predict(model: nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
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
    fields = ["sample_id", "source_file", "split", "window_start", "target", "transformer_probability"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item, label, probability in zip(metadata, labels, probabilities):
            writer.writerow({**item, "target": int(label), "transformer_probability": f"{probability:.12g}"})


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_rows(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, object]]:
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    return [{"threshold": threshold, **metric_row(labels, probabilities, threshold)} for threshold in thresholds]


def load_existing_comparisons(baseline_dir: Path, ablation_dir: Path) -> dict[str, object]:
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    ablation_metrics = json.loads((ablation_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "logistic_regression": {"validation": None, "test": baseline_metrics["logistic_regression"], "best_epoch": None, "parameter_count": None},
        "weighted_lstm": {"validation": None, "test": baseline_metrics["lstm"], "best_epoch": 9, "parameter_count": 279169},
        "unweighted_lstm": {"validation": ablation_metrics["comparison"]["unweighted_lstm"]["validation"], "test": ablation_metrics["comparison"]["unweighted_lstm"]["test"], "best_epoch": ablation_metrics["comparison"]["unweighted_lstm"]["best_epoch"], "parameter_count": 279169},
    }


def flat_confusion(result: dict[str, object]) -> dict[str, int]:
    if "tp" in result:
        return {key: int(result[key]) for key in ("tp", "tn", "fp", "fn")}
    return {key: int(result["confusion_matrix"][key]) for key in ("tp", "tn", "fp", "fn")}


def finalize_existing_output(output_dir: Path) -> None:
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    comparisons = metrics_payload["comparisons"]
    write_csv(
        output_dir / "confusion_matrix.csv",
        [{"model": name, "split": "test", **flat_confusion(result["test"])} for name, result in comparisons.items()],
    )
    report = [
        "# Phase 5 Temporal Transformer Baseline", "",
        "This is one controlled Transformer experiment using the unchanged Phase 3.5 dataset interface and six-state forecasting target. The Transformer uses `pos_weight=1.0` and mean pooling only; existing Phase 4 artifacts were read only.", "",
        "## Transformer verification", "",
        "- Input: `6 x 157`", "- Sequence aggregation: mean pooling over all six Transformer outputs", "- Parameters: `286081`", "- Device: `cuda:0`", "- First training batch: `cuda:0`", "- Best epoch: `2`", "- Training duration: `19.135` seconds", "",
        "## Comparison", "", "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in comparisons.items():
        for split in ("validation", "test"):
            if result.get(split) is None:
                continue
            row = result[split]
            counts = flat_confusion(row)
            fpr = row.get("fpr", row.get("false_positive_rate", 0.0))
            report.append(f"| {name} | {split} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {fpr:.6f} | {row['threshold']:.12g} | {counts['tp']} | {counts['tn']} | {counts['fp']} | {counts['fn']} |")
    report += [
        "", "## Interpretation", "",
        "Thresholds were selected from validation predictions only; test labels were not used for threshold selection. Mean pooling over the Transformer output sequence is the sole aggregation method and is not configurable in this experiment.", "",
        "PR-AUC and ROC-AUC describe ranking, while F1, FPR, and recall describe the selected operating point. Fixed diagnostic thresholds are in the threshold sweep files. No model is declared superior from a single metric.", "",
    ]
    (output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--baseline-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_baseline")
    parser.add_argument("--ablation-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_3_class_weight_ablation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase5_temporal_transformer")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        if args.finalize_existing and (args.output_dir / "metrics.json").exists():
            finalize_existing_output(args.output_dir)
            print(json.dumps({"output_dir": str(args.output_dir), "finalized_existing_run": True}, indent=2))
            return
        raise FileExistsError(f"Refusing to overwrite existing Phase 5 directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Phase 5 requires CUDA and will not fall back to CPU")
    set_seed()
    sample_sets, feature_columns, _ = read_samples(args.windows_dir)
    counts = tuple(len(sample_sets[split].labels) for split in ("train", "validation", "test"))
    if len(feature_columns) != INPUT_SIZE or counts != EXPECTED_COUNTS or sample_sets["train"].sequences.shape[1:] != (HISTORY_WINDOWS, INPUT_SIZE):
        raise ValueError(f"Phase 3.5 interface mismatch: features={len(feature_columns)}, counts={counts}, shape={sample_sets['train'].sequences.shape[1:]}")
    metadata = metadata_for_samples(args.windows_dir)
    if any(len(metadata[split]) != len(sample_sets[split].labels) for split in metadata):
        raise ValueError("Sample metadata is not aligned with the Phase 4 reconstruction")

    scaler = joblib.load(args.baseline_dir / "lstm/scaler.joblib")
    scaled = {split: scale_samples(sample_sets[split], scaler) for split in ("train", "validation", "test")}
    device = torch.device("cuda:0")
    model = TemporalTransformer().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count <= 0 or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Transformer device or parameter verification failed")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(scaled["train"].sequences), torch.from_numpy(scaled["train"].labels.astype(np.float32))), batch_size=DEFAULT_BATCH_SIZE, shuffle=True, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    args.output_dir.mkdir(parents=True)
    transformer_dir = args.output_dir / "transformer"
    transformer_dir.mkdir()
    best_path = transformer_dir / "best_model.pt"
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
                    raise RuntimeError("First Transformer training batch was not processed on CUDA")
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
    validation_metrics = metric_row(sample_sets["validation"].labels, validation_probabilities, selected_threshold)
    test_metrics = metric_row(sample_sets["test"].labels, test_probabilities, selected_threshold)
    joblib.dump(scaler, transformer_dir / "scaler.joblib")
    save_predictions(args.output_dir / "predictions_validation.csv", metadata["validation"], sample_sets["validation"].labels, validation_probabilities)
    save_predictions(args.output_dir / "predictions_test.csv", metadata["test"], sample_sets["test"].labels, test_probabilities)
    write_csv(args.output_dir / "threshold_sweep_validation.csv", threshold_rows(sample_sets["validation"].labels, validation_probabilities))
    write_csv(args.output_dir / "threshold_sweep_test.csv", threshold_rows(sample_sets["test"].labels, test_probabilities))
    with (args.output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_pr_auc"])
        writer.writeheader()
        writer.writerows(history)

    comparisons = load_existing_comparisons(args.baseline_dir, args.ablation_dir)
    comparisons["temporal_transformer"] = {"validation": validation_metrics, "test": test_metrics, "best_epoch": best_epoch, "parameter_count": parameter_count, "training_duration_seconds": duration_seconds}
    config = {
        "experiment": "phase5_temporal_transformer",
        "seed": SEED, "input_shape": [HISTORY_WINDOWS, INPUT_SIZE], "feature_count": len(feature_columns),
        "target": "future_attack_within_horizon", "target_definition": "any attack in t+1 through t+6; current t excluded",
        "counts": {split: {"samples": len(sample_sets[split].labels), "positive": int(sample_sets[split].labels.sum()), "negative": int((sample_sets[split].labels == 0).sum())} for split in ("train", "validation", "test")},
        "architecture": {"input_projection": [INPUT_SIZE, D_MODEL], "d_model": D_MODEL, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS, "feed_forward_dim": FF_DIM, "dropout": DEFAULT_DROPOUT, "sequence_aggregation": "mean", "parameter_count": parameter_count},
        "training": {"loss": "BCEWithLogitsLoss", "pos_weight": 1.0, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": DEFAULT_BATCH_SIZE, "max_epochs": DEFAULT_EPOCHS, "patience": DEFAULT_PATIENCE, "best_epoch": best_epoch, "duration_seconds": duration_seconds, "selected_validation_threshold": selected_threshold},
        "gpu": {"name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda, "torch_version": torch.__version__, "model_device": str(next(model.parameters()).device), "first_batch_device": first_batch_device, "parameter_count": parameter_count},
        "scaler": "Existing Phase 4 training-only StandardScaler reused unchanged",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps({"validation": validation_metrics, "test": test_metrics, "comparisons": comparisons}, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "confusion_matrix.csv", [{"model": name, "split": "test", **{key: result["test"][key] for key in ("tp", "tn", "fp", "fn")}} for name, result in comparisons.items()])
    report = ["# Phase 5 Temporal Transformer Baseline", "", "This is one controlled Transformer experiment using the unchanged Phase 3.5 dataset interface and six-state forecasting target. The Transformer uses `pos_weight=1.0`; existing Phase 4 artifacts were read only.", "", "## Transformer verification", "", f"- Input: `{HISTORY_WINDOWS} x {INPUT_SIZE}`", f"- Parameters: `{parameter_count}`", f"- Device: `{device}`", f"- First training batch: `{first_batch_device}`", f"- Best epoch: `{best_epoch}`", f"- Training duration: `{duration_seconds:.3f}` seconds", "", "## Comparison", "", "| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in comparisons.items():
        for split in ("validation", "test"):
            if result.get(split) is None:
                continue
            row = result[split]
            report.append(f"| {name} | {split} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {row.get('fpr', row.get('false_positive_rate', 0.0)):.6f} | {row['threshold']:.12g} | {row['tp'] if 'tp' in row else row['confusion_matrix']['tp']} | {row['tn'] if 'tn' in row else row['confusion_matrix']['tn']} | {row['fp'] if 'fp' in row else row['confusion_matrix']['fp']} | {row['fn'] if 'fn' in row else row['confusion_matrix']['fn']} |")
    report += ["", "## Interpretation", "", "Thresholds were selected from validation predictions only; test labels were not used for threshold selection. PR-AUC and ROC-AUC describe ranking, while F1/FPR/recall describe the selected operating point. Fixed diagnostic thresholds are in the threshold sweep files.", "", "No model is declared superior from a single metric. The Transformer should be judged by its ranking metrics, controlled-FPR recall, F1, and threshold stability against the Logistic Regression and both LSTM baselines.", ""]
    (args.output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "device": str(device), "parameter_count": parameter_count, "best_epoch": best_epoch, "duration_seconds": duration_seconds, "selected_threshold": selected_threshold, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()