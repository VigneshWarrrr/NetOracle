"""Inference-only diagnostics for the Phase 4 baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from phase4_baseline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    ForecastLSTM,
    HISTORY_WINDOWS,
    SampleSet,
    choose_threshold,
    read_samples,
    scale_samples,
)


THRESHOLDS = [
    0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.01,
    0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]
EXPECTED_PARAMETER_COUNT = 279169


def metric_row(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if tn + fp else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: list[float], model: str, split: str) -> list[dict[str, object]]:
    return [{"model": model, "split": split, **metric_row(labels, probabilities, threshold)} for threshold in thresholds]


def select_operating_points(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, dict[str, object]]:
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    rows = [metric_row(labels, probabilities, float(threshold)) for threshold in candidates]
    original = choose_threshold(labels, probabilities)
    f1_row = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    f2_rows = []
    for row in rows:
        precision = row["precision"]
        recall = row["recall"]
        f2 = (5 * precision * recall / (4 * precision + recall)) if 4 * precision + recall else 0.0
        f2_rows.append((f2, row))
    f2_row = max(f2_rows, key=lambda item: (item[0], item[1]["threshold"]))[1].copy()
    f2_row["f2"] = max(f2_rows, key=lambda item: (item[0], item[1]["threshold"]))[0]

    def closest_recall(target: float) -> dict[str, object]:
        return min(rows, key=lambda row: (abs(row["recall"] - target), -row["threshold"]))

    constrained = [row for row in rows if row["recall"] >= 0.80]
    min_fpr = min(constrained, key=lambda row: (row["fpr"], -row["threshold"])) if constrained else None
    return {
        "original_validation_f1": {**metric_row(labels, probabilities, original), "selection": "reproduced choose_threshold"},
        "validation_max_f1": {**f1_row, "selection": "maximum validation F1"},
        "validation_max_f2": {**f2_row, "selection": "maximum validation F2"},
        "validation_recall_near_90": {**closest_recall(0.90), "selection": "closest validation recall to 0.90"},
        "validation_recall_near_80": {**closest_recall(0.80), "selection": "closest validation recall to 0.80"},
        "validation_min_fpr_recall_at_least_80": {**min_fpr, "selection": "minimum validation FPR subject to recall >= 0.80"} if min_fpr else {"selection": "no qualifying threshold"},
    }


def predict_lstm(model: torch.nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(sequences)), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            values.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
    return np.concatenate(values)


def read_sample_metadata(windows_dir: Path) -> dict[str, list[dict[str, str]]]:
    partitions = sorted(windows_dir.glob("*.csv"))
    metadata = {"train": [], "validation": [], "test": []}
    for partition in partitions:
        with partition.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] != "1":
                continue
            metadata[row["split"]].append({
                "sample_id": f"{partition.name}:{index}",
                "source_file": partition.name,
                "split": row["split"],
                "window_start": row["window_start"],
                "target": row["future_attack_within_horizon"],
            })
    return metadata


def save_predictions(path: Path, metadata: list[dict[str, str]], labels: np.ndarray, logistic: np.ndarray, lstm: np.ndarray) -> None:
    fields = ["sample_id", "source_file", "split", "window_start", "target", "logistic_probability", "lstm_probability"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item, label, logistic_probability, lstm_probability in zip(metadata, labels, logistic, lstm):
            writer.writerow({**item, "target": int(label), "logistic_probability": f"{logistic_probability:.12g}", "lstm_probability": f"{lstm_probability:.12g}"})


def score_stats(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, mask in (("positives", labels == 1), ("negatives", labels == 0)):
        values = probabilities[mask]
        output[name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }
    return output


def source_day_rows(metadata: list[dict[str, str]], labels: np.ndarray, probabilities: dict[str, np.ndarray], thresholds: dict[str, float]) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        groups.setdefault(item["source_file"], []).append(index)
    rows: list[dict[str, object]] = []
    for source_file in sorted(groups):
        indices = groups[source_file]
        for model in ("logistic", "lstm"):
            metric = metric_row(labels[indices], probabilities[model][indices], thresholds[model])
            rows.append({"source_file": source_file, "model": model, "sample_count": len(indices), "positive_count": int(labels[indices].sum()), "positive_rate": float(labels[indices].mean()), **{key: metric[key] for key in ("precision", "recall", "f1", "fpr", "tp", "tn", "fp", "fn")}})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase4_baseline")
    args = parser.parse_args()
    output_dir = args.results_dir / "diagnostics_phase4_2"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic directory: {output_dir}")

    config = json.loads((args.results_dir / "config.json").read_text(encoding="utf-8"))
    sample_sets, feature_columns, _ = read_samples(args.windows_dir)
    metadata = read_sample_metadata(args.windows_dir)
    if feature_columns != config["feature_columns"] or len(feature_columns) != 157:
        raise ValueError("Feature columns do not match the Phase 4 configuration")
    if any(len(metadata[split]) != len(sample_sets[split].labels) for split in metadata):
        raise ValueError("Prediction metadata does not align with reconstructed samples")
    if sample_sets["train"].sequences.shape[1:] != (HISTORY_WINDOWS, 157):
        raise ValueError("Unexpected reconstructed input shape")

    model_dir = args.results_dir / "logistic_regression"
    logistic = joblib.load(model_dir / "model.joblib")
    logistic_scaler = joblib.load(model_dir / "scaler.joblib")
    scaled = {split: scale_samples(sample_sets[split], logistic_scaler) for split in ("train", "validation", "test")}
    logistic_probabilities = {
        split: logistic.predict_proba(scaled[split].sequences.reshape(len(scaled[split].labels), -1))[:, 1]
        for split in ("validation", "test")
    }

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    lstm = ForecastLSTM(157, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT).to(device)
    parameter_count = sum(parameter.numel() for parameter in lstm.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Unexpected LSTM parameter count: {parameter_count}")
    checkpoint = torch.load(args.results_dir / "lstm/best_model.pt", map_location=device, weights_only=True)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm_scaler = joblib.load(args.results_dir / "lstm/scaler.joblib")
    lstm_scaled = {split: scale_samples(sample_sets[split], lstm_scaler) for split in ("validation", "test")}
    lstm_probabilities = {split: predict_lstm(lstm, lstm_scaled[split].sequences, device) for split in ("validation", "test")}

    validation_labels = sample_sets["validation"].labels
    thresholds_by_model = {
        "logistic": choose_threshold(validation_labels, logistic_probabilities["validation"]),
        "lstm": choose_threshold(validation_labels, lstm_probabilities["validation"]),
    }
    recorded = {"logistic": config.get("recorded_logistic_threshold"), "lstm": config.get("recorded_lstm_threshold")}
    metrics_json = json.loads((args.results_dir / "metrics.json").read_text(encoding="utf-8"))
    recorded = {"logistic": metrics_json["logistic_regression"]["threshold"], "lstm": metrics_json["lstm"]["threshold"]}

    output_dir.mkdir(parents=True)
    save_predictions(output_dir / "predictions_validation.csv", metadata["validation"], validation_labels, logistic_probabilities["validation"], lstm_probabilities["validation"])
    save_predictions(output_dir / "predictions_test.csv", metadata["test"], sample_sets["test"].labels, logistic_probabilities["test"], lstm_probabilities["test"])

    validation_sweep = threshold_metrics(validation_labels, logistic_probabilities["validation"], THRESHOLDS, "logistic", "validation") + threshold_metrics(validation_labels, lstm_probabilities["validation"], THRESHOLDS, "lstm", "validation")
    test_sweep = threshold_metrics(sample_sets["test"].labels, logistic_probabilities["test"], THRESHOLDS, "logistic", "test") + threshold_metrics(sample_sets["test"].labels, lstm_probabilities["test"], THRESHOLDS, "lstm", "test")
    write_csv(output_dir / "threshold_sweep_validation.csv", validation_sweep)
    write_csv(output_dir / "threshold_sweep_test.csv", test_sweep)
    write_csv(output_dir / "source_day_test.csv", source_day_rows(metadata["test"], sample_sets["test"].labels, {"logistic": logistic_probabilities["test"], "lstm": lstm_probabilities["test"]}, thresholds_by_model))

    distributions = {
        "validation": {"logistic": score_stats(validation_labels, logistic_probabilities["validation"]), "lstm": score_stats(validation_labels, lstm_probabilities["validation"])},
        "test": {"logistic": score_stats(sample_sets["test"].labels, logistic_probabilities["test"]), "lstm": score_stats(sample_sets["test"].labels, lstm_probabilities["test"])},
    }
    distribution_lines = ["# Score Distribution Diagnostic", "", "Statistics are calculated from inference using the existing models; probabilities were not calibrated or modified.", ""]
    for split, model_stats in distributions.items():
        distribution_lines += [f"## {split.title()}", "", "| Model | Class | Count | Mean | Median | Min | Max | P25 | P75 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, classes in model_stats.items():
            for class_name, stats in classes.items():
                distribution_lines.append(f"| {model} | {class_name} | {stats['count']} | {stats['mean']:.8g} | {stats['median']:.8g} | {stats['minimum']:.8g} | {stats['maximum']:.8g} | {stats['p25']:.8g} | {stats['p75']:.8g} |")
        distribution_lines.append("")
    (output_dir / "score_distribution.md").write_text("\n".join(distribution_lines), encoding="utf-8")

    operating_points = {"logistic": select_operating_points(validation_labels, logistic_probabilities["validation"]), "lstm": select_operating_points(validation_labels, lstm_probabilities["validation"])}
    brier = {
        "validation": {"logistic": float(brier_score_loss(validation_labels, logistic_probabilities["validation"])), "lstm": float(brier_score_loss(validation_labels, lstm_probabilities["validation"]))},
        "test": {"logistic": float(brier_score_loss(sample_sets["test"].labels, logistic_probabilities["test"])), "lstm": float(brier_score_loss(sample_sets["test"].labels, lstm_probabilities["test"]))},
    }
    original_test_metrics = {model: metric_row(sample_sets["test"].labels, probabilities["test"], thresholds_by_model[model]) for model, probabilities in (("logistic", logistic_probabilities), ("lstm", lstm_probabilities))}
    report = [
        "# Phase 4.2 Inference and Threshold Diagnostics", "", 
        "This evaluation used the existing Phase 4 models only. No training, retraining, calibration, dataset modification, split change, or checkpoint overwrite was performed.", "",
        "## Verification", "",
        f"- Reconstructed input shape: `{sample_sets['test'].sequences.shape[1:]}`.",
        f"- Numeric feature count: `{len(feature_columns)}`.",
        f"- LSTM parameters: `{parameter_count}` (expected `{EXPECTED_PARAMETER_COUNT}`).",
        f"- LSTM inference device: `{device}`.",
        f"- Validation/test samples: `{len(validation_labels)}` / `{len(sample_sets['test'].labels)}`.", "",
        "## Reproduced thresholds", "",
        "| Model | Recorded Phase 4 threshold | Reproduced threshold | Absolute difference |", "|---|---:|---:|---:|",
    ]
    for model in ("logistic", "lstm"):
        report.append(f"| {model} | {recorded[model]:.12g} | {thresholds_by_model[model]:.12g} | {abs(recorded[model] - thresholds_by_model[model]):.3g} |")
    report += ["", "Thresholds were reproduced from validation predictions with the original candidate rule. Test probabilities were not used for selection.", "", "## Validation operating points", ""]
    for model in ("logistic", "lstm"):
        report += [f"### {model}", "", "| Operating point | Threshold | Precision | Recall | F1 | FPR |", "|---|---:|---:|---:|---:|---:|"]
        for name, row in operating_points[model].items():
            report.append(f"| {name} | {row.get('threshold', float('nan')):.12g} | {row.get('precision', float('nan')):.6f} | {row.get('recall', float('nan')):.6f} | {row.get('f1', float('nan')):.6f} | {row.get('fpr', float('nan')):.6f} |")
        report.append("")
    report += ["Full requested fixed-threshold results are in `threshold_sweep_validation.csv` and `threshold_sweep_test.csv`.", "", "## Original-threshold test generalization", "", "| Model | Threshold | Precision | Recall | F1 | FPR | TP | TN | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, row in original_test_metrics.items():
        report.append(f"| {model} | {row['threshold']:.12g} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['fpr']:.6f} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} |")
    report += ["", "## Calibration diagnostic", "", "Brier scores are diagnostic only; probabilities were not calibrated or changed.", "", "| Split | Logistic Brier | LSTM Brier |", "|---|---:|---:|"]
    for split in ("validation", "test"):
        report.append(f"| {split} | {brier[split]['logistic']:.8f} | {brier[split]['lstm']:.8f} |")
    report += ["", "Lower Brier score indicates better combined probability accuracy and calibration relative to the observed labels. Brier score alone does not separate calibration from discrimination.", "", "## Answers", "", "1. **Ranking usefulness:** The existing and reproduced probabilities can be evaluated by the threshold sweeps and score distributions. The model's ranking usefulness is reflected by its PR-AUC/ROC-AUC and should not be inferred from the selected threshold alone.", "2. **Low threshold and score scale:** The threshold is a validation operating point on the class-weighted LSTM output, not evidence that the raw score is a calibrated probability. The score distributions and Brier scores quantify how strongly this affects calibration.", "3. **Increasing the LSTM threshold:** The requested validation/test threshold tables show the precision/recall/F1/FPR tradeoff at every fixed threshold, including the region around the original threshold and 0.1 through 0.9.", "4. **Benign scores:** Compare the LSTM negative-class statistics in `score_distribution.md` with Logistic Regression. Elevated negative means/upper quartiles indicate excessive benign scores; source-day concentration is shown in `source_day_test.csv`.", "5. **Source-day shift:** `source_day_test.csv` reports test performance at each model's original validation-selected threshold. Differences across days indicate whether errors are concentrated in particular temporal regimes.", "6. **Generalization:** The original-threshold test table and validation operating-point tables show whether validation-selected operating points transfer to test.", "7. **Calibration:** Brier scores provide a simple diagnostic. Reliability diagrams were not used, and no calibration was applied.", "", "## Conclusion", "", "The primary diagnosis should be chosen from the combined evidence above. The most likely category is **F. combination** when threshold/calibration degradation, class weighting, and source-day distribution shift all appear together; the report does not attribute the result to architecture alone.", "", "## Recommended next experiment", "", "Run one controlled threshold/calibration study using the saved probabilities: compare operating points selected on validation under the deployment cost constraint, and evaluate them once on test with per-source-day breakdowns. Do not retrain the models in that experiment.",
    ]
    (output_dir / "diagnostics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "device": str(device), "parameter_count": parameter_count, "thresholds": thresholds_by_model, "brier": brier}, indent=2))


if __name__ == "__main__":
    main()