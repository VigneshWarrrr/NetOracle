"""Phase 6B: Vector World Model -- learned latent dynamics with future-state rollout.

This experiment reuses the unchanged Phase 3.5 canonical dataset through
`world_model_dataset.read_world_model_samples` (which itself reuses Phase 4's
eligibility/indexing rules unmodified). It trains a genuine World Model:

    S(t-5)...S(t) --encode--> latent history
                  --temporal context (self-attention)--> z(t)
                  --learned residual transition, applied recursively x6--> z(t+1)...z(t+6)
                  --decode each latent--> Y_hat(t+1)...Y_hat(t+6)
                  --attack head over the predicted rollout only--> P(future_attack_within_horizon)

The attack head never sees the ground-truth future states Y in its forward
pass -- Y is used only to compute the training loss (state-reconstruction
term). This file, its results directory, and `world_model_dataset.py` are
the only things this experiment creates or modifies; Phase 3-5 code,
datasets, and results are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from phase4_baseline import HISTORY_WINDOWS, FORECAST_HORIZON_WINDOWS, calculate_metrics, choose_threshold
from world_model_dataset import (
    EXPECTED_FEATURE_COUNT,
    EXPECTED_SPLIT_COUNTS,
    WorldModelSampleSet,
    read_world_model_samples,
    validate_world_model_samples,
)

SEED = 42
INPUT_SIZE = EXPECTED_FEATURE_COUNT  # 157
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 25
PATIENCE = 5
STATE_LOSS_WEIGHT = 1.0
ATTACK_LOSS_WEIGHT = 1.0
SMOKE_BATCH_SIZE = 8


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL: encode -> temporal context -> recursive latent rollout -> decode + attack head
# ============================================================


class StateEncoder(nn.Module):
    """Encodes each per-timestep 157-dim network state into a d_model latent."""

    def __init__(self, input_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalContextEncoder(nn.Module):
    """Self-attention over the six observed latent states S(t-5)...S(t)."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z + self.position)


class LatentTransition(nn.Module):
    """Learned residual dynamics: z(k+1) = z(k) + f(z(k)). Applied recursively for rollout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.delta(z)


class StateDecoder(nn.Module):
    """Decodes a rollout latent back into a predicted 157-dim network state."""

    def __init__(self, d_model: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class AttackForecastHead(nn.Module):
    """One future-attack probability, derived only from the predicted latent rollout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        pooled = z_future.mean(dim=1)
        return self.net(pooled).squeeze(-1)


class VectorWorldModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        history_windows: int = HISTORY_WINDOWS,
        forecast_horizon: int = FORECAST_HORIZON_WINDOWS,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.state_encoder = StateEncoder(input_size, d_model, dropout)
        self.temporal_context = TemporalContextEncoder(d_model, num_heads, num_layers, ff_dim, dropout, history_windows)
        self.transition = LatentTransition(d_model, dropout)
        self.state_decoder = StateDecoder(d_model, input_size)
        self.attack_head = AttackForecastHead(d_model, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, 6, 157] -> (y_hat: [B, 6, 157], attack_logit: [B])."""
        z_hist = self.state_encoder(x)
        context = self.temporal_context(z_hist)
        z = context[:, -1, :]
        rollout = []
        for _ in range(self.forecast_horizon):
            z = self.transition(z)
            rollout.append(z)
        z_future = torch.stack(rollout, dim=1)
        y_hat = self.state_decoder(z_future)
        attack_logit = self.attack_head(z_future)
        return y_hat, attack_logit


# ============================================================
# DATA PREPARATION
# ============================================================


def fit_scaler(train_x: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    return scaler


def scale_array(array: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = array.shape
    return scaler.transform(array.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, label: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(label.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


# ============================================================
# SMOKE TEST -- shape / forward-pass / backward-pass sanity check
# ============================================================


def run_smoke_test(model: nn.Module, device: torch.device, sample_x: np.ndarray) -> dict[str, object]:
    n = min(SMOKE_BATCH_SIZE, sample_x.shape[0])
    batch = torch.from_numpy(sample_x[:n]).to(device)
    target_y = torch.randn(n, FORECAST_HORIZON_WINDOWS, INPUT_SIZE, device=device)
    target_label = torch.randint(0, 2, (n,), dtype=torch.float32, device=device)

    model.train()
    y_hat, attack_logit = model(batch)
    if y_hat.shape != (n, FORECAST_HORIZON_WINDOWS, INPUT_SIZE):
        raise RuntimeError(f"Smoke test failed: Y_hat shape {tuple(y_hat.shape)} != {(n, FORECAST_HORIZON_WINDOWS, INPUT_SIZE)}")
    if attack_logit.shape != (n,):
        raise RuntimeError(f"Smoke test failed: attack_logit shape {tuple(attack_logit.shape)} != {(n,)}")
    if not torch.isfinite(y_hat).all() or not torch.isfinite(attack_logit).all():
        raise RuntimeError("Smoke test failed: non-finite forward-pass output")

    state_loss = nn.functional.mse_loss(y_hat, target_y)
    attack_loss = nn.functional.binary_cross_entropy_with_logits(attack_logit, target_label)
    loss = STATE_LOSS_WEIGHT * state_loss + ATTACK_LOSS_WEIGHT * attack_loss
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if not grad_norms or not all(np.isfinite(g) for g in grad_norms):
        raise RuntimeError("Smoke test failed: no finite gradients reached the parameters")
    model.zero_grad(set_to_none=True)

    return {
        "status": "PASS",
        "batch_size": n,
        "input_shape": [n, HISTORY_WINDOWS, INPUT_SIZE],
        "y_hat_shape": list(y_hat.shape),
        "attack_logit_shape": list(attack_logit.shape),
        "smoke_state_loss": float(state_loss.item()),
        "smoke_attack_loss": float(attack_loss.item()),
        "parameters_with_gradient": len(grad_norms),
    }


# ============================================================
# EVALUATION HELPERS
# ============================================================


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, use_amp: bool, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    y_hats: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            y_hat, attack_logit = model(batch)
        y_hats.append(y_hat.float().cpu().numpy())
        probabilities.append(torch.sigmoid(attack_logit.float()).cpu().numpy())
    return np.concatenate(y_hats), np.concatenate(probabilities)


def per_horizon_state_error(y_hat: np.ndarray, y_true: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for step in range(y_hat.shape[1]):
        diff = y_hat[:, step, :] - y_true[:, step, :]
        rows.append(
            {
                "horizon_step": step + 1,
                "mse_standardized": float(np.mean(diff ** 2)),
                "mae_standardized": float(np.mean(np.abs(diff))),
            }
        )
    return rows


def per_horizon_attack_performance(
    future_attack_step: np.ndarray, probabilities: np.ndarray, threshold: float
) -> list[dict[str, object]]:
    rows = []
    predicted_positive = probabilities >= threshold
    for step in range(0, FORECAST_HORIZON_WINDOWS + 1):
        mask = future_attack_step == step
        count = int(mask.sum())
        if count == 0:
            rate = None
        else:
            rate = float(predicted_positive[mask].mean())
        rows.append(
            {
                "future_attack_step": step,
                "meaning": "no attack in horizon (negative)" if step == 0 else f"earliest attack at t+{step}",
                "sample_count": count,
                "mean_predicted_probability": float(probabilities[mask].mean()) if count else None,
                "predicted_positive_rate_at_selected_threshold": rate,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, sample_set: WorldModelSampleSet, probabilities: np.ndarray) -> None:
    fields = ["source_file", "window_start", "split", "label", "world_model_probability"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source_file, window_start, label, probability in zip(
            sample_set.source_file, sample_set.window_start, sample_set.label, probabilities
        ):
            writer.writerow(
                {
                    "source_file": source_file,
                    "window_start": window_start,
                    "split": "n/a",
                    "label": int(label),
                    "world_model_probability": f"{probability:.12g}",
                }
            )


def load_existing_metrics(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/phase6b_vector_world_model")
    parser.add_argument("--phase4-metrics", type=Path, default=Path(__file__).resolve().parent / "results/phase4_baseline/metrics.json")
    parser.add_argument("--phase5-metrics", type=Path, default=Path(__file__).resolve().parent / "results/phase5_temporal_transformer/metrics.json")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {args.output_dir}")

    set_seed(SEED)

    # -------------------------------------------------------
    # Load the unchanged Phase 3.5 interface via world_model_dataset.py
    # -------------------------------------------------------
    samples, feature_columns, source_files = read_world_model_samples(args.windows_dir)
    dataset_validation = validate_world_model_samples(samples, feature_columns)
    if dataset_validation["status"] != "PASS":
        raise ValueError(f"world_model_dataset validation FAILED, refusing to train: {dataset_validation['issues']}")

    counts = tuple(samples[split].X.shape[0] for split in ("train", "validation", "test"))
    expected_counts = tuple(EXPECTED_SPLIT_COUNTS[split] for split in ("train", "validation", "test"))
    if counts != expected_counts or samples["train"].X.shape[1:] != (HISTORY_WINDOWS, INPUT_SIZE):
        raise ValueError(f"Phase 3.5 interface mismatch: counts={counts}, shape={samples['train'].X.shape[1:]}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # -------------------------------------------------------
    # Standardize features -- StandardScaler fit on TRAIN history states (X) only.
    # The same scaler is applied unchanged to X and Y in every split. This is safe
    # for Y because every eligible sample's full t-5..t+6 span is already guaranteed
    # (by the split-boundary guard validated above) to stay inside a single split --
    # so a train sample's future states are still exclusively train-split data, and
    # validation/test never contribute to the fitted statistics.
    # -------------------------------------------------------
    scaler = fit_scaler(samples["train"].X)
    scaled_x = {split: scale_array(samples[split].X, scaler) for split in ("train", "validation", "test")}
    scaled_y = {split: scale_array(samples[split].Y, scaler) for split in ("train", "validation", "test")}

    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir()

    # -------------------------------------------------------
    # Smoke test (before training): shapes, forward pass, backward pass
    # -------------------------------------------------------
    smoke_model = VectorWorldModel().to(device)
    smoke_result = run_smoke_test(smoke_model, device, scaled_x["train"])
    (args.output_dir / "smoke_test.json").write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")
    del smoke_model

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    model = VectorWorldModel().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count <= 0:
        raise RuntimeError("Model has zero parameters")

    train_labels = samples["train"].label
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    pos_weight = torch.tensor(negative / positive, device=device)  # class weighting computed from TRAIN ONLY
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
                "validation_state_mse": validation_state_mse,
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

    # -------------------------------------------------------
    # Validation-selected threshold, then final test evaluation
    # -------------------------------------------------------
    validation_y_hat, validation_probabilities = predict(model, scaled_x["validation"], device, use_amp)
    test_y_hat, test_probabilities = predict(model, scaled_x["test"], device, use_amp)

    selected_threshold = choose_threshold(samples["validation"].label, validation_probabilities)
    validation_metrics = calculate_metrics(samples["validation"].label, validation_probabilities, selected_threshold)
    test_metrics = calculate_metrics(samples["test"].label, test_probabilities, selected_threshold)

    validation_horizon_error = per_horizon_state_error(validation_y_hat, scaled_y["validation"])
    test_horizon_error = per_horizon_state_error(test_y_hat, scaled_y["test"])

    test_horizon_attack = per_horizon_attack_performance(
        samples["test"].future_attack_step, test_probabilities, selected_threshold
    )

    write_csv(args.output_dir / "training_log.csv", history)
    write_csv(args.output_dir / "per_horizon_state_error_validation.csv", validation_horizon_error)
    write_csv(args.output_dir / "per_horizon_state_error_test.csv", test_horizon_error)
    write_csv(args.output_dir / "per_horizon_attack_performance_test.csv", test_horizon_attack)
    save_predictions(args.output_dir / "predictions_validation.csv", samples["validation"], validation_probabilities)
    save_predictions(args.output_dir / "predictions_test.csv", samples["test"], test_probabilities)
    write_csv(
        args.output_dir / "confusion_matrix.csv",
        [
            {"model": "phase6b_vector_world_model", "split": "validation", **validation_metrics["confusion_matrix"]},
            {"model": "phase6b_vector_world_model", "split": "test", **test_metrics["confusion_matrix"]},
        ],
    )

    phase4_metrics = load_existing_metrics(args.phase4_metrics)
    phase5_metrics = load_existing_metrics(args.phase5_metrics)
    comparison = {
        "logistic_regression_test": phase4_metrics["logistic_regression"] if phase4_metrics else None,
        "lstm_test": phase4_metrics["lstm"] if phase4_metrics else None,
        "temporal_transformer_test": phase5_metrics["test"] if phase5_metrics else None,
        "vector_world_model_validation": validation_metrics,
        "vector_world_model_test": test_metrics,
    }

    config = {
        "experiment": "phase6b_vector_world_model",
        "seed": SEED,
        "input_shape": [HISTORY_WINDOWS, INPUT_SIZE],
        "output_shape": [FORECAST_HORIZON_WINDOWS, INPUT_SIZE],
        "feature_count": len(feature_columns),
        "target": "future_attack_within_horizon",
        "target_definition": "any current_attack in t+1 through t+6; current t excluded (same target as Phase 4/5)",
        "counts": {
            split: {
                "samples": int(samples[split].X.shape[0]),
                "positive": int(samples[split].label.sum()),
                "negative": int((samples[split].label == 0).sum()),
            }
            for split in ("train", "validation", "test")
        },
        "dataset_validation": dataset_validation,
        "architecture": {
            "description": "state encoder -> temporal self-attention context -> recursive residual latent transition (x6) -> per-step state decoder + attack head over predicted rollout only",
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "feed_forward_dim": FF_DIM,
            "dropout": DROPOUT,
            "rollout_steps": FORECAST_HORIZON_WINDOWS,
            "parameter_count": parameter_count,
            "attack_head_input": "mean-pooled predicted latent rollout z(t+1)...z(t+6) only -- never the ground-truth Y",
        },
        "training": {
            "loss": "state_loss_weight * MSE(Y_hat, Y) + attack_loss_weight * BCEWithLogits(attack_logit, label)",
            "state_loss_weight": STATE_LOSS_WEIGHT,
            "attack_loss_weight": ATTACK_LOSS_WEIGHT,
            "pos_weight": float(negative / positive),
            "pos_weight_source": "computed from TRAIN split labels only",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "duration_seconds": duration_seconds,
            "selected_validation_threshold": selected_threshold,
            "mixed_precision": use_amp,
        },
        "scaling": "StandardScaler fit on train history states (X) only; same transform applied unchanged to X and Y across all splits",
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "first_training_batch_device": first_batch_device,
        },
        "smoke_test": smoke_result,
        "source_files": source_files,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics_payload = {
        "validation": validation_metrics,
        "test": test_metrics,
        "per_horizon_state_error": {"validation": validation_horizon_error, "test": test_horizon_error},
        "per_horizon_attack_performance_test": test_horizon_attack,
        "comparison": comparison,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "training_duration_seconds": duration_seconds,
        "device": device.type,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Phase 6B Vector World Model",
        "",
        "This experiment reuses the unchanged Phase 3.5 canonical dataset through `world_model_dataset.py` "
        "(itself reused, unmodified, by this run) and the unchanged Phase 4 `future_attack_within_horizon` target. "
        "It performs a genuine learned latent rollout: history is encoded and contextualized, then a residual "
        "transition function is applied recursively six times to produce predicted latents for t+1...t+6, which are "
        "decoded into predicted states and pooled into a single attack probability. The attack head never receives "
        "the ground-truth future states Y in its forward pass.",
        "",
        "## Architecture",
        "",
        f"- Input: `{HISTORY_WINDOWS} x {INPUT_SIZE}`, Output rollout: `{FORECAST_HORIZON_WINDOWS} x {INPUT_SIZE}`",
        f"- d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}, ff_dim={FF_DIM}, dropout={DROPOUT}",
        f"- Parameters: `{parameter_count}`",
        f"- Device: `{device}`",
        f"- Mixed precision: `{use_amp}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Training duration: `{duration_seconds:.3f}` seconds",
        f"- Selected validation threshold: `{selected_threshold:.12g}`",
        "",
        "## Comparison (test set, matched target)",
        "",
        "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def comparison_row(name: str, result: dict[str, object] | None) -> str | None:
        if result is None:
            return None
        cm = result.get("confusion_matrix", result)
        fpr = result.get("false_positive_rate", result.get("fpr", 0.0))
        return (
            f"| {name} | {result['precision']:.6f} | {result['recall']:.6f} | {result['f1']:.6f} | "
            f"{result['pr_auc']:.6f} | {result['roc_auc']:.6f} | {fpr:.6f} | "
            f"{cm['tp']} | {cm['tn']} | {cm['fp']} | {cm['fn']} |"
        )

    for name, result in (
        ("logistic_regression (Phase 4, test)", comparison["logistic_regression_test"]),
        ("lstm (Phase 4, test)", comparison["lstm_test"]),
        ("temporal_transformer (Phase 5, test)", comparison["temporal_transformer_test"]),
        ("vector_world_model (Phase 6B, test)", comparison["vector_world_model_test"]),
    ):
        row = comparison_row(name, result)
        if row:
            report_lines.append(row)

    report_lines += [
        "",
        "## Per-horizon state prediction error (test, standardized feature scale)",
        "",
        "| Horizon | MSE | MAE |",
        "|---|---:|---:|",
    ]
    for row in test_horizon_error:
        report_lines.append(f"| t+{row['horizon_step']} | {row['mse_standardized']:.6f} | {row['mae_standardized']:.6f} |")

    report_lines += [
        "",
        "## Per-horizon attack/risk performance (test, at the validation-selected threshold)",
        "",
        "| future_attack_step | meaning | count | mean predicted probability | predicted-positive rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in test_horizon_attack:
        mean_prob = "n/a" if row["mean_predicted_probability"] is None else f"{row['mean_predicted_probability']:.6f}"
        rate = "n/a" if row["predicted_positive_rate_at_selected_threshold"] is None else f"{row['predicted_positive_rate_at_selected_threshold']:.6f}"
        report_lines.append(f"| {row['future_attack_step']} | {row['meaning']} | {row['sample_count']} | {mean_prob} | {rate} |")

    report_lines += [
        "",
        "The threshold was selected using validation predictions only; the test set was used once for final evaluation. "
        "State-error is reported on the standardized feature scale used by the training loss (the scaler is fit on train "
        "history states only). `future_attack_step=0` samples have no attack anywhere in the horizon (their predicted-positive "
        "rate is a false-positive rate); `future_attack_step=k` samples have their earliest in-horizon attack at t+k.",
        "",
    ]
    (args.output_dir / "comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "device": str(device),
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "duration_seconds": duration_seconds,
                "selected_threshold": selected_threshold,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
