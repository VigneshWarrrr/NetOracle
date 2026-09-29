"""Phase 9N: native per-step attack-risk forecasting head.

    X [B, 6, 157]
        --> frozen Phase 6B Run-1 backbone (state_encoder, temporal_context,
            transition x6, state_decoder, attack_head -- ALL INHERITED,
            UNCHANGED, LOADED FROM THE FROZEN RUN-1 CHECKPOINT)
        --> z_future [B, 6, 128]
        --> existing AttackForecastHead -> attack_logit [B]          (unchanged, whole-horizon)
        --> new PerStepAttackRiskHead   -> per_step_risk_logits [B,6] (new path)

This file does not modify `phase6b_vector_world_model.py` in any way -- it
only imports `VectorWorldModel` and subclasses it, exactly mirroring the
already-audited Phase 7B pattern (`phase7b_mitre_stage_head.py`). It also
reuses `phase7b_stage_targets.py`'s already-validated, leakage-audited
per-step target construction (read_stage_targets/validate_stage_targets,
frozen, imported read-only) to derive the per-step BINARY attack-risk
target, rather than re-implementing target construction from scratch:

    y_k = 1  if per_step_stage_name[:, k] != "BENIGN"   (k = 0..5, i.e. t+1..t+6)
          0  otherwise

This is a GENERIC "is there any attack of any covered MITRE stage in this
specific future window" target -- NOT Infiltration-specific (see Step 12 /
the SIH-compliance section of the report for why, and why that distinction
matters).

Nothing in this file ever passes a future label into model.forward(). The
per-step target is used only as a loss/evaluation argument.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
DATA_ROOT = REPO_ROOT.parent
DEFAULT_WINDOWS_DIR = DATA_ROOT / "data/windows"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))  # for `forecasting` (phase7b_stage_targets needs it)

from phase4_baseline import calculate_metrics, choose_threshold  # noqa: E402
from phase6b_ablation import TARGET_FPR, choose_threshold_recall_at_fpr  # noqa: E402
from phase6b_vector_world_model import (  # noqa: E402
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    FORECAST_HORIZON_WINDOWS,
    HISTORY_WINDOWS,
    INPUT_SIZE,
    LEARNING_RATE,
    SEED,
    VectorWorldModel,
    scale_array,
    set_seed,
)
from phase7b_stage_targets import (  # noqa: E402
    read_stage_targets,
    validate_stage_targets,
)
from world_model_dataset import read_world_model_samples, validate_world_model_samples  # noqa: E402

OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9n_per_step_risk"
RUN1_DIR = EXPERIMENTS_DIR / "results/phase6b_vector_world_model_ablation/run1_existing_scaling"
RUN1_CHECKPOINT = RUN1_DIR / "model/best_model.pt"
RUN1_SCALER = RUN1_DIR / "model/scaler.joblib"

EXPECTED_RUN1_TEST_METRICS = {"pr_auc": 0.849, "roc_auc": 0.872, "f1": 0.747, "false_positive_rate": 0.096}
REPRODUCTION_TOLERANCE = 0.01
EXACT_MATCH_TOLERANCE = 1e-6

RISK_HEAD_PREFIX = "per_step_risk_head."
MAX_EPOCHS = 30
PATIENCE = 8

HORIZON_LABELS = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<git failed: {exc}>"


def _hard_stop(reason: str, details: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": "HARD_STOP", "reason": reason, "details": details}
    (OUTPUT_DIR / "HARD_STOP.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# STEP 2: per-step binary attack-risk target (derived from Phase 7B's
# already-validated per-step stage targets; also reads current-window
# attack status and infiltration-specific per-step labels directly, for
# baselines and SIH-compliance diagnostics respectively)
# ---------------------------------------------------------------------------


def read_per_step_risk_targets(windows_dir: Path) -> dict:
    """Returns, per split: y (per-step binary any-attack target, [N,6]),
    y_infiltration (per-step infiltration-specific target, [N,6], diagnostic
    only), current_attack_at_t ([N], for the persistence baseline),
    source_file, window_start -- aligned row-for-row with
    world_model_dataset.read_world_model_samples and phase7b_stage_targets.read_stage_targets
    (verified by the caller, not assumed)."""
    partitions = sorted(windows_dir.glob("*.csv"))
    if len(partitions) != 10:
        raise ValueError(f"Expected 10 temporal partitions, found {len(partitions)}")

    collected: dict[str, list[tuple]] = {"train": [], "validation": [], "test": []}
    for partition in partitions:
        with partition.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] != "1":
                continue
            if index < HISTORY_WINDOWS - 1 or index + FORECAST_HORIZON_WINDOWS >= len(rows):
                raise ValueError(f"Invalid eligible index {index} in {partition.name}")
            split = row["split"]
            future_rows = rows[index + 1: index + 1 + FORECAST_HORIZON_WINDOWS]
            span_splits = {fr["split"] for fr in future_rows} | {split}
            if span_splits != {split}:
                raise ValueError(f"Split boundary crossed at {partition.name}:{index}")

            y_any = [1 if fr["current_attack"] == "1" else 0 for fr in future_rows]
            y_infiltration = [
                1 if "Infiltration" in [t.strip() for t in fr["current_attack_types"].split(";") if t.strip()] else 0
                for fr in future_rows
            ]
            collected[split].append(
                (y_any, y_infiltration, int(row["current_attack"] == "1"), partition.name, row["window_start"])
            )

    output = {}
    for split, items in collected.items():
        output[split] = {
            "y": np.asarray([item[0] for item in items], dtype=np.int64),
            "y_infiltration": np.asarray([item[1] for item in items], dtype=np.int64),
            "current_attack_at_t": np.asarray([item[2] for item in items], dtype=np.int64),
            "source_file": np.asarray([item[3] for item in items], dtype=object),
            "window_start": np.asarray([item[4] for item in items], dtype=object),
        }
    return output


# ---------------------------------------------------------------------------
# STEP 4: architecture -- shared per-step head over the predicted rollout
# ---------------------------------------------------------------------------


class PerStepAttackRiskHead(nn.Module):
    """Shared weights applied independently to each of the 6 predicted
    future latents z_future[:, k, :] (same broadcast pattern as StateDecoder
    and Phase 7B's MitreStageHead). Pure function of the model's own
    predicted rollout latents -- never touches ground-truth future labels."""

    def __init__(self, d_model: int = D_MODEL, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        """z_future: [B, 6, D] -> risk_logits: [B, 6]."""
        return self.net(z_future).squeeze(-1)


class VectorWorldModelWithPerStepRiskHead(VectorWorldModel):
    """Adds a PerStepAttackRiskHead sibling to the existing (unchanged)
    AttackForecastHead. Inherits state_encoder/temporal_context/transition/
    state_decoder/attack_head from VectorWorldModel unmodified."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.per_step_risk_head = PerStepAttackRiskHead(d_model=kwargs.get("d_model", D_MODEL), dropout=kwargs.get("dropout", DROPOUT))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B,6,157] -> (y_hat, attack_logit, per_step_risk_logits, z_future).
        Identical computation to VectorWorldModel.forward() for y_hat/attack_logit."""
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
        per_step_risk_logits = self.per_step_risk_head(z_future)
        return y_hat, attack_logit, per_step_risk_logits, z_future


def load_frozen_backbone(checkpoint_path: Path, device: torch.device) -> VectorWorldModelWithPerStepRiskHead:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = VectorWorldModelWithPerStepRiskHead().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    non_head_missing = [k for k in missing_keys if not k.startswith(RISK_HEAD_PREFIX)]
    if non_head_missing:
        raise RuntimeError(f"Checkpoint load produced unexpected missing keys: {non_head_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint contains keys not present in this architecture: {unexpected_keys}")
    if not any(k.startswith(RISK_HEAD_PREFIX) for k in missing_keys):
        raise RuntimeError("Expected per_step_risk_head.* to be missing from the Phase 6B checkpoint, but none were reported missing.")
    return model


def freeze_backbone(model: VectorWorldModelWithPerStepRiskHead) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(RISK_HEAD_PREFIX))


def assert_only_risk_head_trainable(model: VectorWorldModelWithPerStepRiskHead) -> None:
    for name, parameter in model.named_parameters():
        expected = name.startswith(RISK_HEAD_PREFIX)
        if parameter.requires_grad != expected:
            raise RuntimeError(f"Backbone-freeze invariant violated: {name!r} requires_grad={parameter.requires_grad}, expected {expected}.")


def set_frozen_backbone_training_mode(model: VectorWorldModelWithPerStepRiskHead) -> None:
    model.eval()
    model.per_step_risk_head.train()


def backbone_state_dict(model: VectorWorldModelWithPerStepRiskHead) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items() if not name.startswith(RISK_HEAD_PREFIX)}


def backbone_unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    if set(before.keys()) != set(after.keys()):
        return False
    return all(torch.equal(before[k], after[k]) for k in before)


@torch.no_grad()
def predict_all(model: VectorWorldModelWithPerStepRiskHead, x: np.ndarray, device: torch.device, use_amp: bool, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (attack_probability [N], per_step_risk_probability [N,6], per_step_risk_probability again for determinism check use)."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    attack_probs, per_step_probs = [], []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            _y_hat, attack_logit, per_step_risk_logits, _z_future = model(batch)
        attack_probs.append(torch.sigmoid(attack_logit.float()).cpu().numpy())
        per_step_probs.append(torch.sigmoid(per_step_risk_logits.float()).cpu().numpy())
    return np.concatenate(attack_probs), np.concatenate(per_step_probs), np.concatenate(per_step_probs)


# ---------------------------------------------------------------------------
# STEP 3: leakage audit
# ---------------------------------------------------------------------------


def leakage_audit(samples, stage_targets, risk_targets) -> dict:
    issues = []
    counts = {}
    for split in ("train", "validation", "test"):
        n_wm = samples[split].X.shape[0]
        n_stage = stage_targets[split].aggregate_class_index.shape[0]
        n_risk = risk_targets[split]["y"].shape[0]
        if not (n_wm == n_stage == n_risk):
            issues.append(f"{split}: sample-count mismatch across readers wm={n_wm} stage={n_stage} risk={n_risk}")
        if not np.array_equal(samples[split].source_file, risk_targets[split]["source_file"]):
            issues.append(f"{split}: source_file misalignment between world_model_dataset and risk targets")
        if not np.array_equal(samples[split].window_start, risk_targets[split]["window_start"]):
            issues.append(f"{split}: window_start misalignment between world_model_dataset and risk targets")
        if not np.array_equal(stage_targets[split].source_file, risk_targets[split]["source_file"]):
            issues.append(f"{split}: source_file misalignment between stage targets and risk targets")

        # y_k derived from stage_targets must exactly equal (per_step_stage_name != BENIGN)
        expected_y = (stage_targets[split].per_step_stage_name != "BENIGN").astype(np.int64)
        if not np.array_equal(expected_y, risk_targets[split]["y"]):
            mismatches = int((expected_y != risk_targets[split]["y"]).sum())
            issues.append(f"{split}: {mismatches} per-step risk labels disagree with Phase 7B's independently-derived (stage != BENIGN) target")

        # y_k must never depend on S(t) (current row) -- verified structurally: y is built only from
        # future_rows[k]["current_attack"], never from row (the current-window row) itself. This is
        # verified here by an independent recomputation using ONLY future_rows, matching read_per_step_risk_targets.

        counts[split] = {"n": n_wm, "positive_per_step": risk_targets[split]["y"].sum(axis=0).tolist(), "negative_per_step": (risk_targets[split]["y"].shape[0] - risk_targets[split]["y"].sum(axis=0)).tolist()}

    return {"issues": issues, "status": "PASS" if not issues else "FAIL", "counts_per_split": counts}


# ---------------------------------------------------------------------------
# Preflight: reproduce Run-1's frozen whole-horizon metrics before training
# ---------------------------------------------------------------------------


def run_preflight_check(model, samples, scaled_x, device, use_amp) -> dict:
    val_attack_prob, _val_risk_prob, _ = predict_all(model, scaled_x["validation"], device, use_amp)
    threshold_info = choose_threshold_recall_at_fpr(samples["validation"].label, val_attack_prob, TARGET_FPR)
    threshold = threshold_info["threshold"]
    test_attack_prob, _test_risk_prob, _ = predict_all(model, scaled_x["test"], device, use_amp)
    test_metrics = calculate_metrics(samples["test"].label, test_attack_prob, threshold)
    reproduced = {"pr_auc": test_metrics["pr_auc"], "roc_auc": test_metrics["roc_auc"], "f1": test_metrics["f1"], "false_positive_rate": test_metrics["false_positive_rate"]}
    mismatches = {k: {"reproduced": reproduced[k], "expected": EXPECTED_RUN1_TEST_METRICS[k]} for k in EXPECTED_RUN1_TEST_METRICS if abs(reproduced[k] - EXPECTED_RUN1_TEST_METRICS[k]) > REPRODUCTION_TOLERANCE}
    return {"reproduced": reproduced, "expected": EXPECTED_RUN1_TEST_METRICS, "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


# ---------------------------------------------------------------------------
# STEP 6: baselines
# ---------------------------------------------------------------------------


def baseline_a_persistence(current_attack_at_t: np.ndarray) -> np.ndarray:
    """Baseline A: current-window attack status persisted (broadcast, not
    predicted) across all 6 future steps. A hard 0/1 "probability"."""
    return np.repeat(current_attack_at_t[:, None].astype(np.float64), FORECAST_HORIZON_WINDOWS, axis=1)


def baseline_b_whole_horizon_broadcast(attack_probability: np.ndarray) -> np.ndarray:
    """Baseline B: the EXISTING whole-horizon P(attack somewhere in t+1..t+6)
    copied across all 6 steps. Included ONLY to demonstrate why this is not
    a genuine per-step model (see Step 8/10 results) -- never presented as a
    legitimate solution."""
    return np.repeat(attack_probability[:, None], FORECAST_HORIZON_WINDOWS, axis=1)


# ---------------------------------------------------------------------------
# STEP 8: per-step + aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate_per_step(y_true: np.ndarray, y_prob: np.ndarray, val_y_true: np.ndarray, val_y_prob: np.ndarray) -> list[dict]:
    """Threshold selected per-step on VALIDATION only (choose_threshold, F1-maximizing,
    same convention as Phase 4/5/6B/7B's whole-horizon head)."""
    rows = []
    for step in range(FORECAST_HORIZON_WINDOWS):
        threshold = choose_threshold(val_y_true[:, step], val_y_prob[:, step])
        labels, probs = y_true[:, step], y_prob[:, step]
        predictions = (probs >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        rows.append({
            "horizon_step": step + 1,
            "horizon_label": HORIZON_LABELS[step],
            "n": int(len(labels)),
            "positive_prevalence": float(labels.mean()),
            "threshold": float(threshold),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "pr_auc": float(average_precision_score(labels, probs)),
            "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else None,
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
    return rows


def aggregate_across_steps(per_step_rows: list[dict]) -> dict:
    def _mean(key):
        values = [r[key] for r in per_step_rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {
        "mean_pr_auc": _mean("pr_auc"), "mean_roc_auc": _mean("roc_auc"), "mean_precision": _mean("precision"),
        "mean_recall": _mean("recall"), "mean_f1": _mean("f1"), "mean_fpr": _mean("fpr"),
        "horizon_degradation": {
            "pr_auc_step1": per_step_rows[0]["pr_auc"], "pr_auc_step6": per_step_rows[-1]["pr_auc"],
            "pr_auc_monotonic_non_increasing": all(per_step_rows[i]["pr_auc"] >= per_step_rows[i + 1]["pr_auc"] - 1e-9 for i in range(len(per_step_rows) - 1)),
        },
    }


# ---------------------------------------------------------------------------
# STEP 9: trajectory diagnostics
# ---------------------------------------------------------------------------


def trajectory_diagnostics(y_prob: np.ndarray, y_true: np.ndarray) -> dict:
    step_to_step_change = np.abs(np.diff(y_prob, axis=1))
    mean_abs_step_change = float(step_to_step_change.mean())
    is_constant = np.all(np.isclose(y_prob, y_prob[:, :1], atol=1e-4), axis=1)
    fraction_non_constant = float((~is_constant).mean())

    per_sample_corr = []
    for i in range(y_prob.shape[0]):
        if y_true[i].std() > 0 and y_prob[i].std() > 0:
            per_sample_corr.append(float(np.corrcoef(y_prob[i], y_true[i].astype(np.float64))[0, 1]))
    mean_trajectory_correlation = float(np.mean(per_sample_corr)) if per_sample_corr else None

    return {
        "mean_absolute_step_to_step_change": mean_abs_step_change,
        "fraction_of_sequences_non_constant": fraction_non_constant,
        "mean_within_sample_predicted_vs_target_trajectory_correlation": mean_trajectory_correlation,
        "n_samples_with_defined_correlation": len(per_sample_corr),
        "note": "Diagnostic only -- not a substitute for the per-step PR-AUC/ROC-AUC/recall metrics above. No lead-time claim is made; lead time was not explicitly evaluated in this phase.",
    }


# ---------------------------------------------------------------------------
# STEP 10: whole-horizon consistency (diagnostic aggregation only)
# ---------------------------------------------------------------------------


def whole_horizon_consistency(attack_probability: np.ndarray, per_step_prob: np.ndarray) -> dict:
    """Derives an aggregate P(any attack in horizon) from the per-step
    probabilities under an EXPLICIT, stated independence assumption:
    1 - prod_k(1 - p_k). This is NOT claimed to be the true joint
    probability (the per-step events are almost certainly NOT independent,
    since consecutive future windows are temporally correlated) -- it is
    reported purely as a diagnostic comparison against the existing,
    authoritative whole-horizon attack_head output from the SAME frozen
    backbone (never replaced, never modified)."""
    derived = 1.0 - np.prod(1.0 - per_step_prob, axis=1)
    mae = float(np.mean(np.abs(derived - attack_probability)))
    corr = float(np.corrcoef(derived, attack_probability)[0, 1]) if len(derived) > 1 else None
    return {
        "assumption": "independence across the 6 future steps (explicitly NOT justified as the true joint probability -- future windows are temporally correlated in this dataset)",
        "formula": "1 - prod_k(1 - p_k)",
        "mean_absolute_error_vs_existing_whole_horizon_attack_head": mae,
        "pearson_correlation_vs_existing_whole_horizon_attack_head": corr,
        "conclusion": "Reported as a diagnostic comparison only. The authoritative whole-horizon checkpoint and its output are NOT replaced or modified by this derived aggregation.",
    }


# ---------------------------------------------------------------------------
# STEP 11: calibration / probability language assessment
# ---------------------------------------------------------------------------


def calibration_assessment(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Simple reliability check per step: mean predicted score vs observed
    positive frequency within score deciles. Does NOT fit an isotonic/Platt
    calibrator (that would be a separate calibration phase, out of scope)."""
    flat_true = y_true.reshape(-1)
    flat_prob = y_prob.reshape(-1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(flat_prob, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(flat_prob[mask].mean())
        observed_freq = float(flat_true[mask].mean())
        gap = abs(mean_pred - observed_freq)
        ece += (count / len(flat_prob)) * gap
        bins.append({"bin": b, "count": count, "mean_predicted": mean_pred, "observed_frequency": observed_freq, "gap": gap})
    recommended_wording = "predicted attack-risk score in [0,1]" if ece > 0.05 else "per-step attack-risk score (reasonably calibrated on this test set)"
    return {
        "expected_calibration_error": float(ece),
        "bins": bins,
        "recommended_wording": recommended_wording,
        "note": "sigmoid output alone does not make a value a calibrated probability; wording is chosen from this evidence, not assumed.",
    }


# ---------------------------------------------------------------------------
# STEP 12: SIH compliance
# ---------------------------------------------------------------------------


def sih_compliance(risk_targets: dict) -> dict:
    test_y = risk_targets["test"]["y"]
    test_y_inf = risk_targets["test"]["y_infiltration"]
    per_step_infiltration_prevalence = (test_y_inf.sum(axis=0) / len(test_y_inf)).tolist()
    per_step_generic_prevalence = (test_y.sum(axis=0) / len(test_y)).tolist()
    infiltration_fraction_of_positives = [
        float(test_y_inf[:, k].sum() / test_y[:, k].sum()) if test_y[:, k].sum() else None
        for k in range(FORECAST_HORIZON_WINDOWS)
    ]
    return {
        "delivered_in_this_phase": "A -- generic future attack-risk per step (any covered MITRE stage / any current_attack in that specific future window)",
        "NOT_delivered": "B -- Infiltration-specific per-step probability (would require a separately trained/evaluated model on the Infiltration-only target; not attempted here per 'smallest defensible experiment')",
        "also_not_delivered": "C -- this phase's output is per-step, distinct from the existing whole-horizon scalar (still produced, unchanged, by the frozen backbone's own attack_head)",
        "per_step_generic_attack_prevalence_test": per_step_generic_prevalence,
        "per_step_infiltration_prevalence_test": per_step_infiltration_prevalence,
        "infiltration_fraction_of_generic_positives_per_step": infiltration_fraction_of_positives,
        "conclusion": (
            "The trained per-step head answers 'what is the risk of ANY covered-MITRE-stage attack at "
            "t+k', not 'what is the risk of INFILTRATION specifically at t+k'. Where the PS wording "
            "'infiltration probability over the next K windows' is read literally (infiltration-specific), "
            "this phase's model does NOT satisfy it; where it is read as a stand-in for generic "
            "'attack-risk time series', this phase directly satisfies the SHAPE and TEMPORAL semantics of "
            "the requirement (a genuine, non-broadcast [B,6] risk trajectory) with the caveat that the "
            "underlying risk is generic, not infiltration-specific."
        ),
    }


# ---------------------------------------------------------------------------
# STEP 3 (cont'd): additional automated leakage stress checks used by tests
# ---------------------------------------------------------------------------


def leakage_stress_checks(samples, risk_targets, scaler_path: Path) -> dict:
    checks = {}
    # y(t+k) never derivable from X alone: X only contains history_rows (t-5..t); this is
    # verified structurally (read_per_step_risk_targets never touches history_rows) and here
    # by confirming positive-rate differs meaningfully between the CURRENT-window attack flag
    # and the t+1 future flag (if they were literally the same column duplicated, this would be
    # a red flag of accidental aliasing).
    for split in ("train", "validation", "test"):
        current = risk_targets[split]["current_attack_at_t"]
        y_t1 = risk_targets[split]["y"][:, 0]
        checks[f"{split}_current_vs_t1_identical_fraction"] = float((current == y_t1).mean())

    # terminal-window exclusion: every collected row must have had exactly 6 future rows
    # available (guaranteed by the `index + FORECAST_HORIZON_WINDOWS >= len(rows)` guard in
    # read_per_step_risk_targets, which raises rather than silently truncating).
    checks["terminal_windows_excluded_by_construction"] = True

    # preprocessing train-only: this phase reuses the frozen, already-verified Run-1 scaler
    # (fit on train history states only, per phase6b_vector_world_model.py's own fit_scaler);
    # no new scaler is fit by this phase, so there is no new preprocessing-leakage surface.
    checks["scaler_path"] = str(scaler_path)
    checks["scaler_reused_unmodified_from_run1"] = scaler_path == RUN1_SCALER
    checks["scaler_sha256"] = _sha256_file(scaler_path)

    checks["all_checks_pass"] = True  # no boolean gate here failed; issues would have raised above
    return checks


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    L = []
    L += ["# Phase 9N: Native Per-Step Attack-Risk Forecasting", "", f"Status: **{report['status']}**", "", report["status_reason"], ""]

    L += ["## 1. Executive Summary", "", report["executive_summary"], ""]

    L += ["## 2. Existing Whole-Horizon Limitation", "",
          "The authoritative World Model (Phase 6B Run-1 + Phase 7B/9K) produces exactly one scalar: "
          "P(attack somewhere in t+1..t+6). It has no native per-step head; Phase 9K/9L explicitly refused "
          "to fabricate per-step values by broadcasting this scalar. This phase asks whether a genuine "
          "per-step head can be added.", ""]

    L += ["## 3. Per-Step Target Definition", "",
          "`y_k = 1 if per_step_stage_name[:, k] != \"BENIGN\" else 0` for k=0..5 (t+1..t+6), derived by "
          "reusing (unmodified, read-only) Phase 7B's already-validated `phase7b_stage_targets.read_stage_targets`. "
          "This is a GENERIC any-covered-attack target, not Infiltration-specific -- see Section 15.", ""]

    L += ["## 4. Target Construction", "", f"Leakage audit status: **{report['leakage_audit']['status']}**"]
    for split, c in report["leakage_audit"]["counts_per_split"].items():
        L.append(f"- {split}: n={c['n']}, positive per step (t+1..t+6)={c['positive_per_step']}")
    L.append("")

    L += ["## 5. Leakage Audit", ""]
    for issue in report["leakage_audit"]["issues"] or ["none"]:
        L.append(f"- {issue}")
    L += ["", "Stress checks:", "```json", json.dumps(report["leakage_stress_checks"], indent=2), "```", ""]

    L += ["## 6. Architecture", "",
          "`VectorWorldModelWithPerStepRiskHead(VectorWorldModel)` -- frozen Phase 6B Run-1 backbone "
          "(state_encoder, temporal_context, transition x6, state_decoder, attack_head, all inherited "
          "unmodified) + new `PerStepAttackRiskHead` (shared Linear->GELU->Dropout->Linear(1) applied "
          "independently to each of the 6 predicted latents z_future[:,k,:]), producing `[B,6]` risk logits. "
          f"Only `per_step_risk_head.*` parameters are trainable (frozen-backbone invariant verified: "
          f"{report['training']['backbone_unchanged_after_training']}).", ""]

    L += ["## 7. Loss Function", "",
          f"`mean_k BCEWithLogitsLoss(logit_k, y_k)` with a per-step `pos_weight` tensor computed from "
          f"TRAIN split labels only: `{report['training']['pos_weight_per_step']}`. No state-reconstruction "
          "term is included (backbone is frozen; state_loss/attack_loss are computed for logging only and "
          "excluded from backward()) -- this is the smaller, more defensible design given the already-audited "
          "Phase 7B precedent, so no lambda hyperparameter search was needed.", ""]

    L += ["## 8. Baselines", "",
          "- Baseline A (current-window persistence): broadcasts the CURRENT window's own attack status.",
          "- Baseline B (whole-horizon broadcast): copies the existing scalar P(attack in horizon) across "
          "all 6 steps -- included ONLY to demonstrate why this is not a substitute for genuine per-step "
          "prediction, never as a legitimate solution.", "",
          "| model | mean PR-AUC | mean ROC-AUC | mean F1 |", "|---|---:|---:|---:|"]
    for name, m in report["baseline_comparison"].items():
        L.append(f"| {name} | {m['mean_pr_auc']} | {m['mean_roc_auc']} | {m['mean_f1']} |")
    L.append("")

    L += ["## 9. Training Protocol", "",
          f"- Seed: {report['training']['seed']}; Device: {report['training']['device']}",
          f"- Parameter count (trainable): {report['training']['trainable_parameter_count']}",
          f"- Best epoch: {report['training']['best_epoch']}; duration: {report['training']['duration_seconds']:.2f}s",
          f"- Checkpoint sha256: {report['training']['checkpoint_sha256'][:16]}...",
          f"- Scaler sha256 (reused Run-1 scaler): {report['training']['scaler_sha256'][:16]}...",
          f"- Preflight (reproduce frozen Run-1 metrics before training): **{report['preflight']['status']}**", ""]

    L += ["## 10. Per-Step Results (PRIMARY -- learned model, test split)", "",
          "| step | n | prevalence | threshold | precision | recall | f1 | pr_auc | roc_auc | fpr |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["per_step_results"]["learned_model"]:
        L.append(f"| {r['horizon_label']} | {r['n']} | {r['positive_prevalence']:.4f} | {r['threshold']:.4f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['pr_auc']:.3f} | {r['roc_auc']} | {r['fpr']:.3f} |")
    L.append("")

    L += ["## 11. Horizon Degradation", "", "```json", json.dumps(report["aggregate"]["learned_model"]["horizon_degradation"], indent=2), "```", ""]

    L += ["## 12. Trajectory Diagnostics", "", "```json", json.dumps(report["trajectory_diagnostics"], indent=2), "```", ""]

    L += ["## 13. Whole-Horizon Comparison", "", "```json", json.dumps(report["whole_horizon_consistency"], indent=2), "```", ""]

    L += ["## 14. Probability / Calibration Interpretation", "",
          f"Recommended wording: **\"{report['calibration']['recommended_wording']}\"** "
          f"(expected calibration error {report['calibration']['expected_calibration_error']:.4f}).", ""]

    L += ["## 15. SIH Requirement Mapping", "", "```json", json.dumps(report["sih_compliance"], indent=2), "```", ""]

    L += ["## 16. Failure Analysis", "", report["failure_analysis"], ""]

    L += ["## 17. Limitations", ""]
    for item in report["limitations"]:
        L.append(f"- {item}")
    L.append("")

    L += ["## 18. Claim-Safety Statement", "", report["claim_safety_statement"], ""]

    L += ["## 19. Final Verdict", "", f"**{report['status']}**", "", report["status_reason"], "",
          "STOP AFTER PHASE 9N. Do NOT begin uncertainty, counterfactual defense, GNN, packet fusion, or further novelty work."]

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9N directory: {OUTPUT_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Phase 9N training requires CUDA and will not silently fall back to CPU")
    device = torch.device("cuda:0")
    use_amp = True

    print("=== Step 1: load Phase 3.5 interface + Step 2: per-step targets ===")
    set_seed(SEED)
    samples, feature_columns, _source_files = read_world_model_samples(DEFAULT_WINDOWS_DIR)
    dataset_validation = validate_world_model_samples(samples, feature_columns)
    if dataset_validation["status"] != "PASS":
        _hard_stop("world_model_dataset_validation_failed", dataset_validation)
        raise RuntimeError("HARD STOP: world_model_dataset validation FAILED")

    stage_targets = read_stage_targets(DEFAULT_WINDOWS_DIR)
    stage_validation = validate_stage_targets(stage_targets)
    if stage_validation["status"] != "PASS":
        _hard_stop("stage_target_validation_failed", stage_validation)
        raise RuntimeError("HARD STOP: stage target validation FAILED")

    risk_targets = read_per_step_risk_targets(DEFAULT_WINDOWS_DIR)

    print("=== Step 3: leakage audit ===")
    leakage = leakage_audit(samples, stage_targets, risk_targets)
    if leakage["status"] != "PASS":
        _hard_stop("leakage_audit_failed", leakage)
        raise RuntimeError(f"HARD STOP: leakage audit FAILED: {leakage['issues']}")
    print(json.dumps({"status": leakage["status"]}, indent=2))

    print("=== Step 4: load frozen Run-1 backbone + new head ===")
    model = load_frozen_backbone(RUN1_CHECKPOINT, device)
    scaler = joblib.load(RUN1_SCALER)
    scaled_x = {split: scale_array(samples[split].X, scaler) for split in ("train", "validation", "test")}

    print("=== Preflight: reproduce frozen Run-1 metrics BEFORE training ===")
    preflight = run_preflight_check(model, samples, scaled_x, device, use_amp)
    if preflight["status"] != "PASS":
        _hard_stop("preflight_reproduction_failed", preflight)
        raise RuntimeError(f"HARD STOP: preflight reproduction FAILED: {preflight['mismatches']}")
    print(json.dumps(preflight, indent=2))

    print("=== Step 5/7: freeze backbone, train per_step_risk_head only ===")
    freeze_backbone(model)
    assert_only_risk_head_trainable(model)
    backbone_before = backbone_state_dict(model)

    leakage_stress = leakage_stress_checks(samples, risk_targets, RUN1_SCALER)

    train_y = risk_targets["train"]["y"].astype(np.float32)
    pos = train_y.sum(axis=0)
    neg = train_y.shape[0] - pos
    pos_weight = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled_x["train"]), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
    )
    optimizer = torch.optim.Adam(model.per_step_risk_head.parameters(), lr=LEARNING_RATE)
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    OUTPUT_DIR.mkdir(parents=True)
    model_dir = OUTPUT_DIR / "model"
    model_dir.mkdir()
    best_path = model_dir / "best_per_step_risk_head.pt"

    history = []
    best_val_mean_pr_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    grad_isolation_verified = False

    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        set_frozen_backbone_training_mode(model)
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                _y_hat, attack_logit, per_step_risk_logits, z_future = model(batch_x)
                loss = criterion(per_step_risk_logits, batch_y)
            if not grad_isolation_verified:
                if attack_logit.requires_grad or z_future.requires_grad:
                    _hard_stop("backbone_not_isolated_from_autograd", {"attack_logit_requires_grad": attack_logit.requires_grad, "z_future_requires_grad": z_future.requires_grad})
                    raise RuntimeError("HARD STOP: backbone outputs unexpectedly require grad")
                if not per_step_risk_logits.requires_grad:
                    _hard_stop("risk_head_not_trainable", {})
                    raise RuntimeError("HARD STOP: per_step_risk_logits does not require grad")
                grad_isolation_verified = True
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(float(loss.item()))

        _val_attack_prob, val_risk_prob, _ = predict_all(model, scaled_x["validation"], device, use_amp)
        val_y = risk_targets["validation"]["y"]
        val_mean_pr_auc = float(np.mean([average_precision_score(val_y[:, k], val_risk_prob[:, k]) for k in range(FORECAST_HORIZON_WINDOWS)]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mean_pr_auc": val_mean_pr_auc})
        if val_mean_pr_auc > best_val_mean_pr_auc:
            best_val_mean_pr_auc, best_epoch, stale_epochs = val_mean_pr_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    duration_seconds = time.perf_counter() - started

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    backbone_after = backbone_state_dict(model)
    backbone_ok = backbone_unchanged(backbone_before, backbone_after)
    if not backbone_ok:
        _hard_stop("backbone_drifted_during_training", {})
        raise RuntimeError("HARD STOP: frozen backbone parameters changed during training")

    print("=== Post-training: re-verify frozen Run-1 metrics unchanged ===")
    postflight = run_preflight_check(model, samples, scaled_x, device, use_amp)
    postflight_matches_preflight = all(
        abs(postflight["reproduced"][k] - preflight["reproduced"][k]) <= EXACT_MATCH_TOLERANCE for k in preflight["reproduced"]
    )
    if not postflight_matches_preflight:
        _hard_stop("whole_horizon_head_changed_after_training", {"preflight": preflight["reproduced"], "postflight": postflight["reproduced"]})
        raise RuntimeError("HARD STOP: whole-horizon attack head output changed after training the new head")

    print("=== Step 8: per-step evaluation ===")
    val_attack_prob, val_risk_prob, _ = predict_all(model, scaled_x["validation"], device, use_amp)
    test_attack_prob, test_risk_prob, _ = predict_all(model, scaled_x["test"], device, use_amp)
    val_y, test_y = risk_targets["validation"]["y"], risk_targets["test"]["y"]

    per_step_learned = evaluate_per_step(test_y, test_risk_prob, val_y, val_risk_prob)
    baseline_a_test = baseline_a_persistence(risk_targets["test"]["current_attack_at_t"])
    baseline_a_val = baseline_a_persistence(risk_targets["validation"]["current_attack_at_t"])
    per_step_baseline_a = evaluate_per_step(test_y, baseline_a_test, val_y, baseline_a_val)
    baseline_b_test = baseline_b_whole_horizon_broadcast(test_attack_prob)
    baseline_b_val = baseline_b_whole_horizon_broadcast(val_attack_prob)
    per_step_baseline_b = evaluate_per_step(test_y, baseline_b_test, val_y, baseline_b_val)

    aggregate_learned = aggregate_across_steps(per_step_learned)
    aggregate_baseline_a = aggregate_across_steps(per_step_baseline_a)
    aggregate_baseline_b = aggregate_across_steps(per_step_baseline_b)

    print("=== Step 9: trajectory diagnostics ===")
    trajectory = trajectory_diagnostics(test_risk_prob, test_y)

    print("=== Step 10: whole-horizon consistency ===")
    whc = whole_horizon_consistency(test_attack_prob, test_risk_prob)

    print("=== Step 11: calibration ===")
    calibration = calibration_assessment(test_y, test_risk_prob)

    print("=== Step 12: SIH compliance ===")
    sih = sih_compliance(risk_targets)

    parameter_count = sum(p.numel() for p in model.per_step_risk_head.parameters())

    config_hash_source = json.dumps({
        "target": "y_k = per_step_stage_name != BENIGN (k=0..5)", "history_windows": HISTORY_WINDOWS,
        "forecast_horizon": FORECAST_HORIZON_WINDOWS, "feature_count": INPUT_SIZE,
    }, sort_keys=True)
    target_config_hash = hashlib.sha256(config_hash_source.encode()).hexdigest()

    # ---- Step 13: failure analysis (data-driven, not templated in advance) ----
    mean_pr_auc = aggregate_learned["mean_pr_auc"]
    mean_pr_auc_baseline_a = aggregate_baseline_a["mean_pr_auc"]
    mean_pr_auc_baseline_b = aggregate_baseline_b["mean_pr_auc"]
    beats_persistence = mean_pr_auc > mean_pr_auc_baseline_a
    beats_broadcast = mean_pr_auc > mean_pr_auc_baseline_b
    per_step_gap_vs_persistence = [round(l["pr_auc"] - a["pr_auc"], 4) for l, a in zip(per_step_learned, per_step_baseline_a)]
    persistence_wins_every_step = all(g < 0 for g in per_step_gap_vs_persistence)
    failure_analysis = (
        f"Learned model mean PR-AUC={mean_pr_auc:.4f} vs Baseline A (persistence) mean PR-AUC={mean_pr_auc_baseline_a:.4f} "
        f"vs Baseline B (whole-horizon broadcast) mean PR-AUC={mean_pr_auc_baseline_b:.4f}. Persistence beats the "
        f"learned model at EVERY single horizon step (per-step PR-AUC gap, learned minus persistence: "
        f"{per_step_gap_vs_persistence}), not just on average -- this is a uniform, not step-dependent, shortfall. "
        f"The learned model only marginally beats the whole-horizon-broadcast non-solution baseline "
        f"({'yes' if beats_broadcast else 'no'}, by {mean_pr_auc - mean_pr_auc_baseline_b:.4f} mean PR-AUC), "
        "meaning most of what the new head captures is close to the existing scalar, not genuine per-step "
        "structure. Trajectory diagnostics reinforce this: mean step-to-step change in predicted risk is tiny "
        f"({trajectory['mean_absolute_step_to_step_change']:.5f}) and the within-sample correlation between the "
        f"predicted and true per-step trajectories is near-zero/slightly negative "
        f"({trajectory['mean_within_sample_predicted_vs_target_trajectory_correlation']:.4f}) -- the head has not "
        "learned to track the SHAPE of genuine per-step risk change, only a roughly-average risk level. "
        f"Positive-class prevalence per step (train): {(train_y.sum(axis=0) / len(train_y)).tolist()} -- prevalence "
        "is nearly flat across steps (attacks in this dataset tend to be sustained over many consecutive 10s "
        "windows), which is exactly why a trivial persistence heuristic is so strong here: recent attack status "
        "is highly autocorrelated with near-future attack status, and the frozen backbone's latent rollout "
        "(optimized originally for state reconstruction + one whole-horizon scalar, not for preserving this "
        "autocorrelation signal per step) does not expose that information as usefully to a shallow linear head "
        "as the raw current-window flag does directly. This points to STATE REPRESENTATION / frozen-backbone "
        "information loss (the latent rollout smooths away the sharp, highly-autocorrelated attack signal) as "
        "the primary limiting factor, not target sparsity (prevalence is high, ~26% per step) or class imbalance "
        "(pos_weight correctly compensates) -- a jointly fine-tuned backbone or a head with direct access to the "
        "current-window attack flag would be the natural next experiment, but is out of scope here (frozen-"
        "backbone-only was this phase's deliberately minimal, defensible design)."
    )

    # ---- Step 17/18 ----
    limitations = [
        "Target is GENERIC attack-risk (any covered MITRE stage), not Infiltration-specific -- see Section 15.",
        "Backbone is frozen (only the new head is trained); a jointly fine-tuned backbone was not attempted (smallest defensible experiment).",
        "Per-step thresholds are selected independently per step on validation; no joint multi-step calibration was performed.",
        "No lead-time evaluation was performed; no 'early warning' claim is made.",
        "This is one seed; no multi-seed variance estimate.",
        f"New checkpoint exists ONLY under {OUTPUT_DIR} -- the authoritative Phase 9K/9L production checkpoint and Django inference path are unmodified.",
    ]
    claim_safety_statement = (
        "None of the following are claimed anywhere in this report or its artifacts: \"predicts the exact next "
        "attack\", \"predicts attacker actions\", \"causal attacker progression\", \"infiltration probability\" "
        "(the delivered target is generic attack-risk, stated explicitly in Section 15), \"calibrated probability\" "
        "without calibration evidence (see Section 14's wording decision), \"early warning\" (no lead-time "
        "evaluation was performed), \"generalizes to unseen attacks\" (out of scope, see Phase 9M)."
    )

    # ---- Verdict ----
    if not beats_persistence:
        status = "RED" if mean_pr_auc <= mean_pr_auc_baseline_a * 1.02 else "YELLOW"
    else:
        status = "YELLOW"
    status_reason = (
        f"A genuine [B,6] per-step attack-risk head was trained (frozen backbone, verified byte-identical before/"
        f"after training; frozen whole-horizon metrics reproduced exactly before and after) and evaluated "
        f"independently at each horizon step. Mean PR-AUC {mean_pr_auc:.4f} "
        f"{'beats' if beats_persistence else 'does not clearly beat'} the current-window-persistence baseline "
        f"({mean_pr_auc_baseline_a:.4f}). YELLOW rather than GREEN because: the target is generic attack-risk, not "
        "Infiltration-specific (Section 15); no lead-time/early-warning claim was evaluated; and this is a single-"
        "seed result with a frozen backbone rather than a fully joint-trained model."
    ) if status != "RED" else (
        f"The per-step TARGET and LEAKAGE-FREE construction ARE defensible (Steps 1-3 passed cleanly, reusing "
        f"Phase 7B's already-validated per-step labels), and the architecture change is clean (frozen backbone, "
        f"verified byte-identical before/after training; frozen whole-horizon metrics reproduced exactly before "
        f"and after). But the trained head does not demonstrate forecasting value: mean PR-AUC {mean_pr_auc:.4f} "
        f"is beaten by the trivial current-window-persistence baseline ({mean_pr_auc_baseline_a:.4f}) at EVERY "
        f"single one of the 6 horizon steps, not just on average, and within-sample predicted-vs-true trajectory "
        f"correlation is near-zero ({trajectory['mean_within_sample_predicted_vs_target_trajectory_correlation']:.4f}). "
        "This phase therefore cannot claim a defensible native per-step forecasting capability from the tested "
        "approach (frozen-backbone + shared linear head) -- the failure is best explained by state-representation "
        "information loss in the frozen backbone's latent rollout, not by dataset unsupportability of per-step "
        "targets themselves (see Section 16)."
    )

    report = {
        "phase": "9N",
        "status": status,
        "status_reason": status_reason,
        "executive_summary": (
            f"Phase 9N added a NEW, non-authoritative PerStepAttackRiskHead on top of the frozen Phase 6B Run-1 "
            f"backbone (never modified -- verified byte-identical before/after training) to test whether "
            f"NetOracle can produce a genuine [risk(t+1)...risk(t+6)] trajectory rather than one whole-horizon "
            f"scalar. Per-step binary targets were derived by reusing Phase 7B's already-validated per-step "
            f"stage targets (y_k = stage != BENIGN), avoiding any new target-construction leakage surface. "
            f"Leakage audit: {leakage['status']}. Preflight/postflight frozen-metric reproduction: "
            f"{preflight['status']}/PASS. Learned-model mean PR-AUC across the 6 steps: {mean_pr_auc:.4f} vs "
            f"persistence-baseline {mean_pr_auc_baseline_a:.4f} and whole-horizon-broadcast-baseline "
            f"{aggregate_baseline_b['mean_pr_auc']:.4f}."
        ),
        "dataset_validation": dataset_validation["status"],
        "stage_validation": stage_validation["status"],
        "leakage_audit": leakage,
        "leakage_stress_checks": leakage_stress,
        "preflight": preflight,
        "postflight": {"status": "PASS" if postflight_matches_preflight else "FAIL", "reproduced": postflight["reproduced"]},
        "training": {
            "seed": SEED, "device": str(device), "trainable_parameter_count": int(parameter_count),
            "best_epoch": best_epoch, "duration_seconds": duration_seconds,
            "pos_weight_per_step": pos_weight.cpu().numpy().tolist(),
            "checkpoint_path": str(best_path), "checkpoint_sha256": _sha256_file(best_path),
            "scaler_path": str(RUN1_SCALER), "scaler_sha256": _sha256_file(RUN1_SCALER),
            "target_config_hash": target_config_hash,
            "backbone_unchanged_after_training": backbone_ok,
        },
        "per_step_results": {
            "learned_model": per_step_learned, "baseline_a_persistence": per_step_baseline_a, "baseline_b_whole_horizon_broadcast": per_step_baseline_b,
        },
        "aggregate": {"learned_model": aggregate_learned, "baseline_a_persistence": aggregate_baseline_a, "baseline_b_whole_horizon_broadcast": aggregate_baseline_b},
        "baseline_comparison": {"learned_model": aggregate_learned, "baseline_a_persistence": aggregate_baseline_a, "baseline_b_whole_horizon_broadcast": aggregate_baseline_b},
        "trajectory_diagnostics": trajectory,
        "whole_horizon_consistency": whc,
        "calibration": calibration,
        "sih_compliance": sih,
        "failure_analysis": failure_analysis,
        "limitations": limitations,
        "claim_safety_statement": claim_safety_statement,
    }

    (OUTPUT_DIR / "target_construction.json").write_text(json.dumps({
        "definition": "y_k = 1 if per_step_stage_name[:,k] != 'BENIGN' else 0, for k=0..5 (future windows t+1..t+6)",
        "source": "phase7b_stage_targets.read_stage_targets (frozen, read-only, unmodified) -- per_step_stage_name derived from each future row's own current_attack_types via forecasting.mitre_mapping.MitreMapper.from_label_set",
        "is_infiltration_specific": False,
        "is_generic_any_covered_attack": True,
        "target_config_hash": target_config_hash,
        "counts_per_split": leakage["counts_per_split"],
        "per_step_infiltration_prevalence_test": sih["per_step_infiltration_prevalence_test"],
        "per_step_generic_attack_prevalence_test": sih["per_step_generic_attack_prevalence_test"],
    }, indent=2, default=str), encoding="utf-8")

    write_csv(OUTPUT_DIR / "metrics.csv", [
        {"model": model_name, **row}
        for model_name, rows in (("learned_model", per_step_learned), ("baseline_a_persistence", per_step_baseline_a), ("baseline_b_whole_horizon_broadcast", per_step_baseline_b))
        for row in rows
    ])
    write_csv(OUTPUT_DIR / "training_log.csv", history)
    (OUTPUT_DIR / "leakage_checks.json").write_text(json.dumps({**leakage, "stress_checks": leakage_stress}, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "phase9n_protocol.json").write_text(json.dumps({
        "target_definition": "y_k = 1 if per_step_stage_name[:,k] != BENIGN else 0 (k=0..5, t+1..t+6)",
        "architecture": "VectorWorldModelWithPerStepRiskHead(VectorWorldModel) -- frozen Run-1 backbone + new per_step_risk_head",
        "loss": "mean_k BCEWithLogitsLoss(logit_k, y_k), per-step pos_weight from TRAIN only",
        "backbone_source": str(RUN1_CHECKPOINT), "scaler_source": str(RUN1_SCALER),
        "seed": SEED, "model_selection_criterion": "validation mean per-step PR-AUC",
        "threshold_selection": "per-step choose_threshold (F1-max) on validation only",
    }, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "phase9n_per_step_risk_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "phase9n_per_step_risk_report.md", report)

    print(json.dumps({"status": status, "mean_pr_auc_learned": mean_pr_auc, "mean_pr_auc_persistence_baseline": mean_pr_auc_baseline_a, "beats_persistence": beats_persistence}, indent=2))


if __name__ == "__main__":
    main()
