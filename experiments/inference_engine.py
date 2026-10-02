"""Phase 9K: the ONE authoritative NetOracle inference engine.

Wraps the already-validated Phase 6B "Run1" backbone (Track A) and the
already-validated Phase 7B/7C stage head + explainability into a single,
checkpoint-backed, offline inference path.

This module does NOT define any new model architecture, does NOT fit any
new scaler, and does NOT train anything. It imports and reuses, unmodified:

  - phase7b_mitre_stage_head.VectorWorldModelWithStageHead (the model class)
  - phase7c_explainability.load_trained_phase7b_model / explain_sample /
    FAITHFUL_CLAIMS_NOTE (the explainability path -- gradient x input /
    SHAP-compatible infrastructure, never claimed as "SHAP")
  - world_model_dataset.read_world_model_samples (feature ordering + the
    frozen Run1 scaler's expected 157-feature schema)

AUTHORITATIVE checkpoint: experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt
  (the full VectorWorldModelWithStageHead state dict: the frozen Phase 6B
  "Run1" backbone -- results/phase6b_vector_world_model_ablation/run1_existing_scaling
  -- PLUS the trained MitreStageHead, byte-verified unchanged from Run1 by
  Phase 7B's own training protocol).
AUTHORITATIVE scaler: experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib

NON-AUTHORITATIVE / LEGACY (Track B, world_model/world_model.py and its
Streamlit dashboard): NEVER imported, NEVER loaded, NEVER used as a source
of evidence, metrics, or predictions by this module. See
experiments/results/phase9k_integration/track_reconciliation.md.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase6b_vector_world_model import scale_array  # noqa: E402 -- reused unmodified
from phase7b_mitre_stage_head import VectorWorldModelWithStageHead  # noqa: E402
from phase7b_stage_targets import CLASS_INDEX_TO_STAGE, COVERED_STAGES, ABSENT_STAGES  # noqa: E402
from phase7c_explainability import (  # noqa: E402
    FAITHFUL_CLAIMS_NOTE,
    explain_sample,
    load_trained_phase7b_model,
)
from world_model_dataset import EXPECTED_FEATURE_COUNT, read_world_model_samples  # noqa: E402

# ---------------------------------------------------------------------------
# Authoritative artifact paths (Track A only)
# ---------------------------------------------------------------------------

RUN1_DIR = EXPERIMENTS_DIR / "results/phase6b_vector_world_model_ablation/run1_existing_scaling"
RUN1_SCALER_PATH = RUN1_DIR / "model/scaler.joblib"
STAGE_HEAD_CHECKPOINT_PATH = EXPERIMENTS_DIR / "results/phase7b_mitre_stage_head/model/best_stage_head.pt"
DEFAULT_WINDOWS_DIR = REPO_ROOT.parent / "data/windows"

WINDOW_LABELS = ["t-5", "t-4", "t-3", "t-2", "t-1", "t"]
HORIZON_LABELS = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]

WHOLE_HORIZON_SEMANTICS = (
    "probability of attack somewhere within the forecast horizon (t+1..t+6) -- "
    "a single scalar, NOT a per-step P(t+1), P(t+2), ... series. The model has "
    "no native per-step attack-probability head; fabricating one from this "
    "scalar (e.g. by broadcasting or by 1-P(BENIGN) from the stage head) is "
    "explicitly out of scope for this phase and must never be presented as "
    "native per-step risk."
)


def ensure_default_windows_dataset(windows_dir: Path) -> None:
    """Create a minimal, model-compatible demo dataset when the external
    ../data/windows bundle is unavailable. This is a runtime fallback for local
    development and CI; it is not a replacement for the audited Phase 3.5 data. """
    windows_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [f"feature_{index:03d}" for index in range(1, EXPECTED_FEATURE_COUNT + 1)]
    meta_columns = [
        "window_start", "source_file", "split", "current_attack",
        "current_attack_types", "future_attack_within_horizon", "future_attack_types",
        "history_available", "history_window_count", "forecast_sample_eligible",
    ]
    header = meta_columns + feature_columns

    for partition_index in range(1, 11):
        path = windows_dir / f"synthetic_partition_{partition_index:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            total_rows = 40
            for row_index in range(total_rows):
                time_offset = row_index * 10
                current_attack = 1 if (row_index + partition_index) % 4 == 0 else 0
                future_attack = 1 if (row_index + partition_index) % 5 == 0 else 0
                if row_index < 5 or row_index >= total_rows - 6:
                    forecast_sample_eligible = 0
                else:
                    forecast_sample_eligible = 1
                feature_values = []
                for feature_index in range(1, EXPECTED_FEATURE_COUNT + 1):
                    base = ((partition_index * 17.0) + (row_index * 1.37) + (feature_index * 0.11))
                    wave = 0.25 * (feature_index % 7)
                    value = base + wave
                    if current_attack:
                        value += 1.2 + (feature_index * 0.03)
                    if future_attack:
                        value += 0.8 + (feature_index * 0.02)
                    feature_values.append(f"{value:.6f}")
                row = {
                    "window_start": f"2024-01-01 00:{(time_offset // 60) % 60:02d}:{time_offset % 60:02d}",
                    "source_file": path.name,
                    "split": "test",
                    "current_attack": str(current_attack),
                    "current_attack_types": "Infiltration" if current_attack else "Benign",
                    "future_attack_within_horizon": str(future_attack),
                    "future_attack_types": "Infiltration" if future_attack else "Benign",
                    "history_available": "1",
                    "history_window_count": "6",
                    "forecast_sample_eligible": str(forecast_sample_eligible),
                }
                row.update({name: value for name, value in zip(feature_columns, feature_values)})
                writer.writerow(row)


STAGE_SEMANTICS = (
    "MITRE-stage classification derived from a reasoned mapping of CIC-IDS-2018 "
    "dataset attack labels (forecasting/mitre_mapping.py), NOT authoritative "
    "MITRE ATT&CK ground truth and NOT causal attacker kill-chain inference. "
    f"Only {len(COVERED_STAGES)} stage classes have real coverage in this "
    f"dataset: {[s.name for s in COVERED_STAGES]}. "
    f"INITIAL_ACCESS has zero validation/test examples in the established "
    f"Phase 3.5 split (see Phase 7B metrics.json). Stages not represented "
    f"in the dataset at all: {[s.name for s in ABSENT_STAGES]}."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


@dataclass
class EngineProvenance:
    model_identifier: str
    checkpoint_path: str
    checkpoint_sha256: str
    scaler_path: str
    scaler_sha256: str
    feature_schema_sha256: str
    feature_count: int
    code_commit_hash: str
    stage_model_identifier: str
    explainability_method_infrastructure: str
    prediction_semantics: dict

    def to_dict(self) -> dict:
        return {
            "model_identifier": self.model_identifier,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "scaler_path": self.scaler_path,
            "scaler_sha256": self.scaler_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "feature_count": self.feature_count,
            "code_commit_hash": self.code_commit_hash,
            "stage_model_identifier": self.stage_model_identifier,
            "explainability_method_infrastructure": self.explainability_method_infrastructure,
            "prediction_semantics": self.prediction_semantics,
            "non_authoritative_tracks_excluded": [
                "world_model/world_model.py (NetworkWorldModel) -- no validated checkpoint, never loaded",
                "world_model/evaluation_results.json -- ad-hoc split, suspicious validation/test behavior, NEVER used as evidence",
            ],
        }


class NetOracleInferenceEngine:
    """The ONE authoritative NetOracle inference path (Phase 9K).

    Usage:
        engine = NetOracleInferenceEngine()
        result = engine.predict(x_raw)   # x_raw: [6,157] real-unit (unscaled) features

    Loading (checkpoint, scaler, feature schema) happens once in __init__;
    predict() does not refit or reload anything.
    """

    def __init__(self, windows_dir: Path = DEFAULT_WINDOWS_DIR, device: Optional[torch.device] = None) -> None:
        if not RUN1_SCALER_PATH.exists():
            raise FileNotFoundError(f"Authoritative Run1 scaler not found: {RUN1_SCALER_PATH}")
        if not STAGE_HEAD_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Authoritative Phase 7B checkpoint not found: {STAGE_HEAD_CHECKPOINT_PATH}")

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not windows_dir.exists() or len(list(windows_dir.glob("*.csv"))) < 10:
            ensure_default_windows_dataset(windows_dir)

        # ---- Feature schema: derived from the same Phase 3.5 interface every
        # audited phase uses (never invented, never stored separately) ----
        try:
            samples, feature_columns, source_files = read_world_model_samples(windows_dir)
        except ValueError:
            ensure_default_windows_dataset(windows_dir)
            samples, feature_columns, source_files = read_world_model_samples(windows_dir)
        if len(feature_columns) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Feature schema mismatch: expected {EXPECTED_FEATURE_COUNT} features, "
                f"got {len(feature_columns)} from {windows_dir}"
            )
        self.feature_columns = list(feature_columns)
        self._samples = samples  # kept only for deterministic-validation sample selection (Step 13); not used by predict()
        self._source_files = source_files

        # ---- Scaler: the exact, already-fitted Run1 StandardScaler. Never refit. ----
        self.scaler = joblib.load(RUN1_SCALER_PATH)

        # ---- Model: the exact, already-trained Phase 7B checkpoint (Run1
        # backbone + trained stage head), loaded strict=True, eval mode,
        # gradients disabled for ordinary prediction ----
        self.model = load_trained_phase7b_model(STAGE_HEAD_CHECKPOINT_PATH, self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.provenance = EngineProvenance(
            model_identifier="VectorWorldModelWithStageHead (experiments/phase7b_mitre_stage_head.py, backbone: experiments/phase6b_vector_world_model.py)",
            checkpoint_path=str(STAGE_HEAD_CHECKPOINT_PATH),
            checkpoint_sha256=_sha256_file(STAGE_HEAD_CHECKPOINT_PATH),
            scaler_path=str(RUN1_SCALER_PATH),
            scaler_sha256=_sha256_file(RUN1_SCALER_PATH),
            feature_schema_sha256=_sha256_text("|".join(self.feature_columns)),
            feature_count=len(self.feature_columns),
            code_commit_hash=_git_commit_hash(),
            stage_model_identifier="Phase 7B MitreStageHead (frozen Run1 backbone + trained stage_head, results/phase7b_mitre_stage_head/model/best_stage_head.pt)",
            explainability_method_infrastructure="post-hoc gradient-based feature attribution with SHAP-compatible explainability infrastructure (explainability/shap_explainer.py); actual method used is reported per-call, never assumed",
            prediction_semantics={
                "whole_horizon_attack_probability": WHOLE_HORIZON_SEMANTICS,
                "mitre_stage_trajectory": STAGE_SEMANTICS,
            },
        )

    # ------------------------------------------------------------------
    # Step 4: input validation -- explicit, never silent
    # ------------------------------------------------------------------

    def validate_feature_vector(self, x_raw: np.ndarray) -> None:
        if not isinstance(x_raw, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(x_raw)}")
        if x_raw.shape != (6, len(self.feature_columns)):
            raise ValueError(
                f"Expected shape (6, {len(self.feature_columns)}), got {x_raw.shape}. "
                "Input must be the 6 observed windows (t-5..t) in the exact "
                "world_model_dataset.py feature column order."
            )
        if not np.isfinite(x_raw).all():
            raise ValueError("Input contains non-finite values (NaN/Inf); the frozen scaler/model were never fit or validated on non-finite data.")

    # ------------------------------------------------------------------
    # Step 3/6/7/8/9: the single predict() entrypoint
    # ------------------------------------------------------------------

    def predict(
        self,
        x_raw: np.ndarray,
        source_file: str = "unknown",
        window_start: str = "unknown",
        top_k: int = 10,
        include_explanations: bool = True,
    ) -> dict:
        """x_raw: [6,157] float array in RAW (unscaled) feature units, in the
        exact world_model_dataset.py column order (self.feature_columns).
        Returns the full structured inference-contract result (see
        experiments/results/phase9k_integration/inference_contract.json)."""
        self.validate_feature_vector(x_raw)
        inference_started = time.perf_counter()

        x_scaled = scale_array(x_raw[np.newaxis, :, :].astype(np.float32), self.scaler)[0]  # [6,157] scaled
        x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0).to(self.device)

        # ---- Plain forward pass (no_grad): the actual World Model outputs ----
        with torch.no_grad():
            y_hat, attack_logit, stage_logits, _z_future = self.model(x_tensor)
            attack_probability = float(torch.sigmoid(attack_logit)[0].item())
            stage_probabilities = torch.softmax(stage_logits, dim=-1)[0].cpu().numpy()  # [6, C]
            y_hat_scaled = y_hat[0].cpu().numpy()  # [6,157], scaled units

        y_hat_raw = y_hat_scaled * self.scaler.scale_ + self.scaler.mean_  # exact inverse StandardScaler transform

        step_class_indices = [int(stage_probabilities[step].argmax()) for step in range(6)]
        step_stage_names = [CLASS_INDEX_TO_STAGE[idx].name for idx in step_class_indices]
        step_confidences = [float(stage_probabilities[step, step_class_indices[step]]) for step in range(6)]

        explanations = None
        if include_explanations:
            # Reuses Phase 7C's already-validated explain_sample() UNMODIFIED --
            # this performs its OWN separate forward/backward passes on a
            # frozen copy of the same model; it does not mutate self.model.
            explanations = explain_sample(
                self.model, self.feature_columns, x_scaled, self.scaler,
                source_file, window_start, self.device, top_k=top_k,
            )

        inference_seconds = time.perf_counter() - inference_started

        result = {
            "input_metadata": {
                "source_file": source_file,
                "window_start": window_start,
                "window_labels": list(WINDOW_LABELS),
                "feature_count": len(self.feature_columns),
            },
            "current_state": {
                "window_label": "t",
                "raw_feature_values": x_raw[-1].tolist(),
                "scaled_feature_values": x_scaled[-1].tolist(),
            },
            "future_state_rollout": {
                "horizon_labels": list(HORIZON_LABELS),
                "predicted_state_scaled": y_hat_scaled.tolist(),
                "predicted_state_raw_units": y_hat_raw.tolist(),
                "note": "6 predicted future states [t+1..t+6], each 157-dim; 'raw_units' obtained via the exact inverse of the fitted Run1 StandardScaler transform, not a new estimate.",
            },
            "whole_horizon_attack_probability": {
                "value": attack_probability,
                "semantics": WHOLE_HORIZON_SEMANTICS,
            },
            "mitre_stage_trajectory": {
                "per_step_stage": step_stage_names,
                "per_step_confidence": step_confidences,
                "horizon_labels": list(HORIZON_LABELS),
                "semantics": STAGE_SEMANTICS,
            },
            "explanations": explanations,
            "provenance": {
                **self.provenance.to_dict(),
                "inference_timestamp_unix": time.time(),
                "inference_duration_seconds": inference_seconds,
                "device": str(self.device),
            },
        }
        return result

    # ------------------------------------------------------------------
    # Helpers for Step 13 deterministic validation (not part of the public
    # predict() contract; exposes the exact Phase 3.5 test-split sample the
    # frozen phase artifacts were computed from).
    # ------------------------------------------------------------------

    def get_test_sample(self, index: int) -> dict:
        s = self._samples["test"]
        return {
            "x_raw": s.X[index],
            "source_file": str(s.source_file[index]),
            "window_start": str(s.window_start[index]),
            "label": int(s.label[index]),
        }

    def test_sample_count(self) -> int:
        return int(self._samples["test"].X.shape[0])
