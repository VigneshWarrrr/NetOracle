"""Phase 7C: post-hoc explainability adapter around the FROZEN Phase 7B model.

Nothing in this file modifies `explainability/shap_explainer.py` or
`experiments/phase7b_mitre_stage_head.py`. Both are imported and reused
exactly as they exist:

  - `TemporalAwareExplainer` subclasses `WorldModelExplainer` and overrides
    ONLY `_aggregate_sequence_importance`, so the 6-window temporal axis is
    preserved instead of being averaged away. Every other method (SHAP
    dispatch with its existing zero-vector standardized-space background,
    the gradient x input fallback, `explain_attack_risk`,
    `explain_future_state`, `_build_result`, `_generate_summary`,
    `_prepare_tensor`, ...) is inherited unchanged.
  - `StageLogitView` is a thin `nn.Module` wrapper: its `forward()` calls the
    frozen `VectorWorldModelWithStageHead` exactly as-is and returns one
    well-defined slice of its `stage_logits` output. It adds no parameters
    and mutates nothing.
  - `explain_sample()` orchestrates both to produce one structured,
    analyst-readable explanation for a single [6,157] observed history.

FAITHFUL vs NON-FAITHFUL claims (see FAITHFUL_CLAIMS_NOTE below): attribution
values describe what drove THIS MODEL's own score, computed from real
gradients/SHAP values on the frozen checkpoint. They are not proof of
real-world causality, attacker intent, or a confirmed MITRE technique.
MITRE technique IDs come from the separate, already-audited
`forecasting.mitre_mapping` label-to-technique table, never from this
attribution layer. `AttackChainBuilder.predict_next_stage()` (a `+1`
placeholder heuristic) is never used here as evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # NetOracle/, for `explainability`

from explainability.shap_explainer import ExplanationResult, WorldModelExplainer
from phase7b_mitre_stage_head import VectorWorldModelWithStageHead
from phase7b_stage_targets import CLASS_INDEX_TO_STAGE, NUM_STAGE_CLASSES

FAITHFUL_CLAIMS_NOTE = (
    "Attribution values indicate how much each observed feature, at each of the 6 observed "
    "windows (t-5..t), contributed to THIS MODEL's own risk score or stage-class logit, as "
    "computed by gradient-based or SHAP attribution on the frozen Phase 7B checkpoint. They "
    "are NOT proof of real-world causality, attacker intent, or a confirmed MITRE technique. "
    "MITRE technique IDs come from a separate, independently-audited label-to-technique mapping "
    "(forecasting/mitre_mapping.py) and are never derived from this attribution. "
    "AttackChainBuilder.predict_next_stage() is a +1 placeholder heuristic and is never used "
    "here as evidence."
)

WINDOW_LABELS = ["t-5", "t-4", "t-3", "t-2", "t-1", "t"]


class TemporalAwareExplainer(WorldModelExplainer):
    """WorldModelExplainer with ONLY `_aggregate_sequence_importance` overridden.

    The base implementation (`explainability/shap_explainer.py`, unmodified)
    collapses BOTH the batch axis and the 6-window sequence axis down to one
    score per feature (`while values.ndim > 1: values = values.mean(axis=0)`).
    This override collapses only the batch axis, keeping the window axis, so
    the per-window breakdown survives into `feature_scores` (as
    "t-5|feature", ..., "t|feature" keys, so the inherited `_build_result`'s
    ranking/top_k/summary logic keeps working completely unchanged) and, more
    directly, into `ExplanationResult.raw_values` (already passed through
    untouched by the base class) which `explain_sample()` reshapes into the
    canonical [6, 157] ndarray.
    """

    def _aggregate_sequence_importance(self, values) -> dict[str, float]:
        values = np.abs(np.asarray(values))
        while values.ndim > 2:
            values = values.mean(axis=0)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        window_count, raw_feature_count = values.shape
        feature_count = min(raw_feature_count, len(self.feature_names))
        feature_scores: dict[str, float] = {}
        for window_index in range(min(window_count, len(WINDOW_LABELS))):
            window_label = WINDOW_LABELS[window_index]
            for feature_index in range(feature_count):
                key = f"{window_label}|{self.feature_names[feature_index]}"
                feature_scores[key] = float(values[window_index, feature_index])
        return feature_scores


class StageLogitView(nn.Module):
    """Thin, parameter-free wrapper exposing one well-defined view of the
    frozen model's stage_logits output, so it can be handed to
    WorldModelExplainer (which expects a model whose forward(x) returns
    either a plain tensor or a tuple) with zero change to
    phase7b_mitre_stage_head.py.

    step=None: returns the full six-step stage_logits [B,6,C] -- combined
        with explain_attack_risk's target_index, this averages the selected
        class's logit across all 6 steps.
    step=k (0..5): returns stage_logits[:, k, :] [B,C] -- explains the
        prediction for future step t+(k+1) specifically.

    forward() calls the wrapped model exactly as-is; no parameters are
    added, no weights are touched, no Phase 7B behavior changes.
    """

    def __init__(self, model: VectorWorldModelWithStageHead, step: int | None = None) -> None:
        super().__init__()
        if step is not None and not (0 <= step < 6):
            raise ValueError(f"step must be None or in [0,6), got {step}")
        self.model = model
        self.step = step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, stage_logits, _ = self.model(x)
        if self.step is None:
            return stage_logits
        return stage_logits[:, self.step, :]


def load_trained_phase7b_model(checkpoint_path: Path, device: torch.device) -> VectorWorldModelWithStageHead:
    """Loads the FULL, already-trained Phase 7B checkpoint (frozen backbone
    weights + the trained stage_head from results/phase7b_mitre_stage_head/model/best_stage_head.pt).

    This is deliberately distinct from `phase7b_mitre_stage_head.load_frozen_backbone`
    (imported nowhere by this file), which is designed for the Run-1
    backbone-only checkpoint and intentionally accepts missing stage_head.*
    keys (strict=False, freshly-initialized head). Here every key must be
    present -- strict=True -- since this checkpoint was saved by
    `phase7b_train_stage_head.py` as a complete `model.state_dict()` after
    stage-head training finished.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase 7B checkpoint not found: {checkpoint_path}")
    model = VectorWorldModelWithStageHead().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _freeze_all_parameters(model: nn.Module) -> None:
    """Explanation must never update anything. Every parameter (backbone AND
    stage_head) is frozen before any explanation is generated -- this is
    strictly stronger than Phase 7B's freeze (which left stage_head
    trainable) and is safe: gradient x input / SHAP attribution only needs
    d(output)/d(input), never d(output)/d(parameter), so freezing every
    parameter does not change what is computed, only removes any possibility
    of a stray .grad being populated on a weight."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _winning_step_and_class(step_class_indices: list[int], step_confidences: list[float]) -> tuple[int, int]:
    """Deterministic analog of phase7b_train_stage_head.aggregate_predicted_stage,
    operating on class indices directly: pick the step whose predicted class
    has the highest MitreStage.value; ties broken by (a) highest confidence,
    then (b) earliest step index, for full determinism."""
    stage_values = [CLASS_INDEX_TO_STAGE[index].value for index in step_class_indices]
    max_value = max(stage_values)
    candidate_steps = [i for i, v in enumerate(stage_values) if v == max_value]
    candidate_steps.sort(key=lambda i: (-step_confidences[i], i))
    winning_step = candidate_steps[0]
    return winning_step, step_class_indices[winning_step]


def top_k_per_window(attribution_2d: np.ndarray, feature_names: list[str], k: int = 10) -> list[list[dict[str, object]]]:
    """attribution_2d: [6, F]. Returns, for EACH of the 6 windows independently,
    its own top-k (feature, score) ranking -- distinct from WorldModelExplainer's
    globally-ranked top_features (which ranks all 6*F window-feature pairs together)."""
    result: list[list[dict[str, object]]] = []
    for window_index in range(attribution_2d.shape[0]):
        row = attribution_2d[window_index]
        order = np.argsort(-row)[:k]
        result.append(
            [
                {"feature": feature_names[i], "importance": float(row[i]), "window": WINDOW_LABELS[window_index]}
                for i in order
            ]
        )
    return result


def explain_sample(
    model: VectorWorldModelWithStageHead,
    feature_names: list[str],
    x_scaled: np.ndarray,
    scaler,
    source_file: str,
    window_start: str,
    device: torch.device,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """x_scaled: [6, 157] float32, ALREADY standardized by the exact Run-1
    scaler (same convention the frozen model expects). Takes ONLY the
    observed history as model input -- no label, no future information, no
    stage/attack target is ever passed to this function or to model.forward().
    Fully post-hoc: uses its own separate forward/backward passes; never
    calls an optimizer; never mutates a parameter (see freeze above and the
    Phase 7C test suite's checkpoint-integrity tests).
    """
    if len(feature_names) != 157:
        raise ValueError(f"Expected 157 feature names, got {len(feature_names)}")
    if x_scaled.shape != (6, 157):
        raise ValueError(f"Expected x_scaled shape (6,157), got {x_scaled.shape}")

    model.eval()
    _freeze_all_parameters(model)

    x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0).to(device)

    # ---- Plain frozen-inference forward pass (establishes ground truth to cross-check) ----
    with torch.no_grad():
        y_hat, attack_logit, stage_logits, z_future = model(x_tensor)
        attack_probability = float(torch.sigmoid(attack_logit)[0].item())
        stage_probabilities = torch.softmax(stage_logits, dim=-1)[0].cpu().numpy()  # [6, C]

    step_class_indices = [int(stage_probabilities[step].argmax()) for step in range(6)]
    step_stage_names = [CLASS_INDEX_TO_STAGE[index].name for index in step_class_indices]
    step_confidences = [float(stage_probabilities[step, step_class_indices[step]]) for step in range(6)]
    winning_step, winning_class_index = _winning_step_and_class(step_class_indices, step_confidences)
    overall_stage = CLASS_INDEX_TO_STAGE[winning_class_index].name
    overall_confidence = step_confidences[winning_step]

    # ---- Attack-risk attribution: explain the model directly (its forward()
    # already returns attack_logit at tuple position 1, which
    # _extract_attack_output already handles unmodified) ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    attack_explainer = TemporalAwareExplainer(model, feature_names, device=str(device))
    attack_result: ExplanationResult = attack_explainer.explain_attack_risk(x_scaled, top_k=top_k)
    attack_attribution_2d = np.abs(np.asarray(attack_result.raw_values)).reshape(6, len(feature_names))

    # ---- Stage attribution: explain the winning step's winning class via
    # the thin StageLogitView wrapper (Phase 6B/7B code untouched) ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    stage_view = StageLogitView(model, step=winning_step)
    _freeze_all_parameters(stage_view)
    stage_explainer = TemporalAwareExplainer(stage_view, feature_names, device=str(device))
    stage_result: ExplanationResult = stage_explainer.explain_attack_risk(
        x_scaled, target_index=winning_class_index, top_k=top_k
    )
    stage_attribution_2d = np.abs(np.asarray(stage_result.raw_values)).reshape(6, len(feature_names))

    top_attack_per_window = top_k_per_window(attack_attribution_2d, feature_names, top_k)
    top_stage_per_window = top_k_per_window(stage_attribution_2d, feature_names, top_k)

    # ---- Raw-unit feature values, recovered via inverse standardization
    # through the exact Run-1 scaler (mathematically exact for StandardScaler) ----
    raw_values_2d = x_scaled * scaler.scale_ + scaler.mean_

    return {
        "source_file": source_file,
        "window_start": window_start,
        "feature_names": list(feature_names),
        "window_labels": list(WINDOW_LABELS),
        "current_state_evidence": {
            "window_label": "t",
            "attack_risk_attribution": attack_attribution_2d[-1].tolist(),
            "stage_attribution": stage_attribution_2d[-1].tolist(),
            "top_attack_risk_features": top_attack_per_window[-1],
            "top_stage_features": top_stage_per_window[-1],
            "raw_feature_values": raw_values_2d[-1].tolist(),
        },
        "temporal_evidence": {
            "window_labels": list(WINDOW_LABELS),
            "attack_risk_attribution": attack_attribution_2d.tolist(),
            "stage_attribution": stage_attribution_2d.tolist(),
            "top_attack_risk_features_per_window": top_attack_per_window,
            "top_stage_features_per_window": top_stage_per_window,
            "raw_feature_values": raw_values_2d.tolist(),
        },
        "future_attack_risk_prediction": {
            "attack_probability": attack_probability,
            "explanation_method": attack_result.method,
            "explanation_reported_prediction": attack_result.prediction,
        },
        "mitre_stage_prediction": {
            "per_step_stage": step_stage_names,
            "per_step_confidence": step_confidences,
            "overall_stage": overall_stage,
            "overall_confidence": overall_confidence,
            "explained_step_index": winning_step,
            "explained_stage_class": overall_stage,
            "explanation_method": stage_result.method,
            "explanation_reported_prediction": stage_result.prediction,
        },
        "faithful_claims_only": FAITHFUL_CLAIMS_NOTE,
    }
