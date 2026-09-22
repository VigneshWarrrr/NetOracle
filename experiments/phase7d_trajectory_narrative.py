"""Phase 7D: post-hoc six-step attack trajectory / forecast narrative.

A pure aggregation/presentation layer over already-frozen Phase 6B/7B/7C
outputs. Nothing here trains, fine-tunes, or modifies any model, checkpoint,
dataset, or prior result. Reuses `phase7c_explainability.py`'s public
classes (`StageLogitView`, `TemporalAwareExplainer`) exactly as they are --
it does not modify that file, it just calls its exported API one extra time
per future step (Phase 7C's own `explain_sample` only explains the single
"winning" step; Phase 7D needs genuine attribution for all 6 future steps,
so it loops the same, unmodified explainer over step=0..5).

============================================================
NATIVE MODEL OUTPUTS (Category A -- see Phase 7D inspection report):
============================================================
  - `attack_probability`: ONE number covering the full 6-step/60s horizon
    (VectorWorldModelWithStageHead's AttackForecastHead mean-pools z_future
    across all 6 steps BEFORE its final layer -- there is no per-step
    attack probability anywhere in the frozen model, and this file never
    fabricates one).
  - six per-step MITRE stage predictions + softmax confidences (genuinely
    independent per-step classifications -- no smoothing/recurrence across
    steps in how they were trained).

============================================================
DERIVED QUANTITIES (Category B -- computed here, clearly labeled):
============================================================
  - `overall_predicted_stage`/`overall_confidence`: a deterministic rule
    (highest MitreStage.value among the 6 steps, tie broken by confidence
    then step index) applied to the 6 genuine per-step outputs.
  - `non_benign_probability` per step = 1 - P(BENIGN) from that step's own
    softmax. THIS IS NEVER TO BE CALLED "attack probability" / "attack
    risk" / "infiltration probability" anywhere in this codebase -- see
    NON_BENIGN_LABEL below, which is the only string ever attached to it.
  - `top2_confidence_margin` per step = top1_prob - top2_prob.
  - per-step top-K driving features: gradient attribution (via the
    unmodified Phase 7C explainer classes) explaining THAT STEP's own
    predicted stage class.
  - the disagreement flag/warning (see `detect_disagreement`).

============================================================
NARRATIVE (Category C -- heuristic text, strictly templated from A/B only):
============================================================
Never uses `AttackChainBuilder.predict_next_stage()` (a `+1` placeholder
heuristic, not a model output). Never asserts attacker intent, plans,
identity, target host/asset, certainty, or calibrated uncertainty (none of
that is implemented anywhere in this stack). Describes the six per-step
stage outputs as "independent future-stage assessments" / "predicted
future-stage trajectory," never as a guaranteed kill-chain progression.

MITRE TECHNIQUE IDs are deliberately NOT included in trajectory records:
`forecasting.mitre_mapping.MitreMapper`'s technique-ID table is keyed by
attack LABEL (e.g. "SQL Injection" -> T1190), not by MitreStage. The model
predicts a STAGE, not a label, so there is no label to look a technique up
from at inference time -- inventing a stage-to-technique collapse (picking
one label's technique to represent a whole stage) would fabricate MITRE
evidence the model never produced. Omitting is the honest choice here, not
an oversight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # NetOracle/, for `forecasting`

from phase7b_stage_targets import CLASS_INDEX_TO_STAGE, NUM_STAGE_CLASSES
from phase7c_explainability import StageLogitView, TemporalAwareExplainer, WINDOW_LABELS

BENIGN_STAGE_NAME = "BENIGN"
BENIGN_CLASS_INDEX = next(index for index, stage in CLASS_INDEX_TO_STAGE.items() if stage.name == BENIGN_STAGE_NAME)

NON_BENIGN_LABEL = "MITRE-stage-derived non-benign signal (NOT attack probability, NOT attack risk, NOT infiltration probability)"

# ------------------------------------------------------------------
# Disagreement-rule thresholds. Deterministic constants, documented,
# not fit or tuned against any data -- chosen as round, defensible
# operating points (low native risk vs. a confident non-benign stage
# call), consistent with the "do not tune thresholds" constraint.
# ------------------------------------------------------------------
LOW_ATTACK_PROBABILITY_THRESHOLD = 0.30
HIGH_STAGE_CONFIDENCE_THRESHOLD = 0.70

# Minimum change in the derived non-benign signal (step 1 vs step 6) before
# the narrative calls it "increasing"/"decreasing" rather than "roughly
# stable" -- avoids over-reading floating-point noise as a trend.
TREND_EPSILON = 0.05

TOP_K_FEATURES_PER_STEP = 5


def per_step_stage_predictions(stage_probabilities: np.ndarray) -> list[dict[str, object]]:
    """stage_probabilities: [6, NUM_STAGE_CLASSES] softmax, ALREADY computed
    by a plain frozen forward pass (no gradient, no explanation involved).
    Returns, per step: predicted stage name, its confidence, the top1-top2
    margin, and the derived non_benign_probability. Pure arithmetic on
    already-computed native model outputs -- no new model call here."""
    steps: list[dict[str, object]] = []
    for step in range(6):
        probabilities = stage_probabilities[step]
        order = np.argsort(-probabilities)
        top1_index, top2_index = int(order[0]), int(order[1])
        predicted_stage = CLASS_INDEX_TO_STAGE[top1_index].name
        steps.append(
            {
                "step": step + 1,
                "relative_window": f"t+{step + 1}",
                "predicted_stage": predicted_stage,
                "stage_confidence": float(probabilities[top1_index]),
                "top2_confidence_margin": float(probabilities[top1_index] - probabilities[top2_index]),
                "non_benign_probability": float(1.0 - probabilities[BENIGN_CLASS_INDEX]),
                "non_benign_probability_label": NON_BENIGN_LABEL,
                "_predicted_class_index": top1_index,  # internal use only; stripped before persistence
            }
        )
    return steps


def _winning_step_and_class(steps: list[dict[str, object]]) -> tuple[int, int, str, float]:
    """Same deterministic aggregation POLICY as Phase 7B/7C (highest
    MitreStage.value wins; ties broken by confidence then step index),
    reimplemented here over the per-step dicts this module already built --
    does not call back into Phase 7B/7C code, but is the identical rule."""
    stage_values = [CLASS_INDEX_TO_STAGE[step["_predicted_class_index"]].value for step in steps]
    max_value = max(stage_values)
    candidates = [i for i, v in enumerate(stage_values) if v == max_value]
    candidates.sort(key=lambda i: (-steps[i]["stage_confidence"], i))
    winning_index = candidates[0]
    winning_step_record = steps[winning_index]
    return winning_index, winning_step_record["_predicted_class_index"], winning_step_record["predicted_stage"], winning_step_record["stage_confidence"]


def compute_step_attributions(
    model,
    x_scaled: np.ndarray,
    feature_names: list[str],
    steps: list[dict[str, object]],
    device: torch.device,
    top_k: int = TOP_K_FEATURES_PER_STEP,
    seed: int = 42,
) -> None:
    """Mutates `steps` in place, adding "top_k_driving_features" to each of
    the 6 step dicts. Explains EACH step's own predicted class via the
    unmodified Phase 7C StageLogitView/TemporalAwareExplainer, called once
    per step (Phase 7C's own explain_sample only explains the single
    aggregate-winning step; this loops the identical, unmodified mechanism
    over all 6). Purely post-hoc: no parameter is updated, no optimizer
    exists in this path."""
    for step_record in steps:
        step_index = step_record["step"] - 1
        class_index = step_record["_predicted_class_index"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        view = StageLogitView(model, step=step_index)
        for parameter in view.parameters():
            parameter.requires_grad_(False)
        explainer = TemporalAwareExplainer(view, feature_names, device=str(device))
        result = explainer.explain_attack_risk(x_scaled, target_index=class_index, top_k=top_k)
        attribution_2d = np.abs(np.asarray(result.raw_values)).reshape(6, len(feature_names))
        current_window_row = attribution_2d[-1]  # window "t", the most recent observed state
        order = np.argsort(-current_window_row)[:top_k]
        step_record["top_k_driving_features"] = [
            {"feature": feature_names[i], "importance": float(current_window_row[i])} for i in order
        ]


def detect_disagreement(attack_probability: float, steps: list[dict[str, object]]) -> dict[str, object]:
    """Deterministic, non-silent disagreement detector between the two
    separately-trained model heads. Never resolves the disagreement (e.g.
    never averages the two signals or picks a "winner") -- only reports it."""
    thresholds = {
        "low_attack_probability_threshold": LOW_ATTACK_PROBABILITY_THRESHOLD,
        "high_stage_confidence_threshold": HIGH_STAGE_CONFIDENCE_THRESHOLD,
    }
    if attack_probability > LOW_ATTACK_PROBABILITY_THRESHOLD:
        return {"detected": False, "warning": None, "flagged_steps": [], "thresholds": thresholds}

    flagged = [
        step
        for step in steps
        if step["predicted_stage"] != BENIGN_STAGE_NAME and step["stage_confidence"] >= HIGH_STAGE_CONFIDENCE_THRESHOLD
    ]
    if not flagged:
        return {"detected": False, "warning": None, "flagged_steps": [], "thresholds": thresholds}

    flagged_steps = [step["step"] for step in flagged]
    flagged_stages = [step["predicted_stage"] for step in flagged]
    warning = (
        f"DISAGREEMENT BETWEEN MODEL SIGNALS: the native horizon attack_probability "
        f"({attack_probability:.4f}) is at or below the configured low-risk threshold "
        f"({LOW_ATTACK_PROBABILITY_THRESHOLD}), but the MITRE-stage classifier predicts a "
        f"non-benign stage with confidence >= {HIGH_STAGE_CONFIDENCE_THRESHOLD} at step(s) "
        f"{flagged_steps} ({flagged_stages}). These are two separately-trained model outputs "
        f"and this difference is NOT automatically resolved here -- both signals should be "
        f"reviewed independently rather than assuming either one is correct."
    )
    return {"detected": True, "warning": warning, "flagged_steps": flagged_steps, "thresholds": thresholds}


def generate_narrative(record: dict[str, object]) -> str:
    """Strictly templated from the A/B quantities already computed above.
    No new information is introduced here. See the module docstring for the
    exact list of claims this function is forbidden from making."""
    lines: list[str] = []

    attack_probability = record["attack_probability"]
    lines.append(
        f"Native horizon attack probability (the model's own single risk estimate for the "
        f"complete 6-step / 60-second forecast horizon): {attack_probability:.4f}."
    )

    overall_stage = record["overall_predicted_stage"]
    overall_confidence = record["overall_confidence"]
    lines.append(
        f"Overall predicted MITRE stage, derived by combining the six independent per-step "
        f"assessments: {overall_stage} (confidence {overall_confidence:.4f})."
    )

    stage_sequence = [step["predicted_stage"] for step in record["steps"]]
    if len(set(stage_sequence)) == 1:
        lines.append(
            f"All six independent future-stage assessments (t+1 through t+6) predict "
            f"{stage_sequence[0]}."
        )
    else:
        lines.append(
            "Predicted future-stage trajectory across the six independent per-step assessments "
            "(t+1 -> t+6): " + " -> ".join(stage_sequence) + "."
        )

    non_benign_values = [step["non_benign_probability"] for step in record["steps"]]
    delta = non_benign_values[-1] - non_benign_values[0]
    if delta > TREND_EPSILON:
        trend = "increasing"
    elif delta < -TREND_EPSILON:
        trend = "decreasing"
    else:
        trend = "roughly stable"
    lines.append(
        f"The {NON_BENIGN_LABEL} is {trend} across the horizon "
        f"(step 1: {non_benign_values[0]:.4f}, step 6: {non_benign_values[-1]:.4f})."
    )

    strongest_step = max(record["steps"], key=lambda step: step["stage_confidence"])
    if strongest_step.get("top_k_driving_features"):
        top_feature = strongest_step["top_k_driving_features"][0]
        lines.append(
            f"The feature contributing most (per gradient-based attribution on the frozen model) "
            f"to step {strongest_step['step']}'s predicted stage ({strongest_step['predicted_stage']}) "
            f"was '{top_feature['feature']}'. This describes model attribution only, not a "
            f"real-world or causal explanation."
        )

    if record["disagreement"]["detected"]:
        lines.append(record["disagreement"]["warning"])

    lines.append(
        "These six future-stage assessments are independent per-step model outputs, not a "
        "guaranteed or causally-linked kill-chain progression."
    )

    return " ".join(lines)


def build_trajectory_record(
    model,
    feature_names: list[str],
    x_scaled: np.ndarray,
    scaler,
    source_file: str,
    window_start: str,
    device: torch.device,
    top_k: int = TOP_K_FEATURES_PER_STEP,
    seed: int = 42,
) -> dict[str, object]:
    """Builds one complete trajectory record for a single [6,157] observed,
    already-standardized history. Takes ONLY the observed history as model
    input -- no ground-truth label, no future information, is ever passed
    in or read by this function."""
    if x_scaled.shape != (6, 157):
        raise ValueError(f"Expected x_scaled shape (6,157), got {x_scaled.shape}")
    if len(feature_names) != 157:
        raise ValueError(f"Expected 157 feature names, got {len(feature_names)}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, attack_logit, stage_logits, _ = model(x_tensor)
        attack_probability = float(torch.sigmoid(attack_logit)[0].item())
        stage_probabilities = torch.softmax(stage_logits, dim=-1)[0].cpu().numpy()  # [6, C]

    steps = per_step_stage_predictions(stage_probabilities)
    winning_index, winning_class_index, overall_stage, overall_confidence = _winning_step_and_class(steps)
    compute_step_attributions(model, x_scaled, feature_names, steps, device, top_k=top_k, seed=seed)
    for step_record in steps:
        step_record.pop("_predicted_class_index", None)

    disagreement = detect_disagreement(attack_probability, steps)

    record: dict[str, object] = {
        "source_file": source_file,
        "window_start": window_start,
        "attack_probability": attack_probability,
        "overall_predicted_stage": overall_stage,
        "overall_confidence": overall_confidence,
        "overall_stage_step": winning_index + 1,
        "steps": steps,
        "disagreement": disagreement,
        "mitre_technique_ids": None,
        "mitre_technique_ids_note": (
            "Not included: the model predicts a MITRE STAGE, not a specific attack label, and "
            "forecasting.mitre_mapping.MitreMapper's technique-ID table is keyed by label, not "
            "stage. A stage-to-technique lookup would require an arbitrary collapse Phase 7A "
            "never defined, so it is omitted rather than invented."
        ),
        "faithful_claims_only": (
            "Per-step stage predictions and confidences, and the horizon-level attack_probability, "
            "are direct frozen-model outputs. non_benign_probability, top2_confidence_margin, "
            "overall_predicted_stage, and the disagreement flag are post-hoc arithmetic derived "
            "from those outputs. None of this proves real-world causality, attacker intent, a "
            "confirmed MITRE technique, or a guaranteed attack progression."
        ),
    }
    record["narrative"] = generate_narrative(record)
    return record
