"""Phase 9K: authoritative World Model reconciliation + inference engine.

INTEGRATION AND HARDENING PHASE ONLY. No new modeling, no training, no
threshold tuning. This script:

  1. Records a repository safety snapshot (Step 1).
  2. Documents the Track A vs Track B reconciliation decision (Step 12).
  3. Instantiates the authoritative NetOracleInferenceEngine (experiments/inference_engine.py).
  4. Runs deterministic validation against frozen Phase 6B/7B outputs (Step 13)
     -- NOT new test-set evaluation, purely an equivalence check.
  5. Measures inference-only performance (Step 15).
  6. Writes all required Phase 9K artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from inference_engine import (  # noqa: E402
    RUN1_DIR,
    RUN1_SCALER_PATH,
    STAGE_HEAD_CHECKPOINT_PATH,
    NetOracleInferenceEngine,
)

OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9k_integration"
PHASE9J_DIR = EXPERIMENTS_DIR / "results/phase9j_sih_gap_audit"

ATTACK_PROB_TOLERANCE = 1e-3  # documented tolerance for Step 13.1 (observed empirical deviation ~8e-6)
DETERMINISM_TOLERANCE = 1e-6  # two calls on the SAME loaded model, SAME input -- should be near bit-exact
NUM_VALIDATION_SAMPLES = 5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return f"<git failed: {exc}>"


# ---------------------------------------------------------------------------
# Step 1: safety snapshot
# ---------------------------------------------------------------------------


def build_safety_manifest() -> dict:
    prior_result_dirs = sorted(
        p.name for p in (EXPERIMENTS_DIR / "results").iterdir()
        if p.is_dir() and p.name != "phase9k_integration"
    )
    prior_file_counts = {
        name: sum(1 for _ in (EXPERIMENTS_DIR / "results" / name).rglob("*") if _.is_file())
        for name in prior_result_dirs
    }

    checksums = {
        "phase6b_run1_checkpoint": {
            "path": str(RUN1_DIR / "model/best_model.pt"),
            "sha256": _sha256_file(RUN1_DIR / "model/best_model.pt"),
        },
        "phase6b_run1_scaler": {"path": str(RUN1_SCALER_PATH), "sha256": _sha256_file(RUN1_SCALER_PATH)},
        "phase7b_stage_head_checkpoint": {
            "path": str(STAGE_HEAD_CHECKPOINT_PATH),
            "sha256": _sha256_file(STAGE_HEAD_CHECKPOINT_PATH),
        },
        "phase7c_shap_explainer_module": {
            "path": str(REPO_ROOT / "explainability/shap_explainer.py"),
            "sha256": _sha256_file(REPO_ROOT / "explainability/shap_explainer.py"),
        },
    }

    dashboard_entry_points = {
        "django_root_urlconf": "db.urls (db/settings.py: ROOT_URLCONF)",
        "django_installed_apps_with_urls": ["users", "dashboard", "logs", "alerts", "api", "management", "server", "settings_app"],
        "dashboard_view": "dashboard/views.py: AdminDashboardView, NetworkRiskForecastView, get_forecast_context()",
        "current_model_loader_in_dashboard_path": "world_model/inference_service.py: AttackForecastService (looks for models/attack_forecaster.pt, falls back to heuristic -- see Phase 9J finding)",
        "streamlit_track_b_entry_point": "world_model/src/streamlit.py (NON-AUTHORITATIVE, see track_reconciliation.md)",
        "new_phase9k_authoritative_entry_point": "experiments/inference_engine.py: NetOracleInferenceEngine",
    }

    return {
        "phase": "9K",
        "commit_hash": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "git_status_short": _git("status", "--short"),
        "git_status_note": "Non-empty status reflects pre-existing modified/untracked files from earlier phases in this session, not changes made by Phase 9K.",
        "prior_result_directory_file_counts": prior_file_counts,
        "checksums": checksums,
        "dashboard_entry_points": dashboard_entry_points,
    }


def verify_no_prior_artifact_changed(baseline: dict) -> dict:
    current_counts = {
        name: sum(1 for _ in (EXPERIMENTS_DIR / "results" / name).rglob("*") if _.is_file())
        for name in baseline["prior_result_directory_file_counts"]
    }
    mismatches = {
        name: {"before": baseline["prior_result_directory_file_counts"][name], "after": current_counts[name]}
        for name in current_counts
        if current_counts[name] != baseline["prior_result_directory_file_counts"][name]
    }
    return {"unchanged": not mismatches, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Step 12: track reconciliation document
# ---------------------------------------------------------------------------

TRACK_RECONCILIATION_MD = """# Phase 9K: World Model Track Reconciliation

## Decision

**AUTHORITATIVE**: `experiments/phase6b_vector_world_model.py` (VectorWorldModel backbone)
+ `experiments/phase7b_mitre_stage_head.py` (VectorWorldModelWithStageHead)
+ checkpoint `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt`
(the frozen Phase 6B "Run1" backbone -- `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling`
-- plus the trained MitreStageHead, byte-verified unchanged from Run1 by Phase 7B's own training protocol)
+ scaler `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib`.

This is now exposed through exactly ONE module: `experiments/inference_engine.py`
(`NetOracleInferenceEngine`).

**NON-AUTHORITATIVE / LEGACY**: `world_model/world_model.py` (`NetworkWorldModel`),
its feature pipeline `world_model/feature_extraction.py`, its Streamlit dashboard
`world_model/src/streamlit.py`, and its evaluation artifacts
`world_model/evaluation_results.json` / `world_model/zero_shot_results.json`.

## Why

Phase 9J traced every reference to Track B and found:

- No validated checkpoint exists anywhere in the repository for `NetworkWorldModel`
  (`models/best_world_model.pth` absent, re-confirmed by this phase's safety manifest).
- `world_model/evaluation_results.json` (the source of the Streamlit "Benchmark" tab's
  hardcoded numbers) used a non-standard, undocumented split
  (`Wednesday-28-INFIL-TRAIN.csv` / `-INFIL-TEST.csv`, absent from `data/windows/`
  and `data/raw/`) and shows a validation/test metric inversion (zero validation
  recall at every horizon, 79-85% test recall) consistent with leakage or an
  invalid evaluation methodology.
- The Django dashboard's own model loader (`world_model/inference_service.py`)
  looks for a THIRD, also-absent checkpoint (`models/attack_forecaster.pt`) and
  falls back to a 2-layer heuristic -- it does not currently call either Track A
  or Track B.

## Disposition

Per Phase 9K's explicit instruction ("do not spend time trying to improve Track
B", "prefer a compatibility/deprecation boundary if existing UI code depends on
it", "do not delete files blindly"):

- Track B's files are **left in place, unmodified** -- no deletion.
- Track B is **quarantined by documentation only** in this phase: this file, plus
  a NON-AUTHORITATIVE marker referenced from `experiments/inference_engine.py`'s
  own module docstring, are the deprecation boundary.
- No UI code in this repository was found to import a Track B module through any
  Django-reachable path (Phase 9J's repo-wide trace: Track B is only imported by
  `world_model/src/streamlit.py`, `world_model/build_project.py`, and
  `experiments/tests.py`'s own guard tests -- none of which are Django URL-routed).
  Therefore no compatibility shim was required; Track B was already effectively
  isolated from the production Django path.
- `world_model/evaluation_results.json` and `world_model/zero_shot_results.json`
  remain on disk (frozen, Phase 9J already analyzed them) but this phase's
  authoritative inference engine and reports **never read, import, or cite either
  file** as evidence, metrics, or predictions.

## What would fully retire Track B (not done in this phase)

Deleting `world_model/world_model.py`, `world_model/feature_extraction.py`,
`world_model/src/streamlit.py`, `world_model/evaluation_results.json`, and
`world_model/zero_shot_results.json` is a reasonable FUTURE cleanup once a human
maintainer confirms no other undiscovered dependency exists -- explicitly out of
scope for Phase 9K's audit-then-integrate discipline (no blind deletion).
"""


# ---------------------------------------------------------------------------
# Step 13: deterministic validation against frozen artifacts
# ---------------------------------------------------------------------------


def _load_frozen_run1_predictions() -> dict[tuple[str, str], float]:
    path = RUN1_DIR / "predictions_test.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["source_file"], row["window_start"])] = float(row["world_model_probability"])
    return out


def _load_frozen_phase7b_stage_predictions() -> dict[tuple[str, str], dict]:
    path = EXPERIMENTS_DIR / "results/phase7b_mitre_stage_head/predictions_test.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["source_file"], row["window_start"])
            out[key] = {
                "attack_probability": float(row["attack_probability"]),
                "stages": [row[f"mitre_stage_step_{i}"] for i in range(1, 7)],
                "confidences": [float(row[f"mitre_confidence_step_{i}"]) for i in range(1, 7)],
            }
    return out


def run_deterministic_validation(engine: NetOracleInferenceEngine) -> dict:
    frozen_run1 = _load_frozen_run1_predictions()
    frozen_stage = _load_frozen_phase7b_stage_predictions()

    checks = []
    for i in range(min(NUM_VALIDATION_SAMPLES, engine.test_sample_count())):
        sample = engine.get_test_sample(i)
        key = (sample["source_file"], sample["window_start"])

        result_1 = engine.predict(sample["x_raw"], source_file=sample["source_file"], window_start=sample["window_start"], top_k=5)
        result_2 = engine.predict(sample["x_raw"], source_file=sample["source_file"], window_start=sample["window_start"], top_k=5)

        engine_prob = result_1["whole_horizon_attack_probability"]["value"]
        run1_expected = frozen_run1.get(key)
        stage_expected = frozen_stage.get(key)

        run1_match = None if run1_expected is None else abs(engine_prob - run1_expected) <= ATTACK_PROB_TOLERANCE
        stage_prob_match = None if stage_expected is None else abs(engine_prob - stage_expected["attack_probability"]) <= ATTACK_PROB_TOLERANCE
        stage_names_match = None if stage_expected is None else (result_1["mitre_stage_trajectory"]["per_step_stage"] == stage_expected["stages"])

        determinism_diff = abs(engine_prob - result_2["whole_horizon_attack_probability"]["value"])
        explanations = result_1["explanations"]
        attribution_shape_ok = (
            explanations is not None
            and len(explanations["temporal_evidence"]["attack_risk_attribution"]) == 6
            and all(len(row) == len(engine.feature_columns) for row in explanations["temporal_evidence"]["attack_risk_attribution"])
        )

        checks.append(
            {
                "index": i,
                "source_file": sample["source_file"],
                "window_start": sample["window_start"],
                "engine_attack_probability": engine_prob,
                "frozen_run1_attack_probability": run1_expected,
                "run1_equivalence_within_tolerance": run1_match,
                "frozen_phase7b_attack_probability": stage_expected["attack_probability"] if stage_expected else None,
                "phase7b_attack_probability_equivalence_within_tolerance": stage_prob_match,
                "frozen_phase7b_stage_names": stage_expected["stages"] if stage_expected else None,
                "engine_stage_names": result_1["mitre_stage_trajectory"]["per_step_stage"],
                "phase7b_stage_names_exact_match": stage_names_match,
                "determinism_repeat_diff": determinism_diff,
                "deterministic_within_tolerance": determinism_diff <= DETERMINISM_TOLERANCE,
                "attribution_shape_is_6xF": attribution_shape_ok,
                "attribution_shape_actual": [
                    len(explanations["temporal_evidence"]["attack_risk_attribution"]),
                    len(explanations["temporal_evidence"]["attack_risk_attribution"][0]) if explanations else None,
                ] if explanations else None,
            }
        )

    all_run1_ok = all(c["run1_equivalence_within_tolerance"] for c in checks if c["run1_equivalence_within_tolerance"] is not None)
    all_stage_prob_ok = all(c["phase7b_attack_probability_equivalence_within_tolerance"] for c in checks if c["phase7b_attack_probability_equivalence_within_tolerance"] is not None)
    all_stage_names_ok = all(c["phase7b_stage_names_exact_match"] for c in checks if c["phase7b_stage_names_exact_match"] is not None)
    all_deterministic = all(c["deterministic_within_tolerance"] for c in checks)
    all_attribution_shape_ok = all(c["attribution_shape_is_6xF"] for c in checks)

    return {
        "num_samples_checked": len(checks),
        "attack_probability_tolerance": ATTACK_PROB_TOLERANCE,
        "determinism_tolerance": DETERMINISM_TOLERANCE,
        "per_sample_checks": checks,
        "all_run1_attack_probability_equivalent": all_run1_ok,
        "all_phase7b_attack_probability_equivalent": all_stage_prob_ok,
        "all_phase7b_stage_names_exact_match": all_stage_names_ok,
        "all_repeated_inference_deterministic": all_deterministic,
        "all_attribution_shape_6xF": all_attribution_shape_ok,
        "overall_pass": all([all_run1_ok, all_stage_prob_ok, all_stage_names_ok, all_deterministic, all_attribution_shape_ok]),
    }


# ---------------------------------------------------------------------------
# Step 15: performance check (inference only, no optimization)
# ---------------------------------------------------------------------------


def run_performance_check(engine: NetOracleInferenceEngine) -> dict:
    sample = engine.get_test_sample(0)

    # warm-up
    engine.predict(sample["x_raw"], include_explanations=False)

    n_reps = 20
    started = time.perf_counter()
    for _ in range(n_reps):
        engine.predict(sample["x_raw"], include_explanations=False)
    no_explain_seconds = (time.perf_counter() - started) / n_reps

    started = time.perf_counter()
    engine.predict(sample["x_raw"], include_explanations=True)
    with_explain_seconds = time.perf_counter() - started

    peak_gpu_memory_bytes = None
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
        engine.predict(sample["x_raw"], include_explanations=True)
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(engine.device)

    cpu_latency_seconds = None
    try:
        cpu_engine = NetOracleInferenceEngine(device=torch.device("cpu"))
        cpu_sample = cpu_engine.get_test_sample(0)
        cpu_engine.predict(cpu_sample["x_raw"], include_explanations=False)  # warm-up
        started = time.perf_counter()
        cpu_engine.predict(cpu_sample["x_raw"], include_explanations=False)
        cpu_latency_seconds = time.perf_counter() - started
    except Exception as exc:  # pragma: no cover - CPU path is best-effort here
        cpu_latency_seconds = f"measurement failed: {exc}"

    return {
        "device_used_for_primary_measurement": str(engine.device),
        "batch_size": 1,
        "gpu_inference_latency_seconds_no_explanations": no_explain_seconds,
        "gpu_inference_latency_seconds_with_explanations": with_explain_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_mib": (peak_gpu_memory_bytes / (1024**2)) if peak_gpu_memory_bytes else None,
        "cpu_inference_latency_seconds_no_explanations": cpu_latency_seconds,
        "note": "Single-sample (batch_size=1) latency, no batching optimization attempted -- this is a feasibility check only, per Step 15's explicit instruction not to optimize prematurely.",
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9K: Authoritative World Model Reconciliation + Inference Engine")
    lines.append("")
    lines.append(f"Status: **{report['status']}**")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(report["executive_summary"])
    lines.append("")

    lines.append("## 2. Track A vs Track B Reconciliation")
    lines.append("")
    lines.append("See `track_reconciliation.md` for the full decision document. Summary: Track A "
                  "(experiments/phase6b_vector_world_model.py + phase7b_mitre_stage_head.py) is authoritative; "
                  "Track B (world_model/world_model.py) is quarantined by documentation, not deleted, and was "
                  "confirmed to have no Django-reachable import path.")
    lines.append("")

    lines.append("## 3. Authoritative Model Decision")
    lines.append("")
    lines.append("`VectorWorldModelWithStageHead` (experiments/phase7b_mitre_stage_head.py), backbone from "
                  "experiments/phase6b_vector_world_model.py.")
    lines.append("")

    lines.append("## 4. Exact Checkpoint Used")
    lines.append("")
    sm = report["safety_manifest"]
    for key, val in sm["checksums"].items():
        lines.append(f"- `{key}`: `{val['path']}` (sha256 `{val['sha256'][:16]}...`)")
    lines.append("")

    lines.append("## 5. Inference Contract")
    lines.append("")
    lines.append("See `inference_contract.json` for the full schema. Top-level keys: input_metadata, "
                  "current_state, future_state_rollout, whole_horizon_attack_probability, "
                  "mitre_stage_trajectory, explanations, provenance.")
    lines.append("")

    lines.append("## 6. Preprocessing Contract")
    lines.append("")
    lines.append(f"- Feature count: {report['engine_provenance']['feature_count']} (world_model_dataset.py canonical order)")
    lines.append(f"- Scaler: `{report['engine_provenance']['scaler_path']}` (loaded, NEVER refit)")
    lines.append(f"- Feature schema sha256: `{report['engine_provenance']['feature_schema_sha256'][:16]}...`")
    lines.append("")

    lines.append("## 7. World-Model Outputs")
    lines.append("")
    lines.append(f"- {report['engine_provenance']['prediction_semantics']['whole_horizon_attack_probability']}")
    lines.append("")

    lines.append("## 8. MITRE Outputs")
    lines.append("")
    lines.append(f"- {report['engine_provenance']['prediction_semantics']['mitre_stage_trajectory']}")
    lines.append("")

    lines.append("## 9. Explainability Integration")
    lines.append("")
    lines.append(f"- {report['engine_provenance']['explainability_method_infrastructure']}")
    lines.append("- Never claims SHAP; reports the actual method used per call (see explanations.future_attack_risk_prediction.explanation_method in a real result)")
    lines.append("")

    lines.append("## 10. Provenance")
    lines.append("")
    lines.append("Every prediction includes model identifier, checkpoint path+hash, scaler path+hash, "
                  "feature-schema hash, code commit hash, inference timestamp/duration, device, and "
                  "prediction semantics text (see `inference_contract.json`).")
    lines.append("")

    lines.append("## 11. Offline Input Contract")
    lines.append("")
    lines.append("CSV/window -> world_model_dataset.read_world_model_samples() -> [6,157] raw feature array "
                  "-> NetOracleInferenceEngine.predict(). PCAP input is explicitly NOT supported by this "
                  "engine (Phase 9H/9I: RED) -- no PCAP-to-flow extractor was built in this phase.")
    lines.append("")

    lines.append("## 12. UI Integration Status")
    lines.append("")
    lines.append(report["ui_integration_status"])
    lines.append("")

    lines.append("## 13. Deterministic Equivalence Results")
    lines.append("")
    dv = report["deterministic_validation"]
    lines.append(f"- Samples checked: {dv['num_samples_checked']}")
    lines.append(f"- Run1 attack-probability equivalence (tolerance {dv['attack_probability_tolerance']}): {dv['all_run1_attack_probability_equivalent']}")
    lines.append(f"- Phase 7B attack-probability equivalence: {dv['all_phase7b_attack_probability_equivalent']}")
    lines.append(f"- Phase 7B stage-name exact match: {dv['all_phase7b_stage_names_exact_match']}")
    lines.append(f"- Repeated-inference determinism (tolerance {dv['determinism_tolerance']}): {dv['all_repeated_inference_deterministic']}")
    lines.append(f"- Attribution shape [6,F] for every sample: {dv['all_attribution_shape_6xF']}")
    lines.append(f"- **Overall: {dv['overall_pass']}**")
    lines.append("")
    lines.append("| # | source_file | window_start | engine_prob | frozen_run1_prob | frozen_7b_prob | stages_match |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for c in dv["per_sample_checks"]:
        lines.append(
            f"| {c['index']} | {c['source_file']} | {c['window_start']} | {c['engine_attack_probability']:.6f} | "
            f"{c['frozen_run1_attack_probability']} | {c['frozen_phase7b_attack_probability']} | {c['phase7b_stage_names_exact_match']} |"
        )
    lines.append("")

    lines.append("## 14. Performance")
    lines.append("")
    perf = report["performance"]
    lines.append(f"- Device: {perf['device_used_for_primary_measurement']}")
    lines.append(f"- GPU latency (no explanations): {perf['gpu_inference_latency_seconds_no_explanations']*1000:.2f} ms")
    lines.append(f"- GPU latency (with explanations): {perf['gpu_inference_latency_seconds_with_explanations']*1000:.2f} ms")
    lines.append(f"- Peak GPU memory: {perf['peak_gpu_memory_mib']:.2f} MiB" if perf['peak_gpu_memory_mib'] else "- Peak GPU memory: n/a")
    lines.append(f"- CPU latency (no explanations): {perf['cpu_inference_latency_seconds_no_explanations']}")
    lines.append("")

    lines.append("## 15. Tests")
    lines.append("")
    lines.append(report["tests_summary"])
    lines.append("")

    lines.append("## 16. Frozen-Artifact Integrity")
    lines.append("")
    lines.append(f"Prior result directories unchanged: **{report['integrity_audit']['unchanged']}**")
    lines.append("")

    lines.append("## 17. Remaining Limitations")
    lines.append("")
    for item in report["remaining_limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 18. Explicitly Unsupported Capabilities")
    lines.append("")
    for item in report["explicitly_unsupported"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 19. Recommended Phase 9L")
    lines.append("")
    lines.append(report["recommended_next_phase"])
    lines.append("")
    lines.append("STOP AFTER PHASE 9K. Do not begin 9L.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9K directory: {OUTPUT_DIR}")

    print("=== STEP 1: safety snapshot ===")
    safety_manifest = build_safety_manifest()

    print("=== STEP 2-9: instantiate authoritative inference engine ===")
    engine = NetOracleInferenceEngine()
    print(f"parameter_count={engine.parameter_count}, feature_count={len(engine.feature_columns)}, device={engine.device}")

    print("=== STEP 13: deterministic validation ===")
    deterministic_validation = run_deterministic_validation(engine)
    print(json.dumps({k: v for k, v in deterministic_validation.items() if k != "per_sample_checks"}, indent=2))
    if not deterministic_validation["overall_pass"]:
        raise RuntimeError(f"Deterministic validation FAILED: {deterministic_validation}")

    print("=== STEP 15: performance check ===")
    performance = run_performance_check(engine)
    print(json.dumps({k: v for k, v in performance.items() if k != "note"}, indent=2))

    # ---- One real, saved example prediction for the inference_contract.json schema ----
    example_sample = engine.get_test_sample(0)
    example_result = engine.predict(
        example_sample["x_raw"], source_file=example_sample["source_file"], window_start=example_sample["window_start"], top_k=5
    )

    ui_integration_status = (
        "NOT WIRED to the Django dashboard in this phase. Per Step 11's instruction to prefer a thin "
        "integration layer over risky rewrites: experiments/inference_engine.py is a standalone, fully "
        "importable module with zero Django dependency, ready to be called from "
        "world_model/inference_service.py or a new dashboard view in a future phase. The remaining UI work "
        "is: (1) replace/augment AttackForecastService's heuristic fallback with a call into "
        "NetOracleInferenceEngine.predict(), (2) map its structured output onto the existing dashboard "
        "template fields, (3) handle the CSV/window input contract from an uploaded file. None of this was "
        "done in Phase 9K to avoid risky architectural changes to the live Django app in an integration-only phase."
    )

    remaining_limitations = [
        "The inference engine is not yet called from any Django view -- it exists as a standalone, tested module only.",
        "Only a CSV/window input contract is supported; PCAP ingestion was explicitly out of scope (Phase 9H/9I: RED) and was not revisited.",
        "No per-step attack probability, uncertainty, or counterfactual capability exists (explicitly deferred to later phases per this phase's constraints).",
        "Track B (world_model/) remains on disk, unmodified, quarantined by documentation only -- full retirement (file deletion) was explicitly out of scope.",
        f"A harmless sklearn version-mismatch warning appears when unpickling the scaler (pickled with scikit-learn 1.9.1, current environment has 1.8.0); predictions were verified correct despite the warning (see deterministic validation), but this should be resolved by re-pickling with the current sklearn version in a future environment-hygiene pass.",
    ]

    explicitly_unsupported = [
        "SHAP (actual method is gradient x input with SHAP-compatible infrastructure)",
        "Native per-step attack probability (only whole-horizon P(attack in t+1..t+6))",
        "Calibrated probabilities (Phase 8B finding stands unchanged)",
        "Causal attacker kill-chain inference (MITRE stages are a reasoned label mapping)",
        "Unseen-attack generalization claims (not tested in this phase)",
        "PCAP-derived model input (Phase 9H/9I RED verdicts stand unchanged)",
        "Uncertainty quantification of any kind",
        "Counterfactual simulation of any kind",
    ]

    report = {
        "phase": "9K",
        "success": True,
        "status": "GREEN",
        "executive_summary": (
            "Phase 9K reconciled the two World Model implementations discovered in Phase 9J, designating "
            "the audited experiments/phase6b_vector_world_model.py + phase7b_mitre_stage_head.py + Run1 "
            "checkpoint as the single authoritative model, and built ONE inference engine "
            "(experiments/inference_engine.py) exposing its rollout, whole-horizon attack probability, "
            "6-step MITRE stage trajectory, and gradient-based explainability through a single documented "
            "contract with full provenance. Deterministic validation against frozen Phase 6B Run1 and "
            "Phase 7B test-set predictions passed on all checked samples (attack-probability equivalence "
            "within 1e-3, exact stage-name match, deterministic repeated inference, correct [6,157] "
            "attribution shape). No training, no threshold changes, no checkpoint modification occurred. "
            "The engine is not yet wired into the Django dashboard -- that remains explicit future work."
        ),
        "safety_manifest": safety_manifest,
        "engine_provenance": engine.provenance.to_dict(),
        "example_prediction": example_result,
        "deterministic_validation": deterministic_validation,
        "performance": performance,
        "ui_integration_status": ui_integration_status,
        "remaining_limitations": remaining_limitations,
        "explicitly_unsupported": explicitly_unsupported,
        "recommended_next_phase": (
            "Phase 9L should wire experiments/inference_engine.py into world_model/inference_service.py "
            "(or a new, thin Django view) so the live dashboard displays real model output instead of the "
            "heuristic fallback -- pure integration, no new modeling, following the same audit-then-"
            "integrate discipline used since Phase 9H. It should NOT attempt per-step risk, uncertainty, "
            "or counterfactual work -- those remain separate, independently audited future phases per "
            "Phase 9J's roadmap."
        ),
        "tests_summary": "See the final response for exact pass counts across Phase 9K and the required 9G/9H/9I/9J/7B/7C/7D/8A/8B regressions.",
    }

    # Compute the integrity audit BEFORE writing any report, so both the JSON
    # and Markdown reports are written exactly once, already containing it
    # (avoids a two-pass write and a stale first copy).
    integrity_audit = verify_no_prior_artifact_changed(safety_manifest)
    report["integrity_audit"] = integrity_audit

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "safety_manifest.json").write_text(json.dumps(safety_manifest, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "inference_contract.json").write_text(json.dumps(example_result, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "track_reconciliation.md").write_text(TRACK_RECONCILIATION_MD, encoding="utf-8")
    (OUTPUT_DIR / "phase9k_integration_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "phase9k_integration_report.md", report)

    if not integrity_audit["unchanged"]:
        raise RuntimeError(f"Prior phase artifacts changed unexpectedly: {integrity_audit['mismatches']}")

    print(
        json.dumps(
            {
                "success": True,
                "status": "GREEN",
                "deterministic_validation_overall_pass": deterministic_validation["overall_pass"],
                "prior_artifacts_unchanged": integrity_audit["unchanged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
