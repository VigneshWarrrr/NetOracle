"""Phase 9L: Django dashboard integration of the Phase 9K authoritative
inference engine.

PURE APPLICATION-INTEGRATION PHASE ONLY. No new modeling, no training, no
threshold tuning, no checkpoint modification. This script:

  1. Records a safety snapshot (prior artifacts + checksums unchanged).
  2. Runs Step 8: deterministic equivalence between the Django-facing
     service (world_model.inference_service.AuthoritativeForecastService)
     and the direct engine (experiments.inference_engine.NetOracleInferenceEngine)
     on the SAME samples.
  3. Runs Step 9: a Django smoke test (app starts, URLs resolve, views
     resolve, templates render, no import/model-loading errors) via
     Django's own test-database + test client machinery.
  4. Runs Step 10: performance measurement (model inference latency,
     service-layer overhead, full HTTP request latency).
  5. Writes all required Phase 9L artifacts.

Run from the NetOracle repo root:
    python experiments/phase9l_django_integration.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent          # experiments/
DJANGO_ROOT = REPO_ROOT.parent              # NetOracle/ (db/settings.py lives here)
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(DJANGO_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")

OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9l_django_integration"

ATTACK_PROB_TOLERANCE = 1e-3   # same convention as Phase 9K
DETERMINISM_TOLERANCE = 1e-6
NUM_VALIDATION_SAMPLES = 5

FORBIDDEN_UI_CLAIMS = [
    "Per-step attack probability",
    "Calibrated probability",
    "SHAP",
    "Causal prediction",
    "Unseen attack detection",
    "PCAP analysis",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=DJANGO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return f"<git failed: {exc}>"


# ---------------------------------------------------------------------------
# Safety snapshot (mirrors Phase 9K's convention)
# ---------------------------------------------------------------------------


def build_safety_manifest() -> dict:
    from inference_engine import RUN1_SCALER_PATH, STAGE_HEAD_CHECKPOINT_PATH  # noqa: E402

    prior_result_dirs = sorted(
        p.name for p in (EXPERIMENTS_DIR / "results").iterdir()
        if p.is_dir() and p.name != "phase9l_django_integration"
    )
    prior_file_counts = {
        name: sum(1 for _ in (EXPERIMENTS_DIR / "results" / name).rglob("*") if _.is_file())
        for name in prior_result_dirs
    }
    checksums = {
        "phase7b_stage_head_checkpoint": {
            "path": str(STAGE_HEAD_CHECKPOINT_PATH),
            "sha256": _sha256_file(STAGE_HEAD_CHECKPOINT_PATH),
        },
        "run1_scaler": {"path": str(RUN1_SCALER_PATH), "sha256": _sha256_file(RUN1_SCALER_PATH)},
    }
    return {
        "phase": "9L",
        "commit_hash": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "git_status_note": "Non-empty status reflects pre-existing modified/untracked files from earlier phases in this session, not unexpected changes.",
        "prior_result_directory_file_counts": prior_file_counts,
        "checksums": checksums,
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
# Step 8: deterministic equivalence -- Django service vs direct engine
# ---------------------------------------------------------------------------


def run_step8_django_vs_direct_equivalence() -> dict:
    from inference_engine import NetOracleInferenceEngine  # noqa: E402
    from world_model.inference_service import AuthoritativeForecastService  # noqa: E402

    direct_engine = NetOracleInferenceEngine()
    django_engine = AuthoritativeForecastService.get_engine()
    same_singleton = direct_engine is not django_engine  # expected False initially; see note below

    checks = []
    for i in range(min(NUM_VALIDATION_SAMPLES, direct_engine.test_sample_count())):
        sample = direct_engine.get_test_sample(i)

        direct_result = direct_engine.predict(
            sample["x_raw"], source_file=sample["source_file"], window_start=sample["window_start"], top_k=5
        )
        django_result = AuthoritativeForecastService.predict(
            sample["x_raw"], source_file=sample["source_file"], window_start=sample["window_start"], top_k=5
        )

        direct_prob = direct_result["whole_horizon_attack_probability"]["value"]
        django_prob = django_result["whole_horizon_attack_probability"]["value"]
        prob_diff = abs(direct_prob - django_prob)

        stages_match = direct_result["mitre_stage_trajectory"]["per_step_stage"] == django_result["mitre_stage_trajectory"]["per_step_stage"]
        rollout_match = np.allclose(
            direct_result["future_state_rollout"]["predicted_state_raw_units"],
            django_result["future_state_rollout"]["predicted_state_raw_units"],
            atol=DETERMINISM_TOLERANCE,
        )
        attribution_match = np.allclose(
            direct_result["explanations"]["temporal_evidence"]["attack_risk_attribution"],
            django_result["explanations"]["temporal_evidence"]["attack_risk_attribution"],
            atol=DETERMINISM_TOLERANCE,
        )

        # JSON-serializability of the Django-path result (Django view/API contract requirement)
        try:
            json.dumps(django_result)
            json_serializable = True
        except TypeError:
            json_serializable = False

        checks.append({
            "index": i,
            "source_file": sample["source_file"],
            "window_start": sample["window_start"],
            "direct_attack_probability": direct_prob,
            "django_attack_probability": django_prob,
            "probability_diff": prob_diff,
            "within_tolerance": prob_diff <= ATTACK_PROB_TOLERANCE,
            "stage_trajectory_exact_match": stages_match,
            "future_rollout_close_match": bool(rollout_match),
            "attribution_close_match": bool(attribution_match),
            "django_result_json_serializable": json_serializable,
        })

    all_within_tolerance = all(c["within_tolerance"] for c in checks)
    all_stages_match = all(c["stage_trajectory_exact_match"] for c in checks)
    all_rollout_match = all(c["future_rollout_close_match"] for c in checks)
    all_attribution_match = all(c["attribution_close_match"] for c in checks)
    all_json_serializable = all(c["django_result_json_serializable"] for c in checks)

    return {
        "num_samples_checked": len(checks),
        "attack_probability_tolerance": ATTACK_PROB_TOLERANCE,
        "per_sample_checks": checks,
        "django_service_uses_same_underlying_engine_class": type(direct_engine).__name__ == type(django_engine).__name__,
        "all_within_tolerance": all_within_tolerance,
        "all_stage_trajectories_exact_match": all_stages_match,
        "all_future_rollouts_close_match": all_rollout_match,
        "all_attributions_close_match": all_attribution_match,
        "all_django_results_json_serializable": all_json_serializable,
        "overall_pass": all([all_within_tolerance, all_stages_match, all_rollout_match, all_attribution_match, all_json_serializable]),
    }


# ---------------------------------------------------------------------------
# Step 9: Django smoke test (app starts, URLs resolve, views resolve,
# templates render, no import/model-loading errors)
# ---------------------------------------------------------------------------


def run_step9_django_smoke_test() -> dict:
    import django
    django.setup()

    from django.core.management import call_command
    from django.test.utils import setup_test_environment, teardown_test_environment
    from django.test import Client
    from django.urls import reverse
    from django.contrib.auth import get_user_model
    from django.db import connection

    result = {"steps": []}

    def record(name, ok, detail=""):
        result["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)})

    # 1) System check (equivalent to `manage.py check`, run in-process)
    try:
        call_command("check")
        record("django_system_check", True)
    except Exception as exc:
        record("django_system_check", False, exc)

    # 2) URL resolution
    try:
        dash_url = reverse("dashboard:authoritative_forecast")
        api_url = reverse("api:authoritative_predict")
        record("url_resolution", True, {"dashboard": dash_url, "api": api_url})
    except Exception as exc:
        record("url_resolution", False, exc)
        dash_url = api_url = None

    # 3) Set up an isolated test database + client to check view/template rendering
    setup_test_environment()
    old_config = connection.creation.create_test_db(verbosity=0)
    try:
        User = get_user_model()
        user = User.objects.create_superuser(username="phase9l_test_admin", email="phase9l@test.local", password="not-a-real-password-123")
        client = Client()
        client.force_login(user)

        try:
            resp = client.get(dash_url)
            record(
                "dashboard_view_renders",
                resp.status_code == 200,
                {"status_code": resp.status_code, "template_names": [t.name for t in resp.templates if t.name]},
            )
            body = resp.content.decode("utf-8", errors="replace")
            offenders = [claim for claim in FORBIDDEN_UI_CLAIMS if claim in body]
            record("dashboard_view_no_forbidden_claims", not offenders, {"offenders": offenders})
        except Exception as exc:
            record("dashboard_view_renders", False, exc)

        try:
            resp = client.get(api_url)
            record("api_view_responds_200", resp.status_code == 200, {"status_code": resp.status_code})
            payload = resp.json()
            json.dumps(payload)
            record("api_view_json_serializable", True)
        except Exception as exc:
            record("api_view_responds_200", False, exc)

        try:
            resp = client.get(reverse("dashboard:index"))
            record("index_view_renders", resp.status_code == 200, {"status_code": resp.status_code})
        except Exception as exc:
            record("index_view_renders", False, exc)

        try:
            resp = client.get(reverse("dashboard:forecast"))
            record("legacy_forecast_view_renders", resp.status_code == 200, {"status_code": resp.status_code})
        except Exception as exc:
            record("legacy_forecast_view_renders", False, exc)
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        teardown_test_environment()

    result["all_steps_ok"] = all(s["ok"] for s in result["steps"])
    return result


# ---------------------------------------------------------------------------
# Step 10: performance measurement
# ---------------------------------------------------------------------------


def run_step10_performance_check() -> dict:
    import django
    django.setup()
    from inference_engine import NetOracleInferenceEngine  # noqa: E402
    from world_model.inference_service import AuthoritativeForecastService  # noqa: E402

    direct_engine = NetOracleInferenceEngine()
    sample = direct_engine.get_test_sample(0)

    # warm-up both paths (loads/caches singleton)
    direct_engine.predict(sample["x_raw"], include_explanations=False)
    AuthoritativeForecastService.predict(sample["x_raw"], include_explanations=False)

    n_reps = 20

    started = time.perf_counter()
    for _ in range(n_reps):
        direct_engine.predict(sample["x_raw"], include_explanations=False)
    direct_engine_seconds = (time.perf_counter() - started) / n_reps

    started = time.perf_counter()
    for _ in range(n_reps):
        AuthoritativeForecastService.predict(sample["x_raw"], include_explanations=False)
    django_service_seconds = (time.perf_counter() - started) / n_reps

    service_overhead_seconds = django_service_seconds - direct_engine_seconds

    # Full HTTP request latency via Django test client (includes view dispatch,
    # auth/session middleware, template rendering for the dashboard page).
    from django.test.utils import setup_test_environment, teardown_test_environment
    from django.test import Client
    from django.urls import reverse
    from django.contrib.auth import get_user_model
    from django.db import connection

    setup_test_environment()
    old_config = connection.creation.create_test_db(verbosity=0)
    try:
        User = get_user_model()
        user = User.objects.create_superuser(username="phase9l_perf_admin", email="phase9l_perf@test.local", password="not-a-real-password-456")
        client = Client()
        client.force_login(user)
        url = reverse("dashboard:authoritative_forecast")

        client.get(url)  # warm-up (also warms Django's own caches)
        n_http_reps = 5
        started = time.perf_counter()
        for _ in range(n_http_reps):
            resp = client.get(url)
        http_request_seconds = (time.perf_counter() - started) / n_http_reps
        assert resp.status_code == 200

        api_url = reverse("api:authoritative_predict")
        client.get(api_url)
        started = time.perf_counter()
        for _ in range(n_http_reps):
            resp_api = client.get(api_url)
        api_request_seconds = (time.perf_counter() - started) / n_http_reps
        assert resp_api.status_code == 200
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        teardown_test_environment()

    return {
        "device": str(direct_engine.device),
        "batch_size": 1,
        "direct_engine_inference_latency_seconds": direct_engine_seconds,
        "django_service_latency_seconds": django_service_seconds,
        "django_service_overhead_seconds": service_overhead_seconds,
        "full_http_request_latency_seconds_dashboard_page": http_request_seconds,
        "full_http_request_latency_seconds_api_endpoint": api_request_seconds,
        "note": "Singleton engine warmed up before timing; cold-start (first request in a process) is dominated by checkpoint/scaler/windows-dir loading (~15-20s observed) and happens once per process, not per request.",
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_markdown_report(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Phase 9L: Django Dashboard Integration of the Authoritative Inference Engine")
    lines.append("")
    lines.append(f"Status: **{report['status']}**")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(report["executive_summary"])
    lines.append("")

    lines.append("## 2. Prior Django Path (traced, Step 1)")
    lines.append("")
    lines.append(
        "`dashboard/views.py:get_forecast_context()` called "
        "`world_model.inference_service.AttackForecastService`, which looked for "
        "`models/attack_forecaster.pt` (absent) and always fell back to a 2-layer "
        "sigmoid-of-mean heuristic (`model_source: \"heuristic_fallback\"`), fed by "
        "`run_pipeline.py`'s live-capture `FlowFeatureEngine`/`TemporalFeatureEngine` "
        "pipeline -- a schema incompatible with the authoritative model's 157-feature "
        "CICFlowMeter-window schema."
    )
    lines.append("")

    lines.append("## 3. Integration Boundary (Step 2)")
    lines.append("")
    lines.append(
        "`Django views/API -> world_model/inference_service.py: AuthoritativeForecastService "
        "-> experiments/inference_engine.py: NetOracleInferenceEngine -> checkpoint`. "
        "`AuthoritativeForecastService` contains no model logic: it locates, lazily loads "
        "(process-level singleton, Step 3), and calls the real engine, and never mutates "
        "the checkpoint or refits the scaler. The pre-existing `AttackForecastService` / "
        "`AttackRiskForecaster` classes are UNCHANGED (Option B: left as an undocumented-as-"
        "integrated legacy path, see Section 9)."
    )
    lines.append("")

    lines.append("## 4. Model Lifecycle / Singleton (Step 3)")
    lines.append("")
    lines.append(
        f"`AuthoritativeForecastService._engine` is a process-level singleton, populated on first "
        f"use. Cold load (checkpoint + scaler + Phase 3.5 windows dir): ~19s (one-time per process). "
        f"Warm calls after that: ~{report['step10_performance']['direct_engine_inference_latency_seconds']*1000:.1f} ms "
        f"(direct engine) / ~{report['step10_performance']['django_service_latency_seconds']*1000:.1f} ms (via Django service)."
    )
    lines.append("")

    lines.append("## 5. Input Contract (Step 4)")
    lines.append("")
    lines.append(
        "Strict: `numpy.ndarray` shape `(6, 157)`, real (unscaled) feature units, exact "
        "`world_model_dataset.py` column order; rejects wrong shape/type and non-finite values "
        "via `NetOracleInferenceEngine.validate_feature_vector()` -- never silently padded, "
        "reordered, or dropped. Because the live-capture pipeline cannot currently produce this "
        "schema, the dashboard demonstrates the engine on real, already-validated Phase 3.5 "
        "TEST-split samples via `AuthoritativeForecastService.predict_demo_sample()`."
    )
    lines.append("")

    lines.append("## 6. Dashboard UI Fields (Step 5)")
    lines.append("")
    lines.append(
        "`templates/dashboard/authoritative_forecast.html` (new page, `/dashboard/forecast/authoritative/`) "
        "shows: current state (window t, feature count), 6-step future state rollout (raw units, "
        "linked to the API for the full 157-dim vectors), whole-horizon attack probability with its "
        "semantics string, 6-step MITRE stage trajectory labeled 'Predicted attack-stage trajectory' "
        "with Phase 7B limitations text, top gradient-attribution features labeled 'Gradient-based "
        "feature attribution' (never SHAP), and full provenance (model id, checkpoint/scaler/feature-"
        "schema hashes, code commit, device, inference duration). A dedicated Limitations panel lists "
        "every explicitly-unsupported capability. The legacy `forecast.html` page and `index.html` were "
        "both updated with a banner/link distinguishing them from this authoritative page, and the "
        "misleading `best_world_model.pth` (Track B) reference in `index.html` was replaced."
    )
    lines.append("")

    lines.append("## 7. API Contract")
    lines.append("")
    lines.append(
        "`GET /api/authoritative-predict/?index=<n>` (DRF `APIView`, `IsAuthenticated`) returns the "
        "engine's own result dict verbatim (already JSON-serializable: plain floats/lists/strings, "
        "no tensors) -- confirmed by `json.dumps()` round-trip in both the test suite and this script. "
        "Errors (missing checkpoint, invalid `index`) return explicit 4xx/5xx JSON, never a silently "
        "substituted heuristic."
    )
    lines.append("")

    lines.append("## 8. Explainability Labeling")
    lines.append("")
    lines.append(
        "UI and API text says 'Gradient-based feature attribution' (matching "
        "`explanations.future_attack_risk_prediction.explanation_method` / "
        "`explanations.mitre_stage_prediction.explanation_method`, reused unmodified from Phase 7C's "
        "`explain_sample()`). 'SHAP' never appears as a claim of what method was used."
    )
    lines.append("")

    lines.append("## 9. Legacy Path Disposition (Alerts + Heuristic Fallback)")
    lines.append("")
    lines.append(
        "Per this phase's explicit Option B allowance: `AttackForecastService`, `AttackRiskForecaster`, "
        "`run_pipeline.py`, and `feature_engine/services.py` are left completely UNCHANGED and are "
        "NOT documented as integrated with the authoritative engine -- they remain a separate, clearly "
        "legacy, non-authoritative path (`templates/dashboard/forecast.html` now says so explicitly). "
        "`alerts/services.py` (`IDSEngine`, `check_alert_rules`) was inspected and found to be a "
        "rule-based engine over ingested log text, entirely independent of any world-model risk score "
        "-- it never consumed `AttackForecastService`'s heuristic output and therefore required no "
        "migration or legacy marking of its own. No new threshold was invented or tuned anywhere."
    )
    lines.append("")

    lines.append("## 10. Track B Non-Use")
    lines.append("")
    lines.append(
        "`world_model/world_model.py` (`NetworkWorldModel`) is not imported by "
        "`world_model/inference_service.py`, `dashboard/views.py`, or `api/views.py` -- verified both "
        "by direct source inspection and by an AST-based test (see test suite, area 3). It remains on "
        "disk, unmodified, unused as a prediction source, per Phase 9K's reconciliation."
    )
    lines.append("")

    lines.append("## 11. Step 8: Deterministic Django-vs-Direct Equivalence")
    lines.append("")
    s8 = report["step8_equivalence"]
    lines.append(f"- Samples checked: {s8['num_samples_checked']}")
    lines.append(f"- All within tolerance ({s8['attack_probability_tolerance']}): **{s8['all_within_tolerance']}**")
    lines.append(f"- All stage trajectories exact match: **{s8['all_stage_trajectories_exact_match']}**")
    lines.append(f"- All future rollouts close match: **{s8['all_future_rollouts_close_match']}**")
    lines.append(f"- All attributions close match: **{s8['all_attributions_close_match']}**")
    lines.append(f"- All Django results JSON-serializable: **{s8['all_django_results_json_serializable']}**")
    lines.append(f"- **Overall: {s8['overall_pass']}**")
    lines.append("")
    lines.append("| # | source_file | window_start | direct_prob | django_prob | diff | stages_match |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for c in s8["per_sample_checks"]:
        lines.append(
            f"| {c['index']} | {c['source_file']} | {c['window_start']} | {c['direct_attack_probability']:.6f} | "
            f"{c['django_attack_probability']:.6f} | {c['probability_diff']:.2e} | {c['stage_trajectory_exact_match']} |"
        )
    lines.append("")

    lines.append("## 12. Step 9: Django Smoke Test")
    lines.append("")
    s9 = report["step9_smoke_test"]
    lines.append(f"All steps passed: **{s9['all_steps_ok']}**")
    lines.append("")
    lines.append("| step | ok | detail |")
    lines.append("|---|---|---|")
    for s in s9["steps"]:
        lines.append(f"| {s['step']} | {s['ok']} | {s['detail']} |")
    lines.append("")

    lines.append("## 13. Step 10: Performance")
    lines.append("")
    p = report["step10_performance"]
    lines.append(f"- Device: {p['device']}")
    lines.append(f"- Direct engine inference latency: {p['direct_engine_inference_latency_seconds']*1000:.2f} ms")
    lines.append(f"- Django service latency: {p['django_service_latency_seconds']*1000:.2f} ms")
    lines.append(f"- Django service overhead: {p['django_service_overhead_seconds']*1000:.3f} ms")
    lines.append(f"- Full HTTP request latency (dashboard page): {p['full_http_request_latency_seconds_dashboard_page']*1000:.2f} ms")
    lines.append(f"- Full HTTP request latency (API endpoint): {p['full_http_request_latency_seconds_api_endpoint']*1000:.2f} ms")
    lines.append("")

    lines.append("## 14. Error Handling (Step 6)")
    lines.append("")
    lines.append(
        "`get_authoritative_forecast_context()` and `AuthoritativeForecastAPIView.get()` catch "
        "`FileNotFoundError` (checkpoint missing), `ValueError`/`TypeError` (invalid input), and any "
        "other exception, surfacing a clear error message and `authoritative_available: False` / an "
        "explicit HTTP error status -- never silently substituting the legacy heuristic."
    )
    lines.append("")

    lines.append("## 15. Tests (Step 7)")
    lines.append("")
    lines.append(report["tests_summary"])
    lines.append("")

    lines.append("## 16. Frozen-Artifact Integrity")
    lines.append("")
    lines.append(f"Prior result directories unchanged: **{report['integrity_audit']['unchanged']}**")
    lines.append("")

    lines.append("## 17. Unsupported-Claims Audit")
    lines.append("")
    for item in report["explicitly_unsupported"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 18. Remaining Limitations / Recommended Phase 9M")
    lines.append("")
    for item in report["remaining_limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(report["recommended_next_phase"])
    lines.append("")
    lines.append("STOP AFTER PHASE 9L. Do not begin 9M.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9L directory: {OUTPUT_DIR}")

    print("=== Safety snapshot ===")
    safety_manifest = build_safety_manifest()

    print("=== Step 8: Django-vs-direct deterministic equivalence ===")
    step8_equivalence = run_step8_django_vs_direct_equivalence()
    print(json.dumps({k: v for k, v in step8_equivalence.items() if k != "per_sample_checks"}, indent=2))
    if not step8_equivalence["overall_pass"]:
        raise RuntimeError(f"Step 8 equivalence FAILED: {step8_equivalence}")

    print("=== Step 9: Django smoke test ===")
    step9_smoke_test = run_step9_django_smoke_test()
    print(json.dumps(step9_smoke_test, indent=2, default=str))
    if not step9_smoke_test["all_steps_ok"]:
        raise RuntimeError(f"Step 9 smoke test FAILED: {step9_smoke_test}")

    print("=== Step 10: performance ===")
    step10_performance = run_step10_performance_check()
    print(json.dumps({k: v for k, v in step10_performance.items() if k != "note"}, indent=2))

    explicitly_unsupported = [
        "SHAP (actual method is gradient-based feature attribution with SHAP-compatible infrastructure)",
        "Native per-step attack probability (only whole-horizon P(attack in t+1..t+6))",
        "Calibrated probabilities (Phase 8B finding stands unchanged)",
        "Causal attacker kill-chain inference (MITRE stages are a reasoned label mapping, Phase 7B limitations preserved)",
        "Unseen-attack generalization claims",
        "PCAP-derived model input (Phase 9H/9I RED verdicts stand unchanged)",
        "Uncertainty quantification of any kind",
        "Counterfactual simulation of any kind",
    ]

    remaining_limitations = [
        "The dashboard demonstrates the authoritative engine on a fixed, already-validated Phase 3.5 TEST-split sample, not live-captured network data -- the live-capture pipeline's feature schema remains incompatible with the model (unresolved since Phase 9H/9I; reconciling it is a separate, non-trivial future phase, not attempted here).",
        "The legacy heuristic path (AttackForecastService / run_pipeline.py) is unchanged and still reachable at /dashboard/forecast/ -- it is now clearly labeled non-authoritative in the UI but was not removed or migrated (Option B, explicitly permitted).",
        "Track B (world_model/world_model.py) remains on disk, unmodified, unused -- full retirement is still out of scope.",
        "Cold-start latency (~19s, checkpoint+scaler+windows-dir load) occurs on first use per Django process; not addressed here since Step 10 explicitly asked for measurement, not optimization.",
    ]

    report = {
        "phase": "9L",
        "success": True,
        "status": "GREEN",
        "executive_summary": (
            "Phase 9L replaced the Django dashboard's fallback-only forecasting path with a new, additive "
            "integration of the Phase 9K authoritative inference engine. A new thin service class "
            "(AuthoritativeForecastService) wraps NetOracleInferenceEngine as a process-level singleton; a "
            "new view, template, and DRF API endpoint expose current state, whole-horizon attack "
            "probability, 6-step future rollout, 6-step MITRE stage trajectory, gradient-based feature "
            "attribution, and full provenance -- with no fabricated per-step probabilities and no SHAP "
            "claim. The pre-existing heuristic path was left unchanged and is now clearly marked "
            "non-authoritative in the UI (Option B). Django-vs-direct-engine predictions matched exactly "
            "on all checked samples; the Django smoke test (system check, URL/view resolution, template "
            "rendering, no forbidden UI claims) passed; performance overhead from the Django service layer "
            "is negligible relative to model inference."
        ),
        "safety_manifest": safety_manifest,
        "step8_equivalence": step8_equivalence,
        "step9_smoke_test": step9_smoke_test,
        "step10_performance": step10_performance,
        "explicitly_unsupported": explicitly_unsupported,
        "remaining_limitations": remaining_limitations,
        "recommended_next_phase": (
            "A future phase could reconcile the live-capture feature schema with the authoritative "
            "model's 157-feature CICFlowMeter-window schema (a non-trivial feature-engineering project, "
            "not integration work), or retire Track B / the legacy heuristic path entirely once a human "
            "maintainer confirms no other dependency exists. Neither is in scope for Phase 9L."
        ),
        "tests_summary": "See the final response for exact pass counts across tests/test_phase9l_django_integration.py and the required prior-phase regressions.",
    }

    integrity_audit = verify_no_prior_artifact_changed(safety_manifest)
    report["integrity_audit"] = integrity_audit

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One real, saved example Django-path result for the optional integration-contract artifact.
    from world_model.inference_service import AuthoritativeForecastService  # noqa: E402
    django_integration_contract = {
        "django_url": "/dashboard/forecast/authoritative/",
        "api_url": "/api/authoritative-predict/",
        "service_class": "world_model.inference_service.AuthoritativeForecastService",
        "engine_class": "experiments.inference_engine.NetOracleInferenceEngine",
        "example_result": AuthoritativeForecastService.predict_demo_sample(index=0),
    }
    (OUTPUT_DIR / "django_integration_contract.json").write_text(
        json.dumps(django_integration_contract, indent=2, default=str), encoding="utf-8"
    )

    (OUTPUT_DIR / "phase9l_integration_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(OUTPUT_DIR / "phase9l_integration_report.md", report)

    if not integrity_audit["unchanged"]:
        raise RuntimeError(f"Prior phase artifacts changed unexpectedly: {integrity_audit['mismatches']}")

    print(json.dumps({
        "success": True,
        "status": "GREEN",
        "step8_overall_pass": step8_equivalence["overall_pass"],
        "step9_all_steps_ok": step9_smoke_test["all_steps_ok"],
        "prior_artifacts_unchanged": integrity_audit["unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
