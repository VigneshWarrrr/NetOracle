"""Phase 10B: regression tests for the six submission-blocking fixes (O1, O2, I3, C2, C1, I2)
and for the guarantee that the authoritative AI core was not changed.

Covers: A Django start/import, B engine load, C predict() smoke, D (6,157) input contract,
E explanation output, F attribution rendering (small non-zero values stay visible),
G rendered dashboard HTML, H six-window wording, I whole-horizon scalar wording,
J MITRE trajectory rendering, K legacy path stays non-authoritative, L Phase 9N absent from
the authoritative path, plus README / deck / manifest / Git-provenance checks.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")

import django  # noqa: E402

django.setup()

import dashboard.views as dashboard_views  # noqa: E402
import api.views as api_views  # noqa: E402
import world_model.inference_service as service_module  # noqa: E402
from world_model.inference_service import AuthoritativeForecastService, AttackForecastService  # noqa: E402
import inference_engine  # noqa: E402
from inference_engine import NetOracleInferenceEngine  # noqa: E402

OUT = EXPERIMENTS_DIR / "results/phase10b_submission_remediation"
PY = sys.executable
CHECKPOINT_SHA = "f9d16f1943aeeed4f355659b85e90cb6cf8cb94ae84d2fcbd473c59fa3342fe4"
SCALER_SHA = "b3a0cea7f9d2d0fc59de163c8768116cbd95abbe3b30c127c44a2322f0c24881"
RUN1_SHA = "6374a9c47d722215e64a3cae1b1e24c425a89dad26407b147b153af03404fb79"
# Phase 9K contract / Phase 10A reference for replay window 0
REFERENCE_SAMPLE0_SCORE = 0.9998519420623779
REFERENCE_SAMPLE0_STAGES = ["COMMAND_AND_CONTROL"] * 6
STALE_LANDING_STRINGS = ["Model offline", "Waiting for trained checkpoint", "Checkpoint unavailable", "five observation windows", "Five-window rollout", "Projected attack progression"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# shared, lazily-built fixtures (the engine load and the test database are expensive)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def engine() -> NetOracleInferenceEngine:
    return AuthoritativeForecastService.get_engine()


def sample0_result() -> dict:
    if "s0" not in _CACHE:
        _CACHE["s0"] = AuthoritativeForecastService.predict_demo_sample(index=0)
    return _CACHE["s0"]


def client():
    """Django test client logged in as a staff user, on a throw-away test database."""
    if "client" not in _CACHE:
        from django.contrib.auth import get_user_model
        from django.db import connection
        from django.test import Client
        from django.test.utils import setup_test_environment, teardown_test_environment

        setup_test_environment()
        config = connection.creation.create_test_db(verbosity=0)
        user = get_user_model().objects.create_superuser(username="p10b_admin", email="p10b@test.local", password="not-a-real-password-10b")
        c = Client()
        c.force_login(user)
        _CACHE["client"] = c

        def cleanup():
            connection.creation.destroy_test_db(config, verbosity=0)
            teardown_test_environment()

        unittest.addModuleCleanup(cleanup)
    return _CACHE["client"]


def page(url: str, **patches) -> str:
    resp = client().get(url)
    assert resp.status_code == 200, (url, resp.status_code)
    return resp.content.decode("utf-8")


def rendered(name: str) -> str:
    if name not in _CACHE:
        _CACHE[name] = page({"landing": "/dashboard/", "auth": "/dashboard/forecast/authoritative/", "legacy": "/dashboard/forecast/"}[name])
    return _CACHE[name]


def visible_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    import html as html_lib
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", html))).strip()


# ---------------------------------------------------------------------------
# O1 / A: Django start-up with the REAL declared dependency
# ---------------------------------------------------------------------------


class DjangoStartupTests(unittest.TestCase):
    def test_reportlab_is_the_real_package_not_a_stub(self) -> None:
        import reportlab
        location = Path(reportlab.__file__).resolve()
        self.assertIn("site-packages", str(location))
        self.assertFalse(str(location).startswith(str(REPO_ROOT)), "reportlab must not live inside the project")
        self.assertTrue(hasattr(reportlab, "__version__"))
        from reportlab.platypus import SimpleDocTemplate, Table
        self.assertTrue(callable(SimpleDocTemplate) and callable(Table))
        stubs = [p for p in REPO_ROOT.rglob("reportlab") if p.is_dir() and "myenv" not in p.parts and "site-packages" not in p.parts]
        self.assertEqual(stubs, [], "a reportlab directory inside the project would be a stub")

    def test_manage_py_check_passes(self) -> None:
        result = subprocess.run([PY, "manage.py", "check"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertIn("no issues", result.stdout)

    def test_urls_resolve_to_authoritative_views(self) -> None:
        from django.urls import resolve, reverse
        self.assertIs(resolve(reverse("dashboard:authoritative_forecast")).func.view_class, dashboard_views.AuthoritativeForecastView)
        self.assertIs(resolve(reverse("api:authoritative_predict")).func.view_class, api_views.AuthoritativeForecastAPIView)

    def test_requirements_declares_reportlab_and_no_longer_requires_shap(self) -> None:
        lines = [l.strip() for l in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(l.lower().startswith("reportlab") for l in lines))
        self.assertFalse(any(re.match(r"^shap\b", l, re.I) for l in lines), "shap is not required; the deployed method is Gradient x Input")


# ---------------------------------------------------------------------------
# B / C / D / E: authoritative engine (behaviour must be unchanged)
# ---------------------------------------------------------------------------


class AuthoritativeEngineTests(unittest.TestCase):
    def test_engine_loads_the_frozen_artifacts(self) -> None:
        e = engine()
        self.assertIsInstance(e, NetOracleInferenceEngine)
        self.assertEqual(e.provenance.checkpoint_sha256, CHECKPOINT_SHA)
        self.assertEqual(e.provenance.scaler_sha256, SCALER_SHA)
        self.assertEqual(len(e.feature_columns), 157)
        self.assertEqual(type(e.model).__name__, "VectorWorldModelWithStageHead")

    def test_predict_smoke_matches_reference(self) -> None:
        r = sample0_result()
        self.assertAlmostEqual(r["whole_horizon_attack_probability"]["value"], REFERENCE_SAMPLE0_SCORE, delta=1e-6)
        self.assertEqual(r["mitre_stage_trajectory"]["per_step_stage"], REFERENCE_SAMPLE0_STAGES)
        self.assertEqual(sorted(r.keys()), ["current_state", "explanations", "future_state_rollout", "input_metadata", "mitre_stage_trajectory", "provenance", "whole_horizon_attack_probability"])
        json.dumps(r)

    def test_input_contract_6x157(self) -> None:
        e = engine()
        e.validate_feature_vector(np.zeros((6, 157), dtype=np.float32))
        for bad in (np.zeros((5, 157)), np.zeros((6, 156)), np.zeros((157, 6)), np.zeros((1, 6, 157))):
            with self.assertRaises(ValueError):
                e.validate_feature_vector(bad.astype(np.float32))
        nan = np.zeros((6, 157), dtype=np.float32)
        nan[2, 3] = np.nan
        with self.assertRaises(ValueError):
            e.validate_feature_vector(nan)
        with self.assertRaises(TypeError):
            e.validate_feature_vector([[0.0] * 157] * 6)

    def test_explanation_output_reports_gradient_x_input(self) -> None:
        ex = sample0_result()["explanations"]
        self.assertEqual(ex["future_attack_risk_prediction"]["explanation_method"], "Gradient \u00d7 Input")
        self.assertEqual(ex["mitre_stage_prediction"]["explanation_method"], "Gradient \u00d7 Input")
        top = ex["current_state_evidence"]["top_attack_risk_features"]
        self.assertEqual(len(top), 10)
        self.assertTrue(all(np.isfinite(t["importance"]) and t["importance"] >= 0 for t in top))
        self.assertEqual(np.array(ex["temporal_evidence"]["attack_risk_attribution"]).shape, (6, 157))
        self.assertTrue(ex["faithful_claims_only"])

    def test_authoritative_output_identical_to_pre_change_fingerprint(self) -> None:
        before = json.loads((OUT / "integrity_before.json").read_text(encoding="utf-8"))["sample_0"]
        r = sample0_result()
        self.assertEqual(r["whole_horizon_attack_probability"]["value"], before["attack_risk_score"])
        self.assertEqual(r["mitre_stage_trajectory"]["per_step_stage"], before["stages"])
        self.assertEqual([(f["feature"], f["importance"]) for f in r["explanations"]["current_state_evidence"]["top_attack_risk_features"][:5]],
                         [tuple(x) for x in before["top5_attack_features"]])


# ---------------------------------------------------------------------------
# I3 / F: attribution presentation
# ---------------------------------------------------------------------------


class AttributionFormattingTests(unittest.TestCase):
    def test_small_nonzero_value_is_not_rendered_as_zero(self) -> None:
        for v in (9.735952062328579e-07, 1.234e-06, 1e-9, 5.5e-05, 9.99e-04):
            shown = dashboard_views.format_attribution(v)
            self.assertNotEqual(shown, "0.0000")
            self.assertNotEqual(float(shown), 0.0, f"{v} rendered as {shown}")
            self.assertAlmostEqual(float(shown) / v, 1.0, delta=0.006)

    def test_formats(self) -> None:
        self.assertEqual(dashboard_views.format_attribution(9.735952062328579e-07), "9.74e-07")
        self.assertEqual(dashboard_views.format_attribution(1.234e-06), "1.23e-06")
        self.assertEqual(dashboard_views.format_attribution(0.5972327589988708), "0.5972")
        self.assertEqual(dashboard_views.format_attribution(1e-3), "0.0010")
        self.assertEqual(dashboard_views.format_attribution(0.0), "0")  # only a genuine zero renders as zero

    def test_rows_keep_exact_raw_values_and_relative_share(self) -> None:
        feats = [{"feature": "a", "importance": 9.735952062328579e-07}, {"feature": "b", "importance": 4.867976031164289e-07}, {"feature": "c", "importance": 0.0}]
        rows = dashboard_views.build_attribution_rows(feats)
        self.assertEqual([float(r["raw"]) for r in rows], [f["importance"] for f in feats])  # exact round trip
        self.assertEqual([r["relative_percent"] for r in rows], ["100", "50", "0"])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])

    def test_underlying_engine_values_are_not_modified_by_presentation(self) -> None:
        r = sample0_result()
        before = copy.deepcopy(r["explanations"]["current_state_evidence"]["top_attack_risk_features"])
        dashboard_views.build_attribution_rows(r["explanations"]["current_state_evidence"]["top_attack_risk_features"])
        self.assertEqual(r["explanations"]["current_state_evidence"]["top_attack_risk_features"], before)


class AttributionRenderingTests(unittest.TestCase):
    """Through the real view -> template -> HTTP path."""

    def test_real_demo_sample_shows_visible_nonzero_values(self) -> None:
        html = rendered("auth")
        text = visible_text(html)
        block = text[text.index("Top features for the attack-risk score"):text.index("Values are magnitudes")]
        self.assertNotIn("0.0000", block)
        self.assertRegex(block, r"\d\.\d{2}e-0\d")
        top = sample0_result()["explanations"]["current_state_evidence"]["top_attack_risk_features"]
        self.assertGreater(top[0]["importance"], 0.0)
        self.assertLess(top[0]["importance"], 1e-3)  # the regime that used to be rounded to 0.0000
        self.assertIn(dashboard_views.format_attribution(top[0]["importance"]), block)
        self.assertIn("100% of top", block)

    def test_injected_1e_minus_6_value_renders_and_raw_value_is_kept_exactly(self) -> None:
        real = copy.deepcopy(sample0_result())
        feats = real["explanations"]["current_state_evidence"]["top_attack_risk_features"]
        for f, v in zip(feats, [1.234e-06, 6.17e-07] + [1.0e-07] * 8):
            f["importance"] = v
        with mock.patch.object(dashboard_views.AuthoritativeForecastService, "predict_demo_sample", return_value=real):
            html = page("/dashboard/forecast/authoritative/")
        self.assertIn("1.23e-06", html)
        self.assertIn("Exact raw value: 1.234e-06", html)  # tooltip carries the unrounded value
        self.assertIn("50% of top", html)
        self.assertNotRegex(visible_text(html), r"\b0\.0000 \u00b7")

    def test_method_label_is_gradient_x_input_and_never_shap(self) -> None:
        html = rendered("auth")
        text = visible_text(html)
        self.assertIn("Gradient \u00d7 Input", text)
        self.assertIn("post-hoc", text.lower())
        self.assertNotIn("SHAP", html)
        self.assertNotIn("SHAP", (REPO_ROOT / "templates/dashboard/authoritative_forecast.html").read_text(encoding="utf-8"))

    def test_api_response_still_carries_raw_unrounded_values(self) -> None:
        payload = client().get("/api/authoritative-predict/?index=0").json()
        api_top = payload["explanations"]["current_state_evidence"]["top_attack_risk_features"]
        self.assertEqual([t["importance"] for t in api_top], [t["importance"] for t in sample0_result()["explanations"]["current_state_evidence"]["top_attack_risk_features"]])
        self.assertLess(api_top[0]["importance"], 1e-3)


# ---------------------------------------------------------------------------
# C2 / G / H / I / J: rendered dashboard HTML
# ---------------------------------------------------------------------------


class LandingPageTests(unittest.TestCase):
    def test_stale_legacy_strings_are_gone_from_rendered_landing_page(self) -> None:
        text = visible_text(rendered("landing"))
        for stale in STALE_LANDING_STRINGS:
            self.assertNotIn(stale.lower(), text.lower(), stale)

    def test_stale_strings_gone_from_template_source(self) -> None:
        source = (REPO_ROOT / "templates/dashboard/index.html").read_text(encoding="utf-8")
        for stale in STALE_LANDING_STRINGS + ["best_world_model", "Live model"]:
            self.assertNotIn(stale.lower(), source.lower(), stale)

    def test_landing_page_states_six_windows_and_input_contract(self) -> None:  # H
        text = visible_text(rendered("landing"))
        self.assertIn("six most recent 10-second", text)
        self.assertIn("next six windows", text)
        self.assertIn("157 features", text)
        self.assertIn("t+1 \u2026 t+6", text)
        self.assertRegex(text, r"6\s*windows")
        self.assertIn("6 \u00d7 157", text)

    def test_landing_page_states_whole_horizon_scalar_not_per_step(self) -> None:  # I
        text = visible_text(rendered("landing")).lower()
        self.assertIn("attack-risk score", text)
        self.assertIn("one score for the whole horizon", text)
        self.assertIn("not a six-element per-step series", text)
        self.assertIn("not a calibrated probability", text)

    def test_landing_page_states_stage_trajectory_and_gradient_x_input(self) -> None:  # J (landing side)
        text = visible_text(rendered("landing"))
        self.assertIn("six-step MITRE-stage trajectory", text)
        self.assertIn("Gradient \u00d7 Input", text)

    def test_landing_page_does_not_claim_live_detection_calibration_or_shap(self) -> None:
        text = visible_text(rendered("landing"))
        self.assertIn("not live detection", text)
        self.assertIn("does not perform live network detection", text)
        self.assertNotIn("SHAP", text)
        self.assertNotRegex(text.lower(), r"(?<!not a )calibrated probabilit")
        self.assertNotRegex(text.lower(), r"per-step attack probabilit")
        self.assertNotIn("Live model", text)

    def test_landing_page_reports_artifact_availability_truthfully(self) -> None:
        text = visible_text(rendered("landing"))
        status = dashboard_views.get_authoritative_artifact_status()
        self.assertTrue(status["authoritative_artifacts_ready"], status)
        self.assertIn("Authoritative model artifacts found", text)
        self.assertIn("best_stage_head.pt", text)

    def test_artifact_status_reports_missing_artifacts_instead_of_claiming_offline_model(self) -> None:
        with mock.patch.object(dashboard_views, "AUTHORITATIVE_CHECKPOINT_RELPATH", "experiments/results/does_not_exist/best_stage_head.pt"):
            status = dashboard_views.get_authoritative_artifact_status()
        self.assertFalse(status["authoritative_artifacts_ready"])
        self.assertFalse(status["authoritative_artifacts"][0]["present"])

    def test_status_paths_mirror_the_engine_constants(self) -> None:
        self.assertEqual((REPO_ROOT / dashboard_views.AUTHORITATIVE_CHECKPOINT_RELPATH).resolve(), inference_engine.STAGE_HEAD_CHECKPOINT_PATH.resolve())
        self.assertEqual((REPO_ROOT / dashboard_views.AUTHORITATIVE_SCALER_RELPATH).resolve(), inference_engine.RUN1_SCALER_PATH.resolve())
        self.assertEqual((REPO_ROOT / dashboard_views.AUTHORITATIVE_WINDOWS_RELPATH).resolve(), inference_engine.DEFAULT_WINDOWS_DIR.resolve())


class AuthoritativePageTests(unittest.TestCase):
    def test_page_renders_200_with_expected_template(self) -> None:  # G
        resp = client().get("/dashboard/forecast/authoritative/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("dashboard/authoritative_forecast.html", [t.name for t in resp.templates if t.name])

    def test_whole_horizon_wording_is_attack_risk_score_not_probability(self) -> None:  # I
        text = visible_text(rendered("auth"))
        self.assertIn("Attack-risk score (whole horizon, t+1 \u2026 t+6)", text)
        self.assertIn("not a per-step series", text)
        self.assertIn("not a six-element per-step series", text)
        self.assertIn("uncalibrated", text.lower())
        self.assertNotIn("Whole-horizon attack probability", text)
        self.assertNotIn("P(attack somewhere", text)
        self.assertNotRegex(text.lower(), r"(?<!no )calibrated probabilit")
        self.assertIn(f"{sample0_result()['whole_horizon_attack_probability']['value']:.4f}", text)

    def test_six_step_future_state_and_input_windows_wording(self) -> None:  # H
        text = visible_text(rendered("auth"))
        self.assertIn("Six-step future state rollout", text)
        self.assertIn("157 features observed across 6 windows", text)
        self.assertIn("t+1, t+2, t+3, t+4, t+5, t+6", text)

    def test_mitre_trajectory_renders_six_steps_in_order(self) -> None:  # J
        html = rendered("auth")
        section = html[html.index("MITRE STAGE TRAJECTORY"):html.index("EXPLANATION")]
        pairs = re.findall(r"<span>(t\+\d)</span><strong>([A-Z_]+)</strong>", section)
        self.assertEqual([p[0] for p in pairs], ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"])
        self.assertEqual([p[1] for p in pairs], sample0_result()["mitre_stage_trajectory"]["per_step_stage"])
        self.assertIn("Predicted attack-stage trajectory", section)
        self.assertIn("this is not an attack risk", section)
        self.assertIn("NOT authoritative MITRE", visible_text(section))

    def test_stage_trajectory_is_produced_by_the_stage_head(self) -> None:  # J (source)
        import torch
        from phase6b_vector_world_model import scale_array
        from phase7b_stage_targets import CLASS_INDEX_TO_STAGE
        e = engine()
        s = e.get_test_sample(1000)
        x = torch.from_numpy(scale_array(np.asarray(s["x_raw"], np.float32)[None], e.scaler)).to(e.device)
        with torch.no_grad():
            _, _, _, z_future = e.model(x)
            manual = [CLASS_INDEX_TO_STAGE[int(i)].name for i in e.model.stage_head(z_future)[0].argmax(-1).cpu().numpy()]
        self.assertEqual(e.predict(np.asarray(s["x_raw"], np.float32), include_explanations=False)["mitre_stage_trajectory"]["per_step_stage"], manual)

    def test_limitations_panel_still_discloses_the_known_limits(self) -> None:
        text = visible_text(rendered("auth"))
        for phrase in ("No native per-step attack probability", "No calibrated probabilities", "No unseen-attack-generalization claim", "not live network data"):
            self.assertIn(phrase, text)


# ---------------------------------------------------------------------------
# K / L: legacy stays non-authoritative; Phase 9N stays out
# ---------------------------------------------------------------------------


class LegacyPathTests(unittest.TestCase):  # K
    def test_legacy_page_is_labeled_non_authoritative_and_not_a_live_model(self) -> None:
        html = rendered("legacy")
        text = visible_text(html)
        self.assertIn("NON-AUTHORITATIVE", text)
        source = (REPO_ROOT / "templates/dashboard/forecast.html").read_text(encoding="utf-8")
        self.assertNotIn("Live model output", source)
        self.assertIn("Legacy heuristic output", source)

    def test_landing_page_no_longer_invokes_the_legacy_forecaster(self) -> None:
        with mock.patch.object(dashboard_views, "get_forecast_context", side_effect=AssertionError("legacy heuristic must not run on the landing page")):
            resp = client().get("/dashboard/")
        self.assertEqual(resp.status_code, 200)

    def test_legacy_heuristic_is_still_a_heuristic_and_separate(self) -> None:
        source = Path(service_module.__file__).read_text(encoding="utf-8")
        self.assertIn("_heuristic_risk", source)
        self.assertIn("heuristic_fallback", source)
        out = AttackForecastService(REPO_ROOT / "models" / "attack_forecaster.pt").forecast(np.ones((3, 8), dtype=np.float32))
        self.assertEqual(out["model_source"], "heuristic_fallback")
        self.assertFalse((REPO_ROOT / "models" / "attack_forecaster.pt").exists())

    def test_authoritative_service_never_references_the_legacy_service(self) -> None:
        tree = ast.parse(Path(service_module.__file__).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuthoritativeForecastService")
        names = {n.id for n in ast.walk(cls) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute)}
        self.assertNotIn("AttackForecastService", names)
        self.assertNotIn("_heuristic_risk", names)

    def test_authoritative_context_never_falls_back_to_a_heuristic(self) -> None:
        class Broken:
            @staticmethod
            def predict_demo_sample(index=0):
                raise FileNotFoundError("checkpoint missing (simulated)")
        with mock.patch.object(dashboard_views, "AuthoritativeForecastService", Broken):
            ctx = dashboard_views.get_authoritative_forecast_context()
        self.assertFalse(ctx["authoritative_available"])
        self.assertIsNone(ctx["authoritative_forecast"])
        self.assertIn("not found", ctx["authoritative_error"])

    def test_readme_labels_legacy_path_non_authoritative(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("NON-AUTHORITATIVE", readme)
        self.assertIn("is **not shipped**", readme)


class Phase9NExcludedTests(unittest.TestCase):  # L
    FILES = ["experiments/inference_engine.py", "world_model/inference_service.py", "dashboard/views.py", "api/views.py"]
    PATTERN = re.compile(r"per_step_risk|PerStepAttackRisk|phase9n|best_per_step", re.I)

    def test_no_reference_in_authoritative_code_or_templates(self) -> None:
        paths = [REPO_ROOT / f for f in self.FILES] + sorted((REPO_ROOT / "templates/dashboard").glob("*.html")) + [REPO_ROOT / "README.md"]
        for p in paths:
            hits = [l for l in p.read_text(encoding="utf-8").splitlines() if self.PATTERN.search(l)]
            if p.name == "README.md":  # README may name 9N only to say it is excluded
                self.assertTrue(all("excluded" in l.lower() or "RED" in l for l in hits), hits)
            else:
                self.assertEqual(hits, [], f"{p} references Phase 9N: {hits}")

    def test_no_import_of_phase9n_or_track_b(self) -> None:
        for f in self.FILES:
            tree = ast.parse((REPO_ROOT / f).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node, ast.Import) else []
                for m in mods:
                    self.assertNotIn("phase9n", m)
                    self.assertNotEqual(m, "world_model.world_model")

    def test_engine_model_has_no_per_step_risk_head(self) -> None:
        e = engine()
        self.assertFalse(any("per_step_risk" in name for name, _ in e.model.named_parameters()))
        self.assertFalse(hasattr(e.model, "per_step_risk_head"))

    def test_no_per_step_probability_field_in_engine_output(self) -> None:
        r = sample0_result()
        self.assertEqual([k for k in r if "per_step" in k], [])
        self.assertIsInstance(r["whole_horizon_attack_probability"]["value"], float)


# ---------------------------------------------------------------------------
# C1: README
# ---------------------------------------------------------------------------


class ReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    def test_legacy_checkpoint_is_not_presented_as_the_model(self) -> None:
        for line in self.text.splitlines():
            if "attack_forecaster.pt" in line:
                self.assertTrue(re.search(r"not shipped|non-authoritative|legacy", line, re.I), line)
        self.assertNotIn("python train.py", self.text)
        self.assertNotIn("python run_pipeline.py", self.text)
        self.assertNotIn("C:\\VIGNESHWARAN", self.text)

    def test_documents_the_authoritative_path_and_contract(self) -> None:
        for needle in ("experiments/inference_engine.py", "AuthoritativeForecastService", "NetOracleInferenceEngine",
                       "experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt",
                       "experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib",
                       "(6, 157)", "six-step", "Gradient x Input", "python manage.py runserver", "pip install -r requirements.txt"):
            self.assertIn(needle, self.text, needle)

    def test_hashes_in_readme_match_the_real_files(self) -> None:
        self.assertIn(CHECKPOINT_SHA, self.text)
        self.assertIn(SCALER_SHA, self.text)
        self.assertIn(RUN1_SHA, self.text)
        self.assertEqual(sha256(inference_engine.STAGE_HEAD_CHECKPOINT_PATH), CHECKPOINT_SHA)
        self.assertEqual(sha256(inference_engine.RUN1_SCALER_PATH), SCALER_SHA)
        self.assertEqual(sha256(inference_engine.RUN1_DIR / "model/best_model.pt"), RUN1_SHA)

    def test_every_shap_mention_is_a_denial(self) -> None:
        for line in self.text.splitlines():
            if re.search(r"\bshap\b", line, re.I):
                self.assertRegex(line, r"\*\*not\*\*|is not|not used|not required", line)

    def test_no_forbidden_claims(self) -> None:
        low = self.text.lower()
        for line in self.text.splitlines():
            l = line.lower()
            if "calibrated" in l:
                self.assertRegex(l, r"not|no calibrated|uncalibrated|isn't", line)
            if "live" in l and "detection" in l:
                self.assertRegex(l, r"not|no live|does not|isn't", line)
            if "per-step" in l and "probab" in l:
                self.assertRegex(l, r"not|no per-step|excluded", line)
        self.assertNotIn("docker", low.replace("no docker configuration is provided", ""))
        self.assertNotRegex(low, r"generalizes to unseen|zero-shot|detects unseen|novel attack detection")
        self.assertNotRegex(low, r"packet\+flow fusion (is )?implemented")

    def test_states_the_known_limitations(self) -> None:
        for needle in ("Temporal Transformer is the stronger binary detector", "93.1%", "122", "0.834", "0.112", "INITIAL_ACCESS", "0.678", "not established", "replay window"):
            self.assertIn(needle, self.text, needle)


# ---------------------------------------------------------------------------
# I2: SIH deck
# ---------------------------------------------------------------------------


class DeckTests(unittest.TestCase):
    DECK = REPO_ROOT / "world_model/Presentation SIH.pptx"

    @staticmethod
    def slide_texts(z: zipfile.ZipFile) -> dict:
        return {n: " ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8"))) for n in sorted(z.namelist()) if re.match(r"ppt/slides/slide\d+\.xml$", n)}

    def test_no_shap_anywhere_in_the_deck_package(self) -> None:
        with zipfile.ZipFile(self.DECK) as z:
            self.assertIsNone(z.testzip())
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    self.assertNotRegex(z.read(name).decode("utf-8", "replace"), r"\bSHAP\b", name)

    def test_slides_3_4_5_now_say_gradient_x_input(self) -> None:
        with zipfile.ZipFile(self.DECK) as z:
            texts = self.slide_texts(z)
        self.assertIn("Gradient \u00d7 Input feature attribution clarifies why each forecast was made.", texts["ppt/slides/slide3.xml"])
        self.assertIn("Gradient \u00d7 Input", texts["ppt/slides/slide3.xml"].split("PyTorch")[-1])
        self.assertIn("Gradient \u00d7 Input attribution", texts["ppt/slides/slide4.xml"])
        self.assertIn("Explainability: Gradient \u00d7 Input", re.sub(r"\s+", " ", texts["ppt/slides/slide5.xml"]).replace("Explainability:  ", "Explainability: "))

    def test_slide_xml_is_well_formed(self) -> None:
        from xml.dom import minidom
        with zipfile.ZipFile(self.DECK) as z:
            for name in z.namelist():
                if re.match(r"ppt/slides/slide\d+\.xml$", name):
                    minidom.parseString(z.read(name))

    def test_only_the_three_shap_slides_changed_versus_the_original_commit(self) -> None:
        raw = subprocess.run(["git", "show", "4e54c75:world_model/Presentation SIH.pptx"], cwd=REPO_ROOT, capture_output=True)
        if raw.returncode != 0:
            self.skipTest("original commit 4e54c75 not available")
        import io
        with zipfile.ZipFile(io.BytesIO(raw.stdout)) as orig, zipfile.ZipFile(self.DECK) as new:
            self.assertEqual(orig.namelist(), new.namelist())
            changed = [n for n in orig.namelist() if orig.read(n) != new.read(n)]
        self.assertEqual(changed, ["ppt/slides/slide3.xml", "ppt/slides/slide4.xml", "ppt/slides/slide5.xml"])

    def test_unrelated_slide_text_is_untouched(self) -> None:
        raw = subprocess.run(["git", "show", "4e54c75:world_model/Presentation SIH.pptx"], cwd=REPO_ROOT, capture_output=True)
        if raw.returncode != 0:
            self.skipTest("original commit 4e54c75 not available")
        import io
        with zipfile.ZipFile(io.BytesIO(raw.stdout)) as orig, zipfile.ZipFile(self.DECK) as new:
            o, n = self.slide_texts(orig), self.slide_texts(new)
        for name in o:
            if name not in ("ppt/slides/slide3.xml", "ppt/slides/slide4.xml", "ppt/slides/slide5.xml"):
                self.assertEqual(o[name], n[name], name)


# ---------------------------------------------------------------------------
# O2: reproducible provenance
# ---------------------------------------------------------------------------


class ProvenanceTests(unittest.TestCase):
    MANIFEST = OUT / "authoritative_artifacts.json"

    def test_manifest_hashes_match_the_files_and_the_phase9k_manifest(self) -> None:
        m = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        by_id = {a["id"]: a for a in m["artifacts"]}
        self.assertEqual(by_id["checkpoint"]["sha256"], CHECKPOINT_SHA)
        self.assertEqual(by_id["scaler"]["sha256"], SCALER_SHA)
        for a in m["artifacts"]:
            self.assertEqual(sha256(REPO_ROOT / a["path"]), a["sha256"])
            self.assertEqual((REPO_ROOT / a["path"]).stat().st_size, a["size_bytes"])
            self.assertTrue(a["matches_phase9k_manifest"])
        self.assertEqual(m["expected_input_shape"], [6, 157])
        self.assertEqual(m["feature_schema"]["count"], 157)
        self.assertEqual(len(m["window_files"]), 10)
        for w in m["window_files"]:
            self.assertFalse(w["tracked_by_git"])
            self.assertTrue(w["retrieval"])

    def test_verifier_script_passes_on_this_checkout(self) -> None:
        r = subprocess.run([PY, "scripts/verify_authoritative_artifacts.py"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout[-600:])
        self.assertIn("ALL AUTHORITATIVE ARTIFACTS VERIFIED", r.stdout)

    def test_verifier_script_detects_a_missing_or_altered_file(self) -> None:
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "repo"
            (fake / "scripts").mkdir(parents=True)
            (fake / "experiments/results/phase10b_submission_remediation").mkdir(parents=True)
            shutil.copy(REPO_ROOT / "scripts/verify_authoritative_artifacts.py", fake / "scripts")
            shutil.copy(self.MANIFEST, fake / "experiments/results/phase10b_submission_remediation")
            r = subprocess.run([PY, "scripts/verify_authoritative_artifacts.py", "--no-data"], cwd=fake, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("MISSING", r.stdout)

    def test_gitignore_keeps_broad_rule_and_only_excepts_authoritative_artifacts(self) -> None:
        lines = [l.strip() for l in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        self.assertIn("*.pt", lines)
        self.assertIn("*.pth", lines)
        self.assertEqual(sorted(l for l in lines if l.startswith("!")), sorted([
            "!experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt",
            "!experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt"]))

    @unittest.skipUnless((REPO_ROOT / ".git").exists(), "not a git checkout")
    def test_git_actually_tracks_the_authoritative_source_and_artifacts(self) -> None:
        tracked = set(git("ls-files").stdout.splitlines())
        m = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        required = [a["path"] for a in m["artifacts"]] + [s["path"] for s in m["authoritative_source_closure_and_wiring"]]
        required += ["experiments/inference_engine.py", "README.md", "requirements.txt", "scripts/verify_authoritative_artifacts.py", "tests/test_phase10b_remediation.py"]
        missing = sorted(set(required) - tracked)
        self.assertEqual(missing, [], f"not tracked by Git: {missing}")

    @unittest.skipUnless((REPO_ROOT / ".git").exists(), "not a git checkout")
    def test_other_checkpoints_remain_ignored(self) -> None:
        for rel in ("experiments/results/phase9n_per_step_risk/model/best_per_step_risk_head.pt", "experiments/results/phase5_temporal_transformer/transformer/best_model.pt"):
            if (REPO_ROOT / rel).exists():
                self.assertEqual(git("check-ignore", "-q", rel).returncode, 0, rel)

    @unittest.skipUnless((REPO_ROOT / ".git").exists(), "not a git checkout")
    def test_authoritative_artifacts_are_not_ignored(self) -> None:
        for rel in ("experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt", "experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib"):
            self.assertNotEqual(git("check-ignore", "-q", rel).returncode, 0, rel)


class FrozenCoreUnchangedTests(unittest.TestCase):
    def test_frozen_sources_unchanged_since_before_fingerprint(self) -> None:
        before = json.loads((OUT / "integrity_before.json").read_text(encoding="utf-8"))
        for rel, digest in before["frozen_source_sha256"].items():
            self.assertEqual(sha256(REPO_ROOT / rel), digest, f"frozen source changed: {rel}")

    def test_artifact_hashes_unchanged(self) -> None:
        before = json.loads((OUT / "integrity_before.json").read_text(encoding="utf-8"))["authoritative_artifacts"]
        self.assertEqual(sha256(inference_engine.STAGE_HEAD_CHECKPOINT_PATH), before["checkpoint_best_stage_head.pt"])
        self.assertEqual(sha256(inference_engine.RUN1_SCALER_PATH), before["scaler.joblib"])


if __name__ == "__main__":
    unittest.main()
