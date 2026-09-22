"""Phase 9L: focused tests for the Django integration of the Phase 9K
authoritative inference engine.

Covers (16 required areas): Django service import/wiring, authoritative
engine actually used (not a heuristic), Track B never imported/used,
strict input validation (correct shape accepted; wrong shape/type/non-
finite rejected, never silently padded/reordered/dropped), JSON-
serializable output (no raw tensors), whole-horizon-probability semantics
text present, no fabricated per-step attack probability, six-step future
state rollout present, six-step MITRE stage trajectory present with
Phase 7B limitations preserved, gradient-based explainability exposed
with no SHAP UI claim, full provenance fields present, legacy heuristic
path left untouched and separately reachable, no unsupported UI claims,
error handling never silently substitutes the heuristic, Step 8
deterministic Django-vs-direct equivalence, and Step 9 Django smoke test
(URL/view resolution + template rendering).

The authoritative engine (real checkpoint + real dataset) is loaded ONCE
for the whole module via setUpClass, matching the Phase 9K test
convention, to keep the suite fast.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]  # tests/ -> NetOracle/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")

import django  # noqa: E402
django.setup()

from inference_engine import NetOracleInferenceEngine, EXPECTED_FEATURE_COUNT  # noqa: E402
from world_model.inference_service import AuthoritativeForecastService, AttackForecastService  # noqa: E402
import world_model.inference_service as inference_service_module  # noqa: E402
import dashboard.views as dashboard_views_module  # noqa: E402
import api.views as api_views_module  # noqa: E402

from phase9l_django_integration import (  # noqa: E402
    run_step8_django_vs_direct_equivalence,
    run_step9_django_smoke_test,
    FORBIDDEN_UI_CLAIMS,
)

FORBIDDEN_PHRASES = [
    "Per-step attack probability",
    "Calibrated probability",
    "SHAP",
    "Causal prediction",
    "Unseen attack detection",
    "PCAP analysis",
]


class AuthoritativeServiceTestCase(unittest.TestCase):
    """Base class: loads the real authoritative engine once for all subclasses."""

    engine: NetOracleInferenceEngine
    sample: dict
    result: dict

    @classmethod
    def setUpClass(cls) -> None:
        # Force the Django service's singleton to be a freshly-constructed
        # engine, then use it for every test in this module.
        AuthoritativeForecastService._engine = None
        cls.engine = AuthoritativeForecastService.get_engine()
        cls.sample = cls.engine.get_test_sample(0)
        cls.result = AuthoritativeForecastService.predict(
            cls.sample["x_raw"], source_file=cls.sample["source_file"], window_start=cls.sample["window_start"]
        )


# ---------------------------------------------------------------------------
# Area 1-2: service wiring / authoritative engine actually used
# ---------------------------------------------------------------------------


class ServiceWiringTests(AuthoritativeServiceTestCase):
    def test_authoritative_service_importable_from_django_app(self) -> None:
        self.assertTrue(hasattr(inference_service_module, "AuthoritativeForecastService"))

    def test_authoritative_service_singleton_is_real_engine(self) -> None:
        self.assertIsInstance(AuthoritativeForecastService.get_engine(), NetOracleInferenceEngine)

    def test_authoritative_service_singleton_reused_across_calls(self) -> None:
        e1 = AuthoritativeForecastService.get_engine()
        e2 = AuthoritativeForecastService.get_engine()
        self.assertIs(e1, e2)

    def test_dashboard_views_imports_authoritative_service(self) -> None:
        source = Path(dashboard_views_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "inference_service" in node.module:
                imported_names.update(alias.name for alias in node.names)
        self.assertIn("AuthoritativeForecastService", imported_names)

    def test_api_views_imports_authoritative_service(self) -> None:
        source = Path(api_views_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "inference_service" in node.module:
                imported_names.update(alias.name for alias in node.names)
        self.assertIn("AuthoritativeForecastService", imported_names)

    def test_legacy_attack_forecast_service_unmodified_and_still_present(self) -> None:
        # AttackForecastService must still exist (Option B: left in place, not deleted).
        self.assertTrue(hasattr(inference_service_module, "AttackForecastService"))
        self.assertIs(AttackForecastService, inference_service_module.AttackForecastService)


# ---------------------------------------------------------------------------
# Area 3: Track B never imported/used by the Django-reachable path
# ---------------------------------------------------------------------------


class TrackBExclusionTests(unittest.TestCase):
    def _assert_module_does_not_import_track_b(self, module) -> None:
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("world_model.world_model", (node.module or ""))
                self.assertNotEqual(node.module, "world_model.world_model")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name, "world_model.world_model")

    def test_inference_service_does_not_import_track_b(self) -> None:
        self._assert_module_does_not_import_track_b(inference_service_module)

    def test_dashboard_views_does_not_import_track_b(self) -> None:
        self._assert_module_does_not_import_track_b(dashboard_views_module)

    def test_api_views_does_not_import_track_b(self) -> None:
        self._assert_module_does_not_import_track_b(api_views_module)

    def test_inference_engine_does_not_import_track_b(self) -> None:
        import inference_engine as inference_engine_module
        self._assert_module_does_not_import_track_b(inference_engine_module)

    def test_track_b_module_untouched_on_disk(self) -> None:
        # Existence-only check: Track B must not have been deleted (per Phase 9K
        # disposition -- quarantine by documentation, never delete).
        self.assertTrue((REPO_ROOT / "world_model" / "world_model.py").exists())


# ---------------------------------------------------------------------------
# Area 4-6: strict input contract
# ---------------------------------------------------------------------------


class InputContractTests(AuthoritativeServiceTestCase):
    def test_correct_shape_accepted(self) -> None:
        x = np.zeros((6, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        # Should not raise on validation (may produce a low-confidence prediction, that's fine).
        self.engine.validate_feature_vector(x)

    def test_wrong_shape_rejected(self) -> None:
        x = np.zeros((5, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(x)

    def test_wrong_feature_count_rejected(self) -> None:
        x = np.zeros((6, EXPECTED_FEATURE_COUNT - 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(x)

    def test_wrong_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.validate_feature_vector([[0.0] * EXPECTED_FEATURE_COUNT] * 6)

    def test_non_finite_values_rejected(self) -> None:
        x = np.zeros((6, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(x)
        x[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(x)

    def test_django_service_predict_rejects_invalid_shape_without_fallback(self) -> None:
        bad = np.zeros((3, EXPECTED_FEATURE_COUNT), dtype=np.float32)
        with self.assertRaises(ValueError):
            AuthoritativeForecastService.predict(bad)


# ---------------------------------------------------------------------------
# Area 7: JSON-serializable output, no raw tensors
# ---------------------------------------------------------------------------


class JsonSerializabilityTests(AuthoritativeServiceTestCase):
    def test_result_is_json_serializable(self) -> None:
        json.dumps(self.result)  # raises TypeError if a tensor/ndarray leaked through

    def test_no_torch_tensor_or_ndarray_leaks_into_result(self) -> None:
        def walk(value):
            self.assertNotIn("Tensor", type(value).__name__)
            self.assertNotIsInstance(value, np.ndarray)
            if isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        walk(self.result)


# ---------------------------------------------------------------------------
# Area 8-10: whole-horizon probability semantics, no fabricated per-step prob,
# six-step rollout, six-step MITRE trajectory with Phase 7B limitations
# ---------------------------------------------------------------------------


class PredictionContractTests(AuthoritativeServiceTestCase):
    def test_whole_horizon_probability_present_and_scalar(self) -> None:
        prob = self.result["whole_horizon_attack_probability"]["value"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_whole_horizon_semantics_text_present_and_not_per_step(self) -> None:
        semantics = self.result["whole_horizon_attack_probability"]["semantics"]
        self.assertIn("NOT a per-step", semantics)

    def test_no_native_per_step_attack_probability_field_exists(self) -> None:
        # The contract must expose exactly one scalar whole-horizon probability,
        # never a per-step P(t+1)..P(t+6) probability array.
        self.assertNotIn("per_step_attack_probability", self.result)
        self.assertIsInstance(self.result["whole_horizon_attack_probability"]["value"], float)

    def test_six_step_future_state_rollout_present(self) -> None:
        rollout = self.result["future_state_rollout"]
        self.assertEqual(len(rollout["horizon_labels"]), 6)
        self.assertEqual(len(rollout["predicted_state_raw_units"]), 6)
        self.assertEqual(len(rollout["predicted_state_raw_units"][0]), EXPECTED_FEATURE_COUNT)

    def test_six_step_mitre_stage_trajectory_present(self) -> None:
        trajectory = self.result["mitre_stage_trajectory"]
        self.assertEqual(len(trajectory["per_step_stage"]), 6)
        self.assertEqual(len(trajectory["per_step_confidence"]), 6)
        self.assertEqual(len(trajectory["horizon_labels"]), 6)

    def test_mitre_trajectory_preserves_phase7b_limitations_text(self) -> None:
        semantics = self.result["mitre_stage_trajectory"]["semantics"]
        self.assertIn("NOT authoritative", semantics)
        self.assertIn("NOT causal", semantics)


# ---------------------------------------------------------------------------
# Area 11-12: explainability exposed, gradient-based label (no SHAP UI claim
# in the engine-level "future_attack_risk_prediction" contract wrapper text
# authored by this phase -- the raw per-call method string is intentionally
# not interpolated into the HTML page, see templates/dashboard/authoritative_forecast.html)
# ---------------------------------------------------------------------------


class ExplainabilityTests(AuthoritativeServiceTestCase):
    def test_explanations_present(self) -> None:
        self.assertIsNotNone(self.result["explanations"])

    def test_top_attack_risk_features_present(self) -> None:
        top = self.result["explanations"]["current_state_evidence"]["top_attack_risk_features"]
        self.assertGreater(len(top), 0)
        for item in top:
            self.assertIn("feature", item)
            self.assertIn("importance", item)

    def test_faithful_claims_note_present_in_api_payload(self) -> None:
        self.assertIn("faithful_claims_only", self.result["explanations"])
        self.assertTrue(len(self.result["explanations"]["faithful_claims_only"]) > 0)

    def test_dashboard_html_never_renders_literal_shap_claim(self) -> None:
        template_path = REPO_ROOT / "templates" / "dashboard" / "authoritative_forecast.html"
        source = template_path.read_text(encoding="utf-8")
        self.assertNotIn("SHAP", source)
        self.assertIn("Gradient-based feature attribution", source)


# ---------------------------------------------------------------------------
# Area 13: full provenance
# ---------------------------------------------------------------------------


class ProvenanceTests(AuthoritativeServiceTestCase):
    def test_provenance_has_required_fields(self) -> None:
        prov = self.result["provenance"]
        for field in [
            "model_identifier", "checkpoint_path", "checkpoint_sha256",
            "scaler_path", "scaler_sha256", "feature_schema_sha256",
            "code_commit_hash", "device", "inference_duration_seconds",
        ]:
            self.assertIn(field, prov)
            self.assertTrue(prov[field] not in (None, ""), f"provenance.{field} must not be empty")

    def test_checkpoint_sha256_matches_phase9k_authoritative_checkpoint(self) -> None:
        from inference_engine import STAGE_HEAD_CHECKPOINT_PATH
        import hashlib
        digest = hashlib.sha256()
        with STAGE_HEAD_CHECKPOINT_PATH.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        self.assertEqual(self.result["provenance"]["checkpoint_sha256"], digest.hexdigest())


# ---------------------------------------------------------------------------
# Area 14: legacy heuristic path left untouched / no new threshold
# ---------------------------------------------------------------------------


class LegacyPathUntouchedTests(unittest.TestCase):
    def test_attack_forecast_service_heuristic_unchanged(self) -> None:
        source = Path(inference_service_module.__file__).read_text(encoding="utf-8")
        self.assertIn("_heuristic_risk", source)
        self.assertIn("model_source", source)

    def test_run_pipeline_threshold_not_modified_by_this_phase(self) -> None:
        run_pipeline_path = REPO_ROOT / "run_pipeline.py"
        source = run_pipeline_path.read_text(encoding="utf-8")
        self.assertIn("ATTACK_RISK_THRESHOLD = 0.70", source)

    def test_legacy_forecast_template_marks_itself_non_authoritative(self) -> None:
        template_path = REPO_ROOT / "templates" / "dashboard" / "forecast.html"
        source = template_path.read_text(encoding="utf-8")
        self.assertIn("NON-AUTHORITATIVE", source)


# ---------------------------------------------------------------------------
# Area 15: no unsupported UI claims anywhere in the new page
# ---------------------------------------------------------------------------


class UnsupportedClaimsAuditTests(unittest.TestCase):
    def test_new_template_has_no_forbidden_phrases(self) -> None:
        template_path = REPO_ROOT / "templates" / "dashboard" / "authoritative_forecast.html"
        source = template_path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_PHRASES:
            self.assertNotIn(phrase, source, f"forbidden phrase '{phrase}' found in authoritative_forecast.html")


# ---------------------------------------------------------------------------
# Area 16 / Step 6: error handling never silently substitutes the heuristic
# ---------------------------------------------------------------------------


class ErrorHandlingTests(unittest.TestCase):
    def test_get_authoritative_forecast_context_reports_explicit_error_on_bad_engine(self) -> None:
        class _BrokenService:
            @classmethod
            def predict_demo_sample(cls, index: int = 0):
                raise FileNotFoundError("checkpoint missing (simulated)")

        original = dashboard_views_module.AuthoritativeForecastService
        dashboard_views_module.AuthoritativeForecastService = _BrokenService
        try:
            context = dashboard_views_module.get_authoritative_forecast_context()
        finally:
            dashboard_views_module.AuthoritativeForecastService = original

        self.assertFalse(context["authoritative_available"])
        self.assertIsNone(context["authoritative_forecast"])
        self.assertIsNotNone(context["authoritative_error"])
        self.assertIn("checkpoint", context["authoritative_error"].lower())


# ---------------------------------------------------------------------------
# Step 8: deterministic Django-vs-direct equivalence
# ---------------------------------------------------------------------------


class Step8DeterministicEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.equivalence = run_step8_django_vs_direct_equivalence()

    def test_overall_pass(self) -> None:
        self.assertTrue(self.equivalence["overall_pass"])

    def test_probabilities_within_tolerance(self) -> None:
        self.assertTrue(self.equivalence["all_within_tolerance"])

    def test_stage_trajectories_exact_match(self) -> None:
        self.assertTrue(self.equivalence["all_stage_trajectories_exact_match"])

    def test_future_rollouts_close_match(self) -> None:
        self.assertTrue(self.equivalence["all_future_rollouts_close_match"])

    def test_all_django_results_json_serializable(self) -> None:
        self.assertTrue(self.equivalence["all_django_results_json_serializable"])


# ---------------------------------------------------------------------------
# Step 9: Django smoke test
# ---------------------------------------------------------------------------


class Step9DjangoSmokeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke = run_step9_django_smoke_test()

    def test_all_smoke_steps_ok(self) -> None:
        failed = [s for s in self.smoke["steps"] if not s["ok"]]
        self.assertEqual(failed, [], f"smoke test steps failed: {failed}")

    def test_dashboard_view_resolves_and_renders(self) -> None:
        step = next(s for s in self.smoke["steps"] if s["step"] == "dashboard_view_renders")
        self.assertTrue(step["ok"])

    def test_api_view_responds(self) -> None:
        step = next(s for s in self.smoke["steps"] if s["step"] == "api_view_responds_200")
        self.assertTrue(step["ok"])

    def test_no_forbidden_claims_rendered(self) -> None:
        step = next(s for s in self.smoke["steps"] if s["step"] == "dashboard_view_no_forbidden_claims")
        self.assertTrue(step["ok"])


if __name__ == "__main__":
    unittest.main()
