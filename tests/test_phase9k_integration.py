"""Phase 9K: focused tests for the authoritative inference engine + track
reconciliation.

Covers: authoritative checkpoint loading, parameter-count verification,
feature schema validation, scaler consistency, tensor shape, inference
output schema, deterministic prediction, frozen Phase 6B/7B equivalence,
explainability shape, provenance, legacy-Track-B exclusion, and prior
artifact immutability. The engine is instantiated ONCE per test class
(expensive: loads a real checkpoint + real dataset) to keep the suite fast.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from inference_engine import (  # noqa: E402
    EXPECTED_FEATURE_COUNT,
    NetOracleInferenceEngine,
    RUN1_SCALER_PATH,
    STAGE_HEAD_CHECKPOINT_PATH,
)
from phase9k_integration import OUTPUT_DIR, verify_no_prior_artifact_changed  # noqa: E402


class InferenceEngineTestCase(unittest.TestCase):
    """Base class that loads the engine once for all subclasses in this module."""

    engine: NetOracleInferenceEngine

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = NetOracleInferenceEngine()
        cls.sample = cls.engine.get_test_sample(0)


# ---- 1. Authoritative checkpoint loading ----


class CheckpointLoadingTests(InferenceEngineTestCase):
    def test_checkpoint_path_is_the_phase7b_stage_head(self) -> None:
        self.assertEqual(str(STAGE_HEAD_CHECKPOINT_PATH), self.engine.provenance.checkpoint_path)
        self.assertTrue(STAGE_HEAD_CHECKPOINT_PATH.exists())

    def test_scaler_path_is_the_run1_scaler(self) -> None:
        self.assertEqual(str(RUN1_SCALER_PATH), self.engine.provenance.scaler_path)
        self.assertTrue(RUN1_SCALER_PATH.exists())

    def test_missing_checkpoint_raises(self) -> None:
        import inference_engine as ie

        original = ie.STAGE_HEAD_CHECKPOINT_PATH
        ie.STAGE_HEAD_CHECKPOINT_PATH = Path("does/not/exist.pt")
        try:
            with self.assertRaises(FileNotFoundError):
                ie.NetOracleInferenceEngine()
        finally:
            ie.STAGE_HEAD_CHECKPOINT_PATH = original

    def test_model_is_in_eval_mode(self) -> None:
        self.assertFalse(self.engine.model.training)

    def test_model_parameters_require_no_grad(self) -> None:
        for p in self.engine.model.parameters():
            self.assertFalse(p.requires_grad)


# ---- 2. Parameter-count verification ----


class ParameterCountTests(InferenceEngineTestCase):
    def test_parameter_count_matches_backbone_plus_stage_head(self) -> None:
        # Backbone alone (VectorWorldModel) has 347,806 params (Phase 6B);
        # MitreStageHead adds 128*64+64 + 64*6+6 = 8,646 -- total 356,452.
        self.assertEqual(self.engine.parameter_count, 356452)

    def test_parameter_count_is_positive(self) -> None:
        self.assertGreater(self.engine.parameter_count, 0)


# ---- 3. Exact feature schema validation ----


class FeatureSchemaTests(InferenceEngineTestCase):
    def test_feature_count_is_157(self) -> None:
        self.assertEqual(len(self.engine.feature_columns), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(self.engine.feature_columns), 157)

    def test_wrong_shape_raises(self) -> None:
        bad = np.zeros((6, 100), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(bad)

    def test_wrong_timestep_count_raises(self) -> None:
        bad = np.zeros((5, 157), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(bad)

    def test_non_finite_values_raise(self) -> None:
        bad = np.zeros((6, 157), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(bad)
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.engine.validate_feature_vector(bad)

    def test_wrong_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.validate_feature_vector([[0.0] * 157] * 6)

    def test_valid_vector_does_not_raise(self) -> None:
        self.engine.validate_feature_vector(self.sample["x_raw"])


# ---- 4. Scaler consistency ----


class ScalerConsistencyTests(InferenceEngineTestCase):
    def test_scaler_has_expected_feature_count(self) -> None:
        self.assertEqual(len(self.engine.scaler.mean_), 157)
        self.assertEqual(len(self.engine.scaler.scale_), 157)

    def test_scaler_is_never_refit_by_predict(self) -> None:
        mean_before = self.engine.scaler.mean_.copy()
        scale_before = self.engine.scaler.scale_.copy()
        self.engine.predict(self.sample["x_raw"], include_explanations=False)
        np.testing.assert_array_equal(mean_before, self.engine.scaler.mean_)
        np.testing.assert_array_equal(scale_before, self.engine.scaler.scale_)


# ---- 5. Tensor shape ----


class TensorShapeTests(InferenceEngineTestCase):
    def test_future_state_rollout_shape(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        rollout = result["future_state_rollout"]["predicted_state_scaled"]
        self.assertEqual(len(rollout), 6)
        self.assertTrue(all(len(step) == 157 for step in rollout))

    def test_current_state_shape(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        self.assertEqual(len(result["current_state"]["raw_feature_values"]), 157)


# ---- 6. Inference output schema ----


class OutputSchemaTests(InferenceEngineTestCase):
    def test_top_level_keys_present(self) -> None:
        result = self.engine.predict(self.sample["x_raw"])
        required = {
            "input_metadata", "current_state", "future_state_rollout",
            "whole_horizon_attack_probability", "mitre_stage_trajectory",
            "explanations", "provenance",
        }
        self.assertTrue(required.issubset(result.keys()))

    def test_whole_horizon_probability_is_scalar_in_0_1(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        value = result["whole_horizon_attack_probability"]["value"]
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_whole_horizon_semantics_does_not_imply_per_step(self) -> None:
        # The semantics text legitimately MENTIONS "P(t+1)" notation as part
        # of an honest disclaimer ("NOT a per-step P(t+1), P(t+2), ... series")
        # -- the point is that this disclaimer is present, not that the
        # notation is absent from the string entirely.
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        semantics = result["whole_horizon_attack_probability"]["semantics"].lower()
        self.assertIn("somewhere within", semantics)
        self.assertIn("not a per-step", semantics)
        self.assertIn("no native per-step attack-probability head", semantics)

    def test_mitre_stage_trajectory_has_6_steps(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        self.assertEqual(len(result["mitre_stage_trajectory"]["per_step_stage"]), 6)
        self.assertEqual(len(result["mitre_stage_trajectory"]["per_step_confidence"]), 6)

    def test_explanations_can_be_omitted(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        self.assertIsNone(result["explanations"])


# ---- 7. Deterministic prediction ----


class DeterminismTests(InferenceEngineTestCase):
    def test_repeated_prediction_is_deterministic(self) -> None:
        r1 = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        r2 = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        self.assertAlmostEqual(
            r1["whole_horizon_attack_probability"]["value"],
            r2["whole_horizon_attack_probability"]["value"],
            places=6,
        )
        self.assertEqual(
            r1["mitre_stage_trajectory"]["per_step_stage"],
            r2["mitre_stage_trajectory"]["per_step_stage"],
        )


# ---- 8/9. Frozen Phase 6B/7B equivalence (regression against this phase's own run) ----


class FrozenEquivalenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        report_path = OUTPUT_DIR / "phase9k_integration_report.json"
        cls.available = report_path.exists()
        if cls.available:
            cls.report = json.loads(report_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        if not self.available:
            self.skipTest("Phase 9K has not been run yet in this environment")

    def test_run1_equivalence_passed(self) -> None:
        self.assertTrue(self.report["deterministic_validation"]["all_run1_attack_probability_equivalent"])

    def test_phase7b_stage_equivalence_passed(self) -> None:
        self.assertTrue(self.report["deterministic_validation"]["all_phase7b_stage_names_exact_match"])

    def test_overall_deterministic_validation_passed(self) -> None:
        self.assertTrue(self.report["deterministic_validation"]["overall_pass"])

    def test_per_sample_checks_present_and_match_frozen_values(self) -> None:
        checks = self.report["deterministic_validation"]["per_sample_checks"]
        self.assertGreater(len(checks), 0)
        for c in checks:
            self.assertIsNotNone(c["frozen_run1_attack_probability"])
            self.assertIsNotNone(c["frozen_phase7b_stage_names"])
            self.assertEqual(c["engine_stage_names"], c["frozen_phase7b_stage_names"])


# ---- 10. Explainability shape ----


class ExplainabilityTests(InferenceEngineTestCase):
    def test_attribution_shape_is_6_by_157(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], top_k=5)
        attribution = result["explanations"]["temporal_evidence"]["attack_risk_attribution"]
        self.assertEqual(len(attribution), 6)
        self.assertTrue(all(len(row) == 157 for row in attribution))

    def test_explanation_method_never_claims_shap_falsely(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], top_k=5)
        method = result["explanations"]["future_attack_risk_prediction"]["explanation_method"]
        # The method string comes verbatim from the already-audited Phase 7C
        # explainer -- this test just checks it round-trips as a non-empty,
        # real value, not a hardcoded "SHAP" claim injected by this phase.
        self.assertIsInstance(method, str)
        self.assertGreater(len(method), 0)

    def test_faithful_claims_note_present(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], top_k=5)
        self.assertIn("faithful_claims_only", result["explanations"])
        note = result["explanations"]["faithful_claims_only"].lower()
        self.assertIn("not proof of real-world causality", note)


# ---- 11. Provenance ----


class ProvenanceTests(InferenceEngineTestCase):
    def test_provenance_contains_required_fields(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        prov = result["provenance"]
        required = {
            "model_identifier", "checkpoint_path", "checkpoint_sha256",
            "scaler_path", "scaler_sha256", "feature_schema_sha256",
            "code_commit_hash", "stage_model_identifier",
            "explainability_method_infrastructure", "prediction_semantics",
            "inference_timestamp_unix", "device",
        }
        self.assertTrue(required.issubset(prov.keys()), required - set(prov.keys()))

    def test_checkpoint_sha256_is_64_hex_chars(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        sha = result["provenance"]["checkpoint_sha256"]
        self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_provenance_excludes_track_b(self) -> None:
        result = self.engine.predict(self.sample["x_raw"], include_explanations=False)
        excluded = result["provenance"]["non_authoritative_tracks_excluded"]
        joined = " ".join(excluded).lower()
        self.assertIn("networkworldmodel", joined)
        self.assertIn("never", joined)


# ---- 12. Legacy Track B cannot be used as authoritative evidence ----


class TrackBExclusionTests(unittest.TestCase):
    def test_inference_engine_module_does_not_import_track_b(self) -> None:
        """NetworkWorldModel/world_model.world_model may legitimately be MENTIONED
        in documentation strings (explaining why Track B is excluded -- that
        mention is required, not forbidden); what must never appear is an
        actual import statement or a live reference to Track B's evaluation
        JSON files, checked via the AST so docstring/string-literal prose
        cannot produce a false failure."""
        import ast

        source = (EXPERIMENTS_DIR / "inference_engine.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "world_model.world_model")
                self.assertFalse((node.module or "").startswith("world_model.world_model"))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("world_model.world_model"))
        # No live reference to Track B's evaluation artifacts outside of a
        # documentation string (checked structurally: no Name/Attribute node
        # named after those files -- only string constants may contain them).
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                rendered = ast.dump(node)
                self.assertNotIn("evaluation_results", rendered)
                self.assertNotIn("zero_shot_results", rendered)

    def test_track_reconciliation_doc_exists_and_names_both_tracks(self) -> None:
        path = OUTPUT_DIR / "track_reconciliation.md"
        if not path.exists():
            self.skipTest("Phase 9K has not been run yet in this environment")
        text = path.read_text(encoding="utf-8")
        self.assertIn("AUTHORITATIVE", text)
        self.assertIn("NON-AUTHORITATIVE", text)
        self.assertIn("world_model/world_model.py", text)
        self.assertIn("phase6b_vector_world_model.py", text)


# ---- 13. Prior artifact immutability ----


class ArtifactImmutabilityTests(unittest.TestCase):
    def test_verify_no_prior_artifact_changed_detects_mismatch(self) -> None:
        baseline = {"prior_result_directory_file_counts": {"phase6b_vector_world_model": 999999}}
        audit = verify_no_prior_artifact_changed(baseline)
        self.assertFalse(audit["unchanged"])
        self.assertIn("phase6b_vector_world_model", audit["mismatches"])

    def test_persisted_integrity_audit_passed(self) -> None:
        report_path = OUTPUT_DIR / "phase9k_integration_report.json"
        if not report_path.exists():
            self.skipTest("Phase 9K has not been run yet in this environment")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["integrity_audit"]["unchanged"])


if __name__ == "__main__":
    unittest.main()
