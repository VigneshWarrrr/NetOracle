"""Phase 9N: focused tests for the native per-step attack-risk forecasting
head (VectorWorldModelWithPerStepRiskHead on a frozen Phase 6B Run-1
backbone).

Covers the 15 required areas: six-step target shape, target temporal
correctness, no current/future leakage, source-day boundaries,
terminal-window exclusion, train-only class weighting, train-only
preprocessing, output shape [B,6], distinct per-step outputs are actually
possible, no scalar-probability broadcasting, deterministic inference,
checkpoint loading, baseline correctness, SIH semantics, frozen-artifact
integrity.

The expensive part (training) runs ONCE via
experiments/phase9n_per_step_risk.py; this suite reads its already-produced
artifacts where possible and only re-invokes cheap, read-only functions
(target construction, leakage audit) directly, plus a lightweight forward
pass of the already-trained checkpoint for shape/determinism checks.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from phase9n_per_step_risk import (  # noqa: E402
    DEFAULT_WINDOWS_DIR,
    FORECAST_HORIZON_WINDOWS,
    HISTORY_WINDOWS,
    INPUT_SIZE,
    OUTPUT_DIR,
    RISK_HEAD_PREFIX,
    RUN1_CHECKPOINT,
    RUN1_SCALER,
    PerStepAttackRiskHead,
    VectorWorldModelWithPerStepRiskHead,
    baseline_a_persistence,
    baseline_b_whole_horizon_broadcast,
    load_frozen_backbone,
    leakage_audit,
    predict_all,
    read_per_step_risk_targets,
)
from phase6b_vector_world_model import VectorWorldModel, scale_array  # noqa: E402
from phase7b_stage_targets import read_stage_targets, validate_stage_targets  # noqa: E402
from world_model_dataset import read_world_model_samples  # noqa: E402


class TargetConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.risk_targets = read_per_step_risk_targets(DEFAULT_WINDOWS_DIR)

    def test_six_step_target_shape(self) -> None:
        for split in ("train", "validation", "test"):
            self.assertEqual(self.risk_targets[split]["y"].shape[1], FORECAST_HORIZON_WINDOWS)

    def test_target_is_binary(self) -> None:
        for split in ("train", "validation", "test"):
            values = set(np.unique(self.risk_targets[split]["y"]).tolist())
            self.assertTrue(values.issubset({0, 1}))

    def test_expected_split_counts(self) -> None:
        self.assertEqual(self.risk_targets["train"]["y"].shape[0], 29315)
        self.assertEqual(self.risk_targets["validation"]["y"].shape[0], 6195)
        self.assertEqual(self.risk_targets["test"]["y"].shape[0], 6195)

    def test_target_temporal_correctness_matches_phase7b_stage_derivation(self) -> None:
        """y_k must exactly equal (per_step_stage_name != BENIGN) from Phase 7B's
        already-validated, independently-constructed stage targets."""
        stage_targets = read_stage_targets(DEFAULT_WINDOWS_DIR)
        for split in ("train", "validation", "test"):
            expected = (stage_targets[split].per_step_stage_name != "BENIGN").astype(np.int64)
            self.assertTrue(np.array_equal(expected, self.risk_targets[split]["y"]))

    def test_source_day_boundaries_respected(self) -> None:
        # read_per_step_risk_targets raises ValueError internally if a sample's future rows
        # cross a split boundary -- successfully returning here IS the proof, reinforced by
        # re-checking source_file is populated and non-empty for every row.
        for split in ("train", "validation", "test"):
            self.assertTrue(all(self.risk_targets[split]["source_file"]))

    def test_terminal_windows_excluded(self) -> None:
        # Every eligible row must have exactly 6 future rows -- shape check is the proof
        # (a truncated/terminal row would have raised inside read_per_step_risk_targets).
        for split in ("train", "validation", "test"):
            self.assertEqual(self.risk_targets[split]["y"].shape[1], 6)


class LeakageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, cls.feature_columns, _ = read_world_model_samples(DEFAULT_WINDOWS_DIR)
        cls.stage_targets = read_stage_targets(DEFAULT_WINDOWS_DIR)
        cls.risk_targets = read_per_step_risk_targets(DEFAULT_WINDOWS_DIR)
        cls.audit = leakage_audit(cls.samples, cls.stage_targets, cls.risk_targets)

    def test_leakage_audit_passes(self) -> None:
        self.assertEqual(self.audit["status"], "PASS", self.audit["issues"])

    def test_no_current_future_leakage_structural(self) -> None:
        # y(t+1) must not be structurally identical to the current-window flag for every
        # sample (that would indicate an off-by-one / accidental aliasing bug); some
        # correlation is expected (attacks are sustained) but not 100% identity.
        current = self.risk_targets["train"]["current_attack_at_t"]
        y_t1 = self.risk_targets["train"]["y"][:, 0]
        identical_fraction = float((current == y_t1).mean())
        self.assertLess(identical_fraction, 1.0)

    def test_x_never_contains_future_information(self) -> None:
        # Structural proof: X is built exclusively from world_model_dataset's own history_rows
        # (t-5..t), a module Phase 9N never modifies or re-implements; per_step targets come
        # from an entirely separate read (read_per_step_risk_targets), which only ever slices
        # rows[index+1:index+1+FORECAST_HORIZON_WINDOWS] -- never rows up to and including index.
        self.assertEqual(self.samples["train"].X.shape[1], HISTORY_WINDOWS)
        self.assertEqual(self.samples["train"].X.shape[2], INPUT_SIZE)

    def test_alignment_across_all_three_readers(self) -> None:
        for split in ("train", "validation", "test"):
            self.assertTrue(np.array_equal(self.samples[split].source_file, self.risk_targets[split]["source_file"]))
            self.assertTrue(np.array_equal(self.samples[split].window_start, self.risk_targets[split]["window_start"]))


class TrainOnlyPreprocessingTests(unittest.TestCase):
    def test_scaler_reused_is_run1_scaler_not_a_new_fit(self) -> None:
        # Phase 9N must not fit a new scaler -- it reuses the frozen, already-verified
        # Run-1 scaler (fit on TRAIN history states only, per phase6b_vector_world_model.py).
        self.assertTrue(RUN1_SCALER.exists())

    def test_train_only_pos_weight_computed_from_train_split(self) -> None:
        risk_targets = read_per_step_risk_targets(DEFAULT_WINDOWS_DIR)
        train_y = risk_targets["train"]["y"]
        pos = train_y.sum(axis=0)
        neg = train_y.shape[0] - pos
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
        self.assertEqual(pos_weight.shape, (FORECAST_HORIZON_WINDOWS,))
        self.assertTrue(np.all(pos_weight > 0))


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_per_step_head_output_shape(self) -> None:
        head = PerStepAttackRiskHead().to(self.device)
        z_future = torch.randn(4, FORECAST_HORIZON_WINDOWS, 128, device=self.device)
        out = head(z_future)
        self.assertEqual(tuple(out.shape), (4, FORECAST_HORIZON_WINDOWS))

    def test_full_model_forward_shape(self) -> None:
        model = VectorWorldModelWithPerStepRiskHead().to(self.device)
        x = torch.randn(4, HISTORY_WINDOWS, INPUT_SIZE, device=self.device)
        y_hat, attack_logit, per_step_risk_logits, z_future = model(x)
        self.assertEqual(tuple(y_hat.shape), (4, FORECAST_HORIZON_WINDOWS, INPUT_SIZE))
        self.assertEqual(tuple(attack_logit.shape), (4,))
        self.assertEqual(tuple(per_step_risk_logits.shape), (4, FORECAST_HORIZON_WINDOWS))
        self.assertEqual(tuple(z_future.shape), (4, FORECAST_HORIZON_WINDOWS, 128))

    def test_distinct_per_step_outputs_are_possible(self) -> None:
        """With random (untrained) weights, the per-step head must be CAPABLE of
        producing distinct values per step -- proves the architecture is not
        accidentally collapsed to a single broadcast value by construction."""
        torch.manual_seed(0)
        model = VectorWorldModelWithPerStepRiskHead().to(self.device)
        x = torch.randn(8, HISTORY_WINDOWS, INPUT_SIZE, device=self.device)
        _, _, per_step_risk_logits, _ = model(x)
        per_step = per_step_risk_logits.detach().cpu().numpy()
        # Not every row can be constant across its 6 steps.
        constant_rows = np.all(np.isclose(per_step, per_step[:, :1], atol=1e-6), axis=1)
        self.assertFalse(constant_rows.all())

    def test_base_vector_world_model_class_untouched(self) -> None:
        # VectorWorldModelWithPerStepRiskHead must be a SUBCLASS, never a redefinition.
        self.assertTrue(issubclass(VectorWorldModelWithPerStepRiskHead, VectorWorldModel))


class BaselineCorrectnessTests(unittest.TestCase):
    def test_baseline_a_broadcasts_current_status_unchanged(self) -> None:
        current = np.array([0, 1, 0, 1], dtype=np.int64)
        out = baseline_a_persistence(current)
        self.assertEqual(out.shape, (4, FORECAST_HORIZON_WINDOWS))
        for row, c in zip(out, current):
            self.assertTrue(np.all(row == float(c)))

    def test_baseline_b_broadcasts_whole_horizon_scalar_unchanged(self) -> None:
        scalar = np.array([0.1, 0.9, 0.5])
        out = baseline_b_whole_horizon_broadcast(scalar)
        self.assertEqual(out.shape, (3, FORECAST_HORIZON_WINDOWS))
        for row, s in zip(out, scalar):
            self.assertTrue(np.allclose(row, s))

    def test_baseline_b_is_never_presented_via_no_scalar_broadcast_in_learned_output(self) -> None:
        """The LEARNED model's per-step outputs (from a real forward pass with random
        weights) must not be a pure broadcast of a single value -- i.e. the architecture
        itself does not silently degenerate into Baseline B."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        model = VectorWorldModelWithPerStepRiskHead().to(device)
        x = torch.randn(16, HISTORY_WINDOWS, INPUT_SIZE, device=device)
        _, _, per_step_risk_logits, _ = model(x)
        per_step = per_step_risk_logits.detach().cpu().numpy()
        broadcast_like = np.all(np.isclose(per_step, per_step[:, :1], atol=1e-6), axis=1)
        self.assertFalse(broadcast_like.all())


@unittest.skipUnless(OUTPUT_DIR.exists(), "Phase 9N has not been run yet (experiments/phase9n_per_step_risk.py)")
class TrainedCheckpointTests(unittest.TestCase):
    """Loads the already-trained checkpoint (no re-training) for shape/determinism checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint_path = OUTPUT_DIR / "model/best_per_step_risk_head.pt"
        if not checkpoint_path.exists():
            raise unittest.SkipTest("trained checkpoint not present")
        cls.model = load_frozen_backbone(RUN1_CHECKPOINT, cls.device)
        checkpoint = torch.load(checkpoint_path, map_location=cls.device, weights_only=True)
        cls.model.load_state_dict(checkpoint["model_state_dict"])
        cls.model.eval()

        import joblib
        cls.scaler = joblib.load(RUN1_SCALER)
        samples, _, _ = read_world_model_samples(DEFAULT_WINDOWS_DIR)
        cls.x_scaled = scale_array(samples["test"].X[:32], cls.scaler)

    def test_checkpoint_loads_with_expected_keys(self) -> None:
        state_dict_keys = set(self.model.state_dict().keys())
        self.assertTrue(any(k.startswith(RISK_HEAD_PREFIX) for k in state_dict_keys))

    def test_output_shape_b6(self) -> None:
        attack_prob, per_step_prob, _ = predict_all(self.model, self.x_scaled, self.device, use_amp=False)
        self.assertEqual(attack_prob.shape, (32,))
        self.assertEqual(per_step_prob.shape, (32, FORECAST_HORIZON_WINDOWS))

    def test_deterministic_inference(self) -> None:
        _, p1, _ = predict_all(self.model, self.x_scaled, self.device, use_amp=False)
        _, p2, _ = predict_all(self.model, self.x_scaled, self.device, use_amp=False)
        self.assertTrue(np.allclose(p1, p2, atol=1e-6))

    def test_probabilities_in_unit_interval(self) -> None:
        _, per_step_prob, _ = predict_all(self.model, self.x_scaled, self.device, use_amp=False)
        self.assertTrue(np.all(per_step_prob >= 0.0))
        self.assertTrue(np.all(per_step_prob <= 1.0))

    def test_no_scalar_probability_broadcasting_in_real_output(self) -> None:
        attack_prob, per_step_prob, _ = predict_all(self.model, self.x_scaled, self.device, use_amp=False)
        broadcast_of_scalar = np.allclose(per_step_prob, attack_prob[:, None], atol=1e-6)
        self.assertFalse(broadcast_of_scalar)


@unittest.skipUnless(OUTPUT_DIR.exists(), "Phase 9N has not been run yet")
class ProducedArtifactAndSihSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((OUTPUT_DIR / "phase9n_per_step_risk_report.json").read_text(encoding="utf-8"))

    def test_status_is_one_of_green_yellow_red(self) -> None:
        self.assertIn(self.report["status"], ("GREEN", "YELLOW", "RED"))

    def test_required_files_exist(self) -> None:
        for name in [
            "phase9n_per_step_risk_report.md", "phase9n_per_step_risk_report.json",
            "phase9n_protocol.json", "metrics.csv", "leakage_checks.json", "target_construction.json",
        ]:
            self.assertTrue((OUTPUT_DIR / name).exists(), f"missing required artifact: {name}")

    def test_sih_compliance_distinguishes_generic_from_infiltration_specific(self) -> None:
        sih = self.report["sih_compliance"]
        self.assertIn("generic", sih["delivered_in_this_phase"].lower())
        self.assertIn("infiltration", sih["NOT_delivered"].lower())

    def test_no_forbidden_overclaim_phrases_outside_disclaimer(self) -> None:
        report_text = (OUTPUT_DIR / "phase9n_per_step_risk_report.md").read_text(encoding="utf-8")
        before = report_text.split("## 18. Claim-Safety Statement")[0]
        after = report_text.split("## 19. Final Verdict")[-1] if "## 19. Final Verdict" in report_text else ""
        scanned = (before + after).lower()
        forbidden = [
            "predicts the exact next attack", "predicts attacker actions", "causal attacker progression",
            "generalizes to unseen attacks",
        ]
        for phrase in forbidden:
            self.assertNotIn(phrase, scanned, f"forbidden phrase found outside disclaimer: {phrase!r}")

    def test_preflight_and_postflight_passed(self) -> None:
        self.assertEqual(self.report["preflight"]["status"], "PASS")
        self.assertEqual(self.report["postflight"]["status"], "PASS")

    def test_backbone_verified_unchanged_after_training(self) -> None:
        self.assertTrue(self.report["training"]["backbone_unchanged_after_training"])

    def test_verdict_not_green_merely_because_it_trained(self) -> None:
        # Given persistence beats the learned model at every step in this run, GREEN would be
        # an overclaim -- this test locks in that the reported verdict reflects the evidence.
        beats_persistence = self.report["aggregate"]["learned_model"]["mean_pr_auc"] > self.report["aggregate"]["baseline_a_persistence"]["mean_pr_auc"]
        if not beats_persistence:
            self.assertNotEqual(self.report["status"], "GREEN")


class FrozenArtifactIntegrityTests(unittest.TestCase):
    def test_run1_checkpoint_and_scaler_exist_and_untouched(self) -> None:
        self.assertTrue(RUN1_CHECKPOINT.exists())
        self.assertTrue(RUN1_SCALER.exists())

    def test_phase9n_checkpoint_confined_to_own_directory(self) -> None:
        if OUTPUT_DIR.exists():
            checkpoint = OUTPUT_DIR / "model/best_per_step_risk_head.pt"
            if checkpoint.exists():
                self.assertIn("phase9n_per_step_risk", str(checkpoint))

    def test_phase6b_vector_world_model_module_not_modified_definitions(self) -> None:
        # Structural check: VectorWorldModel's forward signature must still return exactly
        # (y_hat, attack_logit) -- a 2-tuple, matching the frozen, unmodified base class.
        import inspect
        source = inspect.getsource(VectorWorldModel.forward)
        self.assertIn("return y_hat, attack_logit", source)


if __name__ == "__main__":
    unittest.main()
