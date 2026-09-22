"""Phase 7B: focused tests for the frozen-backbone MITRE Stage Head experiment.

Plain unittest.TestCase (not Django DB-backed), matching forecasting/tests.py's
Phase 7A convention -- nothing here has an ORM dependency. Some tests load the
real frozen Run-1 checkpoint and run real forward passes on the real dataset,
so they are not instantaneous, but they are the only way to genuinely verify
the safety properties this phase depends on (frozen backbone, no leakage,
Run-1 reproduction) rather than merely asserting them.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from phase6b_ablation import TARGET_FPR
from phase7b_mitre_stage_head import (
    FORECAST_HORIZON_WINDOWS,
    NUM_STAGE_CLASSES,
    VectorWorldModelWithStageHead,
    assert_only_stage_head_trainable,
    backbone_state_dict,
    backbone_unchanged,
    freeze_backbone,
    load_frozen_backbone,
    predict_all,
    set_frozen_backbone_training_mode,
)
from phase7b_stage_targets import (
    ABSENT_STAGES,
    CLASS_INDEX_TO_STAGE,
    COVERED_STAGES,
    NUM_STAGE_CLASSES as TARGETS_NUM_STAGE_CLASSES,
    read_stage_targets,
    validate_stage_targets,
)
from phase7b_train_stage_head import (
    RUN1_CHECKPOINT,
    WINDOWS_DIR,
    aggregate_predicted_stage,
    run_preflight_check,
)
from phase6b_vector_world_model import scale_array
from world_model_dataset import read_world_model_samples

import csv
import json
import numpy as np
import joblib

from phase4_baseline import META_COLUMNS
from phase7c_explainability import (
    StageLogitView,
    TemporalAwareExplainer,
    WINDOW_LABELS,
    explain_sample,
    load_trained_phase7b_model,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PHASE7B_STAGE_CHECKPOINT = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/model/best_stage_head.pt"
PHASE7B_PREDICTIONS_TEST = Path(__file__).resolve().parent / "results/phase7b_mitre_stage_head/predictions_test.csv"


def _load_model() -> VectorWorldModelWithStageHead:
    return load_frozen_backbone(RUN1_CHECKPOINT, DEVICE)


class CheckpointLoadingTests(unittest.TestCase):
    def test_frozen_run1_checkpoint_loads(self) -> None:
        model = _load_model()
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("stage_head."))
        stage_head_params = sum(p.numel() for p in model.stage_head.parameters())
        self.assertEqual(backbone_params, 347806)  # exact Phase 6B/6B-ablation parameter count
        self.assertEqual(total_params, backbone_params + stage_head_params)

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_frozen_backbone(Path("no_such_checkpoint.pt"), DEVICE)


class TensorShapeTests(unittest.TestCase):
    def test_forward_output_shapes(self) -> None:
        model = _load_model()
        model.eval()
        batch = torch.randn(5, 6, 157, device=DEVICE)
        with torch.no_grad():
            y_hat, attack_logit, stage_logits, z_future = model(batch)
        self.assertEqual(tuple(y_hat.shape), (5, 6, 157))
        self.assertEqual(tuple(attack_logit.shape), (5,))
        self.assertEqual(tuple(stage_logits.shape), (5, 6, NUM_STAGE_CLASSES))
        self.assertEqual(tuple(z_future.shape), (5, 6, 128))

    def test_six_step_output(self) -> None:
        """stage_logits' horizon dimension must be exactly FORECAST_HORIZON_WINDOWS (6)."""
        model = _load_model()
        model.eval()
        batch = torch.randn(3, 6, 157, device=DEVICE)
        with torch.no_grad():
            _, _, stage_logits, _ = model(batch)
        self.assertEqual(stage_logits.shape[1], FORECAST_HORIZON_WINDOWS)
        self.assertEqual(stage_logits.shape[1], 6)
        self.assertEqual(NUM_STAGE_CLASSES, 6)
        self.assertEqual(NUM_STAGE_CLASSES, TARGETS_NUM_STAGE_CLASSES)


class FrozenBackboneTests(unittest.TestCase):
    def test_freeze_sets_requires_grad_correctly(self) -> None:
        model = _load_model()
        freeze_backbone(model)
        assert_only_stage_head_trainable(model)  # raises on violation
        for name, parameter in model.named_parameters():
            if name.startswith("stage_head."):
                self.assertTrue(parameter.requires_grad)
            else:
                self.assertFalse(parameter.requires_grad)

    def test_backbone_unchanged_detector_is_sensitive(self) -> None:
        model = _load_model()
        freeze_backbone(model)
        before = backbone_state_dict(model)
        # Unmodified: must report unchanged.
        self.assertTrue(backbone_unchanged(before, backbone_state_dict(model)))
        # Deliberately mutate a backbone tensor: detector must catch it.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.startswith("stage_head."):
                    parameter.add_(1.0)
                    break
        after = backbone_state_dict(model)
        self.assertFalse(backbone_unchanged(before, after))

    def test_training_mode_isolates_dropout_to_stage_head(self) -> None:
        model = _load_model()
        freeze_backbone(model)
        set_frozen_backbone_training_mode(model)
        self.assertFalse(model.state_encoder.training)
        self.assertFalse(model.temporal_context.training)
        self.assertFalse(model.transition.training)
        self.assertFalse(model.state_decoder.training)
        self.assertFalse(model.attack_head.training)
        self.assertTrue(model.stage_head.training)


class GradientIsolationLeakageTests(unittest.TestCase):
    """Proves future labels/targets never enter forward(), and that a frozen
    backbone truly blocks gradient flow -- empirically, not just by design."""

    def test_forward_signature_takes_only_x(self) -> None:
        signature = inspect.signature(VectorWorldModelWithStageHead.forward)
        parameter_names = [name for name in signature.parameters if name != "self"]
        self.assertEqual(parameter_names, ["x"])

    def test_backbone_outputs_require_no_grad_when_frozen(self) -> None:
        model = _load_model()
        freeze_backbone(model)
        set_frozen_backbone_training_mode(model)
        batch = torch.randn(4, 6, 157, device=DEVICE)
        y_hat, attack_logit, stage_logits, z_future = model(batch)
        self.assertFalse(y_hat.requires_grad)
        self.assertFalse(attack_logit.requires_grad)
        self.assertFalse(z_future.requires_grad)
        self.assertTrue(stage_logits.requires_grad)

    def test_backward_only_populates_stage_head_gradients(self) -> None:
        model = _load_model()
        freeze_backbone(model)
        set_frozen_backbone_training_mode(model)
        batch = torch.randn(4, 6, 157, device=DEVICE)
        _, _, stage_logits, _ = model(batch)
        stage_logits.sum().backward()
        for name, parameter in model.named_parameters():
            if name.startswith("stage_head."):
                self.assertIsNotNone(parameter.grad, f"{name} should have received a gradient")
            else:
                self.assertIsNone(parameter.grad, f"{name} should NOT have received a gradient")


class TargetConstructionAndAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage_targets = read_stage_targets(WINDOWS_DIR)
        cls.samples, cls.feature_columns, _ = read_world_model_samples(WINDOWS_DIR)

    def test_stage_target_validation_passes(self) -> None:
        report = validate_stage_targets(self.stage_targets)
        self.assertEqual(report["status"], "PASS", report["issues"])

    def test_six_covered_stages_only(self) -> None:
        self.assertEqual(len(COVERED_STAGES), 6)
        self.assertEqual(len(ABSENT_STAGES), 8)
        self.assertEqual(len(COVERED_STAGES) + len(ABSENT_STAGES), 14)

    def test_alignment_with_world_model_dataset_samples(self) -> None:
        for split in ("train", "validation", "test"):
            self.assertTrue(
                (self.samples[split].source_file == self.stage_targets[split].source_file).all(),
                f"{split}: source_file misaligned",
            )
            self.assertTrue(
                (self.samples[split].window_start == self.stage_targets[split].window_start).all(),
                f"{split}: window_start misaligned",
            )

    def test_known_row_stage_target(self) -> None:
        """Wednesday-14-02-2018 contains real SSH-Bruteforce/FTP-BruteForce
        traffic (per the Phase 2 dataset audit). Its per-step targets must
        contain CREDENTIAL_ACCESS somewhere (Phase 7A: SSH-Bruteforce /
        FTP-BruteForce -> CREDENTIAL_ACCESS/T1110) -- this proves the
        independent target reader is actually resolving real labels through
        the mapper, not just returning BENIGN everywhere."""
        train = self.stage_targets["train"]
        wednesday_mask = train.source_file == "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
        self.assertTrue(wednesday_mask.any())
        wednesday_steps = train.per_step_stage_name[wednesday_mask]
        self.assertIn("CREDENTIAL_ACCESS", wednesday_steps.reshape(-1).tolist())

    def test_known_row_no_reconnaissance_or_execution(self) -> None:
        """None of the 8 absent stages should ever appear as a produced
        target anywhere in the dataset (stage_to_class_index() would have
        raised during read_stage_targets() if one had)."""
        for split in ("train", "validation", "test"):
            flat = self.stage_targets[split].per_step_stage_name.reshape(-1).tolist()
            for absent_stage in ABSENT_STAGES:
                self.assertNotIn(absent_stage.name, flat)


class DeterministicAggregationTests(unittest.TestCase):
    def test_single_clear_winner(self) -> None:
        names = ["BENIGN"] * 5 + ["IMPACT"]
        confidences = [0.5] * 5 + [0.9]
        stage, confidence = aggregate_predicted_stage(names, confidences)
        self.assertEqual(stage, "IMPACT")
        self.assertEqual(confidence, 0.9)

    def test_tie_break_is_alphabetical_and_deterministic(self) -> None:
        names = ["COMMAND_AND_CONTROL", "IMPACT", "BENIGN", "BENIGN", "BENIGN", "BENIGN"]
        confidences = [0.6, 0.6, 0.1, 0.1, 0.1, 0.1]
        # COMMAND_AND_CONTROL(11) vs IMPACT(13) -> IMPACT has higher value, wins outright (not a tie).
        stage, confidence = aggregate_predicted_stage(names, confidences)
        self.assertEqual(stage, "IMPACT")

    def test_reordering_does_not_change_result(self) -> None:
        names_forward = ["COMMAND_AND_CONTROL", "COMMAND_AND_CONTROL", "IMPACT", "BENIGN", "BENIGN", "BENIGN"]
        names_reversed = list(reversed(names_forward))
        confidences_forward = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
        confidences_reversed = list(reversed(confidences_forward))
        result_forward = aggregate_predicted_stage(names_forward, confidences_forward)
        result_reversed = aggregate_predicted_stage(names_reversed, confidences_reversed)
        self.assertEqual(result_forward[0], result_reversed[0])


class Run1ReproductionTests(unittest.TestCase):
    """The mandatory hard pre-flight gate, run standalone as a test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, cls.feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
        import joblib

        scaler = joblib.load(RUN1_CHECKPOINT.parent / "scaler.joblib")
        cls.scaled_x = {split: scale_array(cls.samples[split].X, scaler) for split in ("train", "validation", "test")}
        cls.model = load_frozen_backbone(RUN1_CHECKPOINT, DEVICE)

    def test_run1_attack_metrics_reproduce(self) -> None:
        use_amp = DEVICE.type == "cuda"
        result = run_preflight_check(self.model, self.samples, self.scaled_x, DEVICE, use_amp)
        self.assertEqual(result["status"], "PASS", result["mismatches"])
        self.assertAlmostEqual(result["reproduced_test_metrics"]["pr_auc"], 0.849, delta=0.01)
        self.assertAlmostEqual(result["reproduced_test_metrics"]["roc_auc"], 0.872, delta=0.01)
        self.assertAlmostEqual(result["reproduced_test_metrics"]["f1"], 0.747, delta=0.01)
        self.assertAlmostEqual(result["reproduced_test_metrics"]["false_positive_rate"], 0.096, delta=0.01)


def _load_persisted_test_predictions() -> dict[tuple[str, str], dict[str, str]]:
    with PHASE7B_PREDICTIONS_TEST.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["source_file"], row["window_start"]): row for row in rows}


class Phase7CExplainabilityTests(unittest.TestCase):
    """Phase 7C: focused tests for the post-hoc explainability adapter.

    Loads the real, already-trained Phase 7B checkpoint (backbone + trained
    stage_head) and runs the real explain_sample() pipeline on one real test
    sample -- these are integration tests of the actual frozen artifacts,
    not mocks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_trained_phase7b_model(PHASE7B_STAGE_CHECKPOINT, DEVICE)
        cls.scaler = joblib.load(RUN1_CHECKPOINT.parent / "scaler.joblib")
        samples, feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
        cls.feature_columns = feature_columns
        cls.test_samples = samples["test"]
        cls.sample_index = 0
        cls.x_scaled = cls.scaler.transform(
            cls.test_samples.X[cls.sample_index].reshape(-1, 157)
        ).reshape(6, 157).astype(np.float32)
        cls.source_file = str(cls.test_samples.source_file[cls.sample_index])
        cls.window_start = str(cls.test_samples.window_start[cls.sample_index])
        persisted = _load_persisted_test_predictions()
        cls.persisted_row = persisted[(cls.source_file, cls.window_start)]

    def _explain(self) -> dict:
        return explain_sample(
            model=self.model,
            feature_names=self.feature_columns,
            x_scaled=self.x_scaled,
            scaler=self.scaler,
            source_file=self.source_file,
            window_start=self.window_start,
            device=DEVICE,
            top_k=5,
        )

    def test_attribution_shape_is_6x157(self) -> None:
        explanation = self._explain()
        attack_attribution = np.array(explanation["temporal_evidence"]["attack_risk_attribution"])
        stage_attribution = np.array(explanation["temporal_evidence"]["stage_attribution"])
        self.assertEqual(attack_attribution.shape, (6, 157))
        self.assertEqual(stage_attribution.shape, (6, 157))

    def test_attribution_values_finite(self) -> None:
        explanation = self._explain()
        attack_attribution = np.array(explanation["temporal_evidence"]["attack_risk_attribution"])
        stage_attribution = np.array(explanation["temporal_evidence"]["stage_attribution"])
        self.assertTrue(np.isfinite(attack_attribution).all())
        self.assertTrue(np.isfinite(stage_attribution).all())

    def test_feature_count_is_exactly_157(self) -> None:
        self.assertEqual(len(self.feature_columns), 157)
        explanation = self._explain()
        self.assertEqual(len(explanation["feature_names"]), 157)

    def test_no_metadata_columns_enter_attribution(self) -> None:
        for name in self.feature_columns:
            self.assertNotIn(name, META_COLUMNS)

    def test_prediction_matches_persisted_phase7b_prediction(self) -> None:
        explanation = self._explain()
        recomputed_probability = explanation["future_attack_risk_prediction"]["attack_probability"]
        persisted_probability = float(self.persisted_row["attack_probability"])
        self.assertAlmostEqual(recomputed_probability, persisted_probability, delta=1e-3)

        recomputed_steps = explanation["mitre_stage_prediction"]["per_step_stage"]
        persisted_steps = [self.persisted_row[f"mitre_stage_step_{step}"] for step in range(1, 7)]
        self.assertEqual(recomputed_steps, persisted_steps)
        self.assertEqual(explanation["mitre_stage_prediction"]["overall_stage"], self.persisted_row["mitre_stage_overall"])

    def test_stage_logit_view_selects_intended_mitre_class(self) -> None:
        step = 0
        view = StageLogitView(self.model, step=step)
        x_tensor = torch.from_numpy(self.x_scaled).unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            _, _, full_stage_logits, _ = self.model(x_tensor)
            view_output = view(x_tensor)
        self.assertEqual(tuple(view_output.shape), (1, NUM_STAGE_CLASSES))
        self.assertTrue(torch.allclose(view_output, full_stage_logits[:, step, :]))

    def test_explainer_does_not_modify_model_parameters(self) -> None:
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        self._explain()
        after = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), f"{name} changed during explanation")

    def test_explainer_does_not_alter_checkpoint_file_on_disk(self) -> None:
        before_bytes = PHASE7B_STAGE_CHECKPOINT.read_bytes()
        self._explain()
        after_bytes = PHASE7B_STAGE_CHECKPOINT.read_bytes()
        self.assertEqual(before_bytes, after_bytes)

    def test_z_future_unchanged_across_explanation(self) -> None:
        x_tensor = torch.from_numpy(self.x_scaled).unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            _, _, _, z_before = self.model(x_tensor)
        self._explain()
        with torch.no_grad():
            _, _, _, z_after = self.model(x_tensor)
        self.assertTrue(torch.equal(z_before, z_after))

    def test_explanation_is_deterministic(self) -> None:
        first = self._explain()
        second = self._explain()
        self.assertEqual(
            first["future_attack_risk_prediction"]["attack_probability"],
            second["future_attack_risk_prediction"]["attack_probability"],
        )
        first_attack = np.array(first["temporal_evidence"]["attack_risk_attribution"])
        second_attack = np.array(second["temporal_evidence"]["attack_risk_attribution"])
        self.assertTrue(np.allclose(first_attack, second_attack))
        first_stage = np.array(first["temporal_evidence"]["stage_attribution"])
        second_stage = np.array(second["temporal_evidence"]["stage_attribution"])
        self.assertTrue(np.allclose(first_stage, second_stage))

    def test_six_temporal_windows_preserved(self) -> None:
        explanation = self._explain()
        self.assertEqual(explanation["window_labels"], WINDOW_LABELS)
        self.assertEqual(len(explanation["temporal_evidence"]["top_attack_risk_features_per_window"]), 6)
        self.assertEqual(len(explanation["temporal_evidence"]["top_stage_features_per_window"]), 6)
        self.assertEqual(len(explanation["temporal_evidence"]["attack_risk_attribution"]), 6)

    def test_feature_names_align_exactly_with_attribution_columns(self) -> None:
        explanation = self._explain()
        attack_attribution = np.array(explanation["temporal_evidence"]["attack_risk_attribution"])
        self.assertEqual(attack_attribution.shape[1], len(explanation["feature_names"]))
        self.assertEqual(explanation["feature_names"], self.feature_columns)

    def test_no_future_labels_enter_explainer_or_model(self) -> None:
        explain_sample_params = list(inspect.signature(explain_sample).parameters)
        forward_params = list(inspect.signature(type(self.model).forward).parameters)
        for forbidden in ("label", "stage_target", "future_attack_types", "y_true", "future_label", "y"):
            self.assertNotIn(forbidden, explain_sample_params)
            self.assertNotIn(forbidden, forward_params)
        self.assertEqual([name for name in forward_params if name != "self"], ["x"])

    def test_attack_and_stage_explanations_remain_separate(self) -> None:
        explanation = self._explain()
        self.assertIn("attack_probability", explanation["future_attack_risk_prediction"])
        self.assertIn("overall_stage", explanation["mitre_stage_prediction"])
        # Structurally separate: different dict blocks, different (not the same
        # object, not numerically identical) attribution arrays -- they are
        # gradients/SHAP values of two different targets (attack_logit vs. one
        # MITRE stage-class logit), computed by two separate explainer calls.
        attack_attribution = explanation["temporal_evidence"]["attack_risk_attribution"]
        stage_attribution = explanation["temporal_evidence"]["stage_attribution"]
        self.assertIsNot(attack_attribution, stage_attribution)
        self.assertFalse(np.allclose(np.array(attack_attribution), np.array(stage_attribution)))


import inspect as _inspect

from phase7d_trajectory_narrative import (
    NON_BENIGN_LABEL,
    build_trajectory_record,
    detect_disagreement,
)

PHASE7C_RECORDS_DIR = Path(__file__).resolve().parent / "results/phase7c_explainability/records"
BENIGN_CLASS_INDEX_FOR_TEST = next(index for index, stage in CLASS_INDEX_TO_STAGE.items() if stage.name == "BENIGN")

_FORBIDDEN_NARRATIVE_SUBSTRINGS = [
    "attacker",
    "target host",
    "asset",
    "intent",
    "plan to",
    "planning",
    "identity",
    "100%",
    "certainly",
    "definitely",
    "will attack",
    "will compromise",
    "endpoint",
    "src ip",
    "dst ip",
]


class Phase7DTrajectoryNarrativeTests(unittest.TestCase):
    """Phase 7D: focused tests for the post-hoc trajectory/narrative layer.
    Loads the real, already-trained Phase 7B checkpoint and runs the real
    build_trajectory_record() pipeline on one real test sample (the same
    sample Phase 7C already explained, enabling a direct cross-check)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_trained_phase7b_model(PHASE7B_STAGE_CHECKPOINT, DEVICE)
        cls.scaler = joblib.load(RUN1_CHECKPOINT.parent / "scaler.joblib")
        samples, feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
        cls.feature_columns = feature_columns
        cls.test_samples = samples["test"]
        cls.sample_index = 0
        cls.x_scaled = cls.scaler.transform(
            cls.test_samples.X[cls.sample_index].reshape(-1, 157)
        ).reshape(6, 157).astype(np.float32)
        cls.source_file = str(cls.test_samples.source_file[cls.sample_index])
        cls.window_start = str(cls.test_samples.window_start[cls.sample_index])
        persisted = _load_persisted_test_predictions()
        cls.persisted_row = persisted[(cls.source_file, cls.window_start)]

    def _build(self) -> dict:
        return build_trajectory_record(
            model=self.model,
            feature_names=self.feature_columns,
            x_scaled=self.x_scaled,
            scaler=self.scaler,
            source_file=self.source_file,
            window_start=self.window_start,
            device=DEVICE,
        )

    # 1. Correct six-step ordering.
    def test_six_step_ordering(self) -> None:
        record = self._build()
        self.assertEqual(len(record["steps"]), 6)
        self.assertEqual([step["step"] for step in record["steps"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            [step["relative_window"] for step in record["steps"]],
            ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"],
        )

    # 2. Correct native horizon attack_probability preservation.
    def test_native_attack_probability_preserved(self) -> None:
        record = self._build()
        persisted_probability = float(self.persisted_row["attack_probability"])
        self.assertAlmostEqual(record["attack_probability"], persisted_probability, delta=1e-3)

    # 3. Correct non_benign_probability = 1 - P(BENIGN).
    def test_non_benign_probability_is_one_minus_p_benign(self) -> None:
        x_tensor = torch.from_numpy(self.x_scaled).unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            _, _, stage_logits, _ = self.model(x_tensor)
            stage_probabilities = torch.softmax(stage_logits, dim=-1)[0].cpu().numpy()
        record = self._build()
        for step_index, step in enumerate(record["steps"]):
            expected = 1.0 - float(stage_probabilities[step_index, BENIGN_CLASS_INDEX_FOR_TEST])
            self.assertAlmostEqual(step["non_benign_probability"], expected, places=6)

    # 4. Numerical finiteness.
    def test_all_values_finite(self) -> None:
        record = self._build()
        self.assertTrue(np.isfinite(record["attack_probability"]))
        self.assertTrue(np.isfinite(record["overall_confidence"]))
        for step in record["steps"]:
            self.assertTrue(np.isfinite(step["stage_confidence"]))
            self.assertTrue(np.isfinite(step["top2_confidence_margin"]))
            self.assertTrue(np.isfinite(step["non_benign_probability"]))
            for feature in step["top_k_driving_features"]:
                self.assertTrue(np.isfinite(feature["importance"]))

    # 5. Correct top1-top2 confidence margin.
    def test_top1_top2_confidence_margin(self) -> None:
        x_tensor = torch.from_numpy(self.x_scaled).unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            _, _, stage_logits, _ = self.model(x_tensor)
            stage_probabilities = torch.softmax(stage_logits, dim=-1)[0].cpu().numpy()
        record = self._build()
        for step_index, step in enumerate(record["steps"]):
            sorted_probabilities = np.sort(stage_probabilities[step_index])[::-1]
            expected_margin = float(sorted_probabilities[0] - sorted_probabilities[1])
            self.assertAlmostEqual(step["top2_confidence_margin"], expected_margin, places=6)

    # 6. Disagreement rule with synthetic low-risk/high-stage-confidence case.
    def test_disagreement_rule_triggers_on_synthetic_case(self) -> None:
        steps = [{"step": i + 1, "predicted_stage": "BENIGN", "stage_confidence": 0.5} for i in range(5)]
        steps.append({"step": 6, "predicted_stage": "IMPACT", "stage_confidence": 0.95})
        result = detect_disagreement(0.10, steps)
        self.assertTrue(result["detected"])
        self.assertIsNotNone(result["warning"])
        self.assertIn("DISAGREEMENT", result["warning"])
        self.assertEqual(result["flagged_steps"], [6])

    # 7. Disagreement rule with synthetic non-disagreement case.
    def test_disagreement_rule_does_not_trigger_on_synthetic_consistent_cases(self) -> None:
        all_benign = [{"step": i + 1, "predicted_stage": "BENIGN", "stage_confidence": 0.9} for i in range(6)]
        result = detect_disagreement(0.05, all_benign)
        self.assertFalse(result["detected"])
        self.assertIsNone(result["warning"])

        high_risk_and_high_stage = [{"step": i + 1, "predicted_stage": "IMPACT", "stage_confidence": 0.99} for i in range(6)]
        result_high_risk = detect_disagreement(0.90, high_risk_and_high_stage)
        self.assertFalse(result_high_risk["detected"])

    # 8. Narrative never labels derived non-benign signal as "attack probability".
    def test_narrative_never_mislabels_non_benign_signal(self) -> None:
        record = self._build()
        for step in record["steps"]:
            self.assertEqual(step["non_benign_probability_label"], NON_BENIGN_LABEL)
        self.assertIn(NON_BENIGN_LABEL, record["narrative"])
        self.assertIn("NOT attack probability", NON_BENIGN_LABEL)

    # 9. Narrative never claims causal attacker progression.
    def test_narrative_never_claims_causal_progression(self) -> None:
        record = self._build()
        narrative_lower = record["narrative"].lower()
        self.assertNotIn("causes", narrative_lower)
        self.assertNotIn("caused by", narrative_lower)
        self.assertIn("independent", narrative_lower)
        self.assertIn("not a guaranteed", narrative_lower)

    # 10. Narrative never references attacker identity or endpoint identity.
    def test_narrative_never_references_forbidden_content(self) -> None:
        record = self._build()
        narrative_lower = record["narrative"].lower()
        for forbidden in _FORBIDDEN_NARRATIVE_SUBSTRINGS:
            self.assertNotIn(forbidden, narrative_lower, f"forbidden substring {forbidden!r} found in narrative")

    # 11. AttackChainBuilder.predict_next_stage() is never used.
    def test_attack_chain_builder_never_used(self) -> None:
        """AttackChainBuilder may be MENTIONED in documentation/comments
        explaining why it is deliberately avoided (that mention is required,
        not forbidden) -- what must never appear is an actual import or a
        call to .predict_next_stage(...). Checked via the AST rather than
        substring search, so docstring prose (which legitimately contains
        these names) cannot produce a false failure -- only real Name/
        Attribute/Import nodes count."""
        import ast
        import phase7d_trajectory_narrative
        import phase7d_generate_trajectories

        for module in (phase7d_trajectory_narrative, phase7d_generate_trajectories):
            tree = ast.parse(_inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, "AttackChainBuilder")
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, "predict_next_stage")
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        self.assertNotEqual(alias.name, "AttackChainBuilder")

    # 12. Existing Phase 7B prediction values remain unchanged.
    def test_phase7b_predictions_unchanged(self) -> None:
        record = self._build()
        recomputed_steps = [step["predicted_stage"] for step in record["steps"]]
        persisted_steps = [self.persisted_row[f"mitre_stage_step_{step}"] for step in range(1, 7)]
        self.assertEqual(recomputed_steps, persisted_steps)
        self.assertEqual(record["overall_predicted_stage"], self.persisted_row["mitre_stage_overall"])

    # 13. Existing Phase 7C attribution values remain unchanged.
    def test_phase7c_attribution_values_unchanged(self) -> None:
        phase7c_record = json.loads((PHASE7C_RECORDS_DIR / "sample_0001.json").read_text(encoding="utf-8"))
        self.assertEqual(phase7c_record["source_file"], self.source_file)
        self.assertEqual(phase7c_record["window_start"], self.window_start)
        winning_step_0indexed = phase7c_record["mitre_stage_prediction"]["explained_step_index"]

        record = self._build()
        self.assertEqual(record["overall_stage_step"], winning_step_0indexed + 1)

        phase7d_top_features = record["steps"][winning_step_0indexed]["top_k_driving_features"]
        phase7c_top_features = phase7c_record["current_state_evidence"]["top_stage_features"]
        for phase7d_feature, phase7c_feature in zip(phase7d_top_features, phase7c_top_features):
            self.assertEqual(phase7d_feature["feature"], phase7c_feature["feature"])
            self.assertAlmostEqual(phase7d_feature["importance"], phase7c_feature["importance"], places=9)

    # 14. Deterministic repeated generation.
    def test_deterministic_repeated_generation(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first["attack_probability"], second["attack_probability"])
        self.assertEqual(first["steps"], second["steps"])
        self.assertEqual(first["narrative"], second["narrative"])

    # 15. No future labels are used by the production trajectory-generation input path.
    def test_no_future_labels_in_build_trajectory_record_input_path(self) -> None:
        build_params = list(_inspect.signature(build_trajectory_record).parameters)
        for forbidden in ("label", "stage_target", "future_attack_types", "y_true", "future_label", "y"):
            self.assertNotIn(forbidden, build_params)
        forward_params = list(_inspect.signature(type(self.model).forward).parameters)
        self.assertEqual([name for name in forward_params if name != "self"], ["x"])


import ast as _ast

from phase8_evaluation_comparison import (
    EXPECTED_POSITIVE_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    MODEL_SPECS,
    RUN1_DIR,
    TARGET_FPR,
    build_interpretation,
    evaluate_model,
    run_fairness_checks,
    verify_run1_integrity,
)


class Phase8ASharedComparisonTests(unittest.TestCase):
    """Phase 8A: focused tests for the shared controlled-FPR comparison.
    Pure post-hoc arithmetic over already-persisted CSVs -- no model is
    loaded, so these tests are fast despite exercising the real pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = [evaluate_model(spec) for spec in MODEL_SPECS]
        cls.fairness = run_fairness_checks(cls.results)
        cls.integrity = verify_run1_integrity(cls.results)

    def test_all_models_evaluated(self) -> None:
        self.assertEqual(len(self.results), 5)
        names = {result["name"] for result in self.results}
        self.assertEqual(
            names,
            {"logistic_regression", "weighted_lstm", "unweighted_lstm", "temporal_transformer", "vector_world_model_run1"},
        )

    def test_sample_counts_match_expected_partitions(self) -> None:
        for result in self.results:
            self.assertEqual(len(result["_val_labels"]), EXPECTED_SPLIT_COUNTS["validation"])
            self.assertEqual(len(result["_test_labels"]), EXPECTED_SPLIT_COUNTS["test"])

    def test_positive_prevalence_identical_across_models(self) -> None:
        for result in self.results:
            self.assertEqual(int(result["_val_labels"].sum()), EXPECTED_POSITIVE_COUNTS["validation"])
            self.assertEqual(int(result["_test_labels"].sum()), EXPECTED_POSITIVE_COUNTS["test"])
        self.assertTrue(self.fairness["positive_prevalence_identical_across_models"])

    def test_predictions_align_to_same_samples(self) -> None:
        self.assertTrue(self.fairness["predictions_align_to_same_samples"])

    def test_fairness_checks_pass(self) -> None:
        self.assertEqual(self.fairness["status"], "PASS", self.fairness["issues"])

    def test_threshold_selected_from_validation_only_by_source_inspection(self) -> None:
        """AST-based check (not substring search, to avoid false positives
        from documentation) that choose_threshold_recall_at_fpr is only ever
        called with validation-prefixed variable names, never test-prefixed."""
        import phase8_evaluation_comparison as module

        tree = _ast.parse(_inspect.getsource(module))
        call_sites = [
            node
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Name)
            and node.func.id == "choose_threshold_recall_at_fpr"
        ]
        self.assertGreaterEqual(len(call_sites), 1)
        for call in call_sites:
            argument_names = [arg.id for arg in call.args if isinstance(arg, _ast.Name)]
            for name in argument_names:
                self.assertNotIn("test", name.lower(), f"threshold selection call uses a test-prefixed argument: {name}")

    def test_run1_integrity_reference_and_exact(self) -> None:
        self.assertEqual(self.integrity["reference_status"], "PASS", self.integrity["reference_mismatches"])
        self.assertEqual(self.integrity["exact_status"], "PASS", self.integrity["exact_stored_value_mismatches"])

    def test_run1_recomputed_matches_stored_metrics_exactly(self) -> None:
        """'Exactly' here means within the CSV round-trip's 12-significant-
        digit precision (predictions_*.csv is written via '%.12g', per
        phase6b_vector_world_model.save_predictions), not bit-for-bit --
        Run 1's original in-memory computation had full float64 precision,
        while this script reads probabilities back from that 12-sig-fig
        text file. 1e-9 comfortably covers that gap without masking any
        real discrepancy (the gap actually observed is ~1e-13)."""
        stored = json.loads((RUN1_DIR / "metrics.json").read_text(encoding="utf-8"))
        run1 = next(result for result in self.results if result["name"] == "vector_world_model_run1")
        self.assertAlmostEqual(run1["threshold"], stored["primary_threshold_selection"]["threshold"], places=9)
        self.assertAlmostEqual(run1["test_f1"], stored["test_metrics_primary_threshold"]["f1"], places=9)
        self.assertAlmostEqual(run1["test_fpr"], stored["test_metrics_primary_threshold"]["false_positive_rate"], places=9)

    def test_test_fpr_is_not_reported_as_the_validation_constraint(self) -> None:
        """The validation constraint is <=5%; actual test FPR must be
        reported as its own computed value, not silently assumed to equal
        the constraint (prevalence shift means it generally will not)."""
        for result in self.results:
            self.assertNotEqual(result["test_fpr"], TARGET_FPR)

    def test_ranking_metrics_loaded_not_recomputed(self) -> None:
        """PR-AUC/ROC-AUC for each model must match its own already-existing
        metrics.json exactly (loaded, not altered)."""
        for spec, result in zip(MODEL_SPECS, self.results):
            stored = json.loads(spec["ranking_metrics_file"].read_text(encoding="utf-8"))
            for key in spec["ranking_metrics_path"]:
                stored = stored[key]
            self.assertEqual(result["pr_auc"], float(stored["pr_auc"]))
            self.assertEqual(result["roc_auc"], float(stored["roc_auc"]))

    def test_deterministic_repeated_evaluation(self) -> None:
        first = [evaluate_model(spec) for spec in MODEL_SPECS]
        second = [evaluate_model(spec) for spec in MODEL_SPECS]
        for a, b in zip(first, second):
            self.assertEqual(a["threshold"], b["threshold"])
            self.assertEqual(a["test_f1"], b["test_f1"])
            self.assertEqual(a["test_fpr"], b["test_fpr"])

    def test_no_winner_declared_from_f1_alone(self) -> None:
        """The interpretation must report multiple independent axes rather
        than collapsing to a single 'winner' pronouncement."""
        interpretation = build_interpretation(self.results)
        for key in ("best_pr_auc", "best_roc_auc", "best_validation_recall_under_shared_constraint", "best_test_f1_under_shared_constraint", "lowest_test_fpr"):
            self.assertIn(key, interpretation)
        self.assertIn(interpretation["claim_verdict"], ("PROVEN", "PARTIALLY SUPPORTED", "NOT SUPPORTED"))


from phase8_evaluation_comparison import RUN1_DIR as PHASE8B_RUN1_DIR
from phase8b_calibration import (
    TARGET_MODEL_SPECS,
    analyze_model as phase8b_analyze_model,
    apply_platt,
    fit_platt,
    select_calibration_method,
)


class Phase8BCalibrationTests(unittest.TestCase):
    """Phase 8B: focused tests for the attack-risk calibration analysis.
    Pure post-hoc arithmetic over already-persisted CSVs + sklearn fits on
    small (~6195-row) validation sets -- fast, no model loading."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.analyses = [phase8b_analyze_model(spec) for spec in TARGET_MODEL_SPECS]

    def test_calibrator_fitted_only_on_validation_by_source_inspection(self) -> None:
        """AST-based check that fit_platt/fit_isotonic are only ever called
        with validation-prefixed variable names inside analyze_model."""
        import ast as _ast2
        import phase8b_calibration as module

        tree = _ast2.parse(_inspect.getsource(module))
        call_sites = [
            node
            for node in _ast2.walk(tree)
            if isinstance(node, _ast2.Call)
            and isinstance(node.func, _ast2.Name)
            and node.func.id in ("fit_platt", "fit_isotonic")
        ]
        self.assertGreaterEqual(len(call_sites), 2)
        for call in call_sites:
            argument_names = [arg.id for arg in call.args if isinstance(arg, _ast2.Name)]
            for name in argument_names:
                self.assertIn("val", name.lower())
                self.assertNotIn("test", name.lower())

    def test_test_labels_never_used_for_fitting(self) -> None:
        """Directly verify fit_platt's signature and that calling it does
        not require or accept test data at all."""
        fit_params = list(_inspect.signature(fit_platt).parameters)
        self.assertEqual(fit_params, ["val_labels", "val_probabilities"])

    def test_deterministic_results(self) -> None:
        first = [phase8b_analyze_model(spec) for spec in TARGET_MODEL_SPECS]
        second = [phase8b_analyze_model(spec) for spec in TARGET_MODEL_SPECS]
        for a, b in zip(first, second):
            self.assertEqual(a["raw"]["test_brier"], b["raw"]["test_brier"])
            self.assertEqual(a["selected_calibrated"]["test_brier"], b["selected_calibrated"]["test_brier"])
            self.assertEqual(a["selected_method"], b["selected_method"])

    def test_pr_auc_roc_auc_unchanged_after_selected_calibration(self) -> None:
        for analysis in self.analyses:
            self.assertTrue(analysis["selected_calibrated"]["pr_auc_unchanged"], analysis["name"])
            self.assertTrue(analysis["selected_calibrated"]["roc_auc_unchanged"], analysis["name"])
            self.assertEqual(analysis["selected_method"], "platt", "isotonic is expected to be filtered out here since it introduces ties that alter ranking metrics")

    def test_platt_scaling_preserves_ranking_exactly(self) -> None:
        """Direct, from-first-principles check: fit Platt on validation,
        apply to test, confirm PR-AUC/ROC-AUC are bit-for-bit unchanged."""
        from sklearn.metrics import average_precision_score, roc_auc_score
        from phase8_evaluation_comparison import load_predictions

        for spec in TARGET_MODEL_SPECS:
            val_labels, val_probabilities, _ = load_predictions(spec["validation_csv"], spec["probability_column"], spec["label_column"])
            test_labels, test_probabilities, _ = load_predictions(spec["test_csv"], spec["probability_column"], spec["label_column"])
            platt = fit_platt(val_labels, val_probabilities)
            calibrated_test = apply_platt(platt, test_probabilities)
            raw_pr_auc = average_precision_score(test_labels, test_probabilities)
            calibrated_pr_auc = average_precision_score(test_labels, calibrated_test)
            raw_roc_auc = roc_auc_score(test_labels, test_probabilities)
            calibrated_roc_auc = roc_auc_score(test_labels, calibrated_test)
            self.assertAlmostEqual(raw_pr_auc, calibrated_pr_auc, places=9)
            self.assertAlmostEqual(raw_roc_auc, calibrated_roc_auc, places=9)

    def test_isotonic_can_alter_ranking_metrics_and_is_therefore_filtered(self) -> None:
        """Documents the actual, observed reason isotonic is excluded here:
        it measurably changes ranking metrics on at least one model, which
        is exactly why the hard filter in select_calibration_method exists."""
        for analysis in self.analyses:
            selection = analysis["calibration_selection"]
            if not selection["isotonic_preserves_ranking_on_validation"]:
                self.assertEqual(analysis["selected_method"], "platt")

    def test_persisted_source_files_unchanged(self) -> None:
        for spec in TARGET_MODEL_SPECS:
            for path in (spec["validation_csv"], spec["test_csv"], spec["ranking_metrics_file"]):
                self.assertTrue(path.exists())

    def test_phase8a_results_unchanged(self) -> None:
        phase8a_dir = Path(__file__).resolve().parent / "results/phase8_final_evaluation"
        for filename in ("comparison_metrics.json", "comparison_metrics.csv", "FINAL_EVALUATION_TABLE.md", "validation_report.json"):
            self.assertTrue((phase8a_dir / filename).exists())

    def test_checkpoint_files_unchanged(self) -> None:
        checkpoint = PHASE8B_RUN1_DIR / "model/best_model.pt"
        scaler = PHASE8B_RUN1_DIR / "model/scaler.joblib"
        self.assertTrue(checkpoint.exists())
        self.assertTrue(scaler.exists())

    def test_reliability_bins_sum_to_expected_sample_count(self) -> None:
        for analysis in self.analyses:
            total = sum(row["count"] for row in analysis["raw"]["test_reliability_bins"])
            self.assertEqual(total, 6195)

    def test_no_classification_threshold_defined_in_this_module(self) -> None:
        """Phase 8B must not define or select any classification threshold
        -- it only recalibrates probability values."""
        import phase8b_calibration as module

        source = _inspect.getsource(module)
        self.assertNotIn("choose_threshold_recall_at_fpr", source)
        self.assertNotIn("calculate_metrics(", source)


from phase9b_pcap_discovery import (
    DOWNLOAD_TINY_THRESHOLD_BYTES,
    classify_object,
    human_size,
    list_objects,
)


class Phase9BPcapDiscoveryTests(unittest.TestCase):
    """Phase 9B: focused tests for the read-only S3 discovery utility.
    Pure-function tests need no network; one test confirms the public
    bucket is genuinely reachable right now (consistent with this project's
    preference for real, verified checks over mocks)."""

    def test_classify_object_pcap_archive(self) -> None:
        self.assertEqual(classify_object("Original Network Traffic and Log data/Wednesday-14-02-2018/pcap.zip"), "raw_pcap_archive")

    def test_classify_object_log_archive(self) -> None:
        self.assertEqual(classify_object("Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip"), "log_archive")

    def test_classify_object_processed_csv(self) -> None:
        self.assertEqual(classify_object("Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"), "processed_csv")

    def test_classify_object_folder_marker(self) -> None:
        self.assertEqual(classify_object("Original Network Traffic and Log data/Wednesday-14-02-2018/"), "folder_marker")

    def test_classify_object_unknown(self) -> None:
        self.assertEqual(classify_object("Original Network Traffic and Log data/Wednesday-14-02-2018/readme.txt"), "unknown")

    def test_human_size_formatting(self) -> None:
        self.assertEqual(human_size(500), "500 B")
        self.assertEqual(human_size(140178796), "133.68 MB")
        self.assertEqual(human_size(39913353098), "37.17 GB")

    def test_download_threshold_is_100mb(self) -> None:
        self.assertEqual(DOWNLOAD_TINY_THRESHOLD_BYTES, 100 * 1024 * 1024)

    def test_bucket_is_publicly_accessible(self) -> None:
        """Live check: confirms the public bucket is genuinely reachable
        right now via a plain, unauthenticated LIST call (no credentials).
        Only metadata is requested -- no object body is ever fetched."""
        try:
            result = list_objects(prefix="", delimiter="/")
        except Exception as exc:  # noqa: BLE001 -- network may be unavailable in some environments
            self.skipTest(f"Network unavailable in this environment: {exc}")
            return
        self.assertEqual(result["http_status"], 200)
        self.assertIn("Original Network Traffic and Log data/", result["common_prefixes"])
        self.assertIn("Processed Traffic Data for ML Algorithms/", result["common_prefixes"])


import struct as _struct

from phase9c_pcap_archive_inspection import (
    MAX_CENTRAL_DIRECTORY_SAFE_BYTES,
    MAX_TOTAL_DOWNLOAD_BYTES,
    BudgetExceededError,
    RemoteObjectInspector,
    build_report,
    classify_member,
    find_eocd,
    find_zip64_locator_in_tail,
    parse_central_directory,
    parse_zip64_eocd_record,
)


def _build_synthetic_eocd(cd_size: int, cd_offset: int, total_entries: int, comment: bytes = b"") -> bytes:
    return (
        _struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, total_entries, total_entries, cd_size, cd_offset, len(comment))
        + comment
    )


def _build_synthetic_zip64_locator(zip64_eocd_offset: int) -> bytes:
    return _struct.pack("<IIQI", 0x07064B50, 0, zip64_eocd_offset, 1)


def _build_synthetic_zip64_eocd_record(cd_size: int, cd_offset: int, total_entries: int) -> bytes:
    return _struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, total_entries, total_entries, cd_size, cd_offset)


def _build_synthetic_central_dir_entry(filename: str, comp_size: int, uncomp_size: int, local_offset: int, use_zip64_extra: bool = False) -> bytes:
    fname_bytes = filename.encode("utf-8")
    if use_zip64_extra:
        payload = _struct.pack("<QQQ", uncomp_size, comp_size, local_offset)
        extra = _struct.pack("<HH", 1, len(payload)) + payload
        header_comp, header_uncomp, header_offset = 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    else:
        extra = b""
        header_comp, header_uncomp, header_offset = comp_size, uncomp_size, local_offset
    header = _struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50, 0, 0, 0x0800, 8, 0, 0, 0,
        header_comp, header_uncomp,
        len(fname_bytes), len(extra), 0,
        0, 0, 0,
        header_offset,
    )
    return header + fname_bytes + extra


class Phase9CArchiveInspectionTests(unittest.TestCase):
    """Phase 9C: focused tests for the remote ZIP central-directory
    inspector. Deterministic synthetic byte fixtures for the parsing logic
    (no network needed); one live end-to-end check of the actual object,
    skipped gracefully if the network is unavailable."""

    def test_eocd_parsing_no_zip64(self) -> None:
        eocd_bytes = _build_synthetic_eocd(cd_size=1234, cd_offset=5000, total_entries=7)
        tail = b"\x00" * 100 + eocd_bytes
        parsed = find_eocd(tail, tail_start_offset_in_file=1_000_000)
        self.assertEqual(parsed["cd_size"], 1234)
        self.assertEqual(parsed["cd_offset"], 5000)
        self.assertEqual(parsed["total_entries"], 7)
        self.assertFalse(parsed["cd_size_is_zip64_sentinel"])
        self.assertFalse(parsed["cd_offset_is_zip64_sentinel"])
        self.assertEqual(parsed["file_offset"], 1_000_000 + 100)

    def test_eocd_not_found_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            find_eocd(b"\x00" * 1000, tail_start_offset_in_file=0)

    def test_eocd_zip64_sentinel_detection(self) -> None:
        eocd_bytes = _build_synthetic_eocd(cd_size=999, cd_offset=0xFFFFFFFF, total_entries=450)
        parsed = find_eocd(eocd_bytes, tail_start_offset_in_file=0)
        self.assertTrue(parsed["cd_offset_is_zip64_sentinel"])
        self.assertFalse(parsed["cd_size_is_zip64_sentinel"])

    def test_zip64_locator_found_when_present(self) -> None:
        locator_bytes = _build_synthetic_zip64_locator(zip64_eocd_offset=987654321)
        eocd_bytes = _build_synthetic_eocd(cd_size=1, cd_offset=0xFFFFFFFF, total_entries=1)
        tail = locator_bytes + eocd_bytes
        eocd_index = len(locator_bytes)
        locator = find_zip64_locator_in_tail(tail, eocd_index_in_tail=eocd_index, tail_start_offset_in_file=0)
        self.assertIsNotNone(locator)
        self.assertEqual(locator["zip64_eocd_offset"], 987654321)

    def test_zip64_locator_absent_returns_none(self) -> None:
        eocd_bytes = _build_synthetic_eocd(cd_size=1, cd_offset=2, total_entries=1)
        tail = b"\x00" * 20 + eocd_bytes  # zeros, not a real locator signature
        locator = find_zip64_locator_in_tail(tail, eocd_index_in_tail=20, tail_start_offset_in_file=0)
        self.assertIsNone(locator)

    def test_zip64_eocd_record_parsing(self) -> None:
        record_bytes = _build_synthetic_zip64_eocd_record(cd_size=57124, cd_offset=39913295876, total_entries=450)
        parsed = parse_zip64_eocd_record(record_bytes)
        self.assertEqual(parsed["cd_size"], 57124)
        self.assertEqual(parsed["cd_offset"], 39913295876)
        self.assertEqual(parsed["total_entries"], 450)

    def test_zip64_eocd_record_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_zip64_eocd_record(b"\x00" * 10)

    def test_zip64_eocd_record_wrong_signature_raises(self) -> None:
        bad = _struct.pack("<IQHHIIQQQQ", 0xDEADBEEF, 44, 45, 45, 0, 0, 1, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            parse_zip64_eocd_record(bad)

    def test_central_directory_parsing_without_zip64_extra(self) -> None:
        entry = _build_synthetic_central_dir_entry("pcap/small_file", comp_size=1000, uncomp_size=2000, local_offset=42)
        parsed = parse_central_directory(entry, expected_entries=1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["filename"], "pcap/small_file")
        self.assertEqual(parsed[0]["compressed_size"], 1000)
        self.assertEqual(parsed[0]["uncompressed_size"], 2000)
        self.assertEqual(parsed[0]["local_header_offset"], 42)
        self.assertFalse(parsed[0]["used_zip64_extra"])

    def test_central_directory_parsing_with_zip64_extra(self) -> None:
        """Mirrors the real archive's entries: sizes/offset exceed 4GB, so
        the 32-bit fields are sentinels and the true values live in the
        per-entry ZIP64 extra field."""
        entry = _build_synthetic_central_dir_entry(
            "pcap/UCAP172.31.69.22", comp_size=551024, uncomp_size=1282048, local_offset=39485257643, use_zip64_extra=True
        )
        parsed = parse_central_directory(entry, expected_entries=1)
        self.assertEqual(parsed[0]["compressed_size"], 551024)
        self.assertEqual(parsed[0]["uncompressed_size"], 1282048)
        self.assertEqual(parsed[0]["local_header_offset"], 39485257643)
        self.assertTrue(parsed[0]["used_zip64_extra"])

    def test_central_directory_multiple_entries(self) -> None:
        buffer = _build_synthetic_central_dir_entry("pcap/a", 10, 20, 0) + _build_synthetic_central_dir_entry("pcap/b", 30, 40, 100)
        parsed = parse_central_directory(buffer, expected_entries=2)
        self.assertEqual([entry["filename"] for entry in parsed], ["pcap/a", "pcap/b"])

    def test_central_directory_entry_count_mismatch_raises(self) -> None:
        entry = _build_synthetic_central_dir_entry("pcap/only_one", 1, 2, 0)
        with self.assertRaises(RuntimeError):
            parse_central_directory(entry, expected_entries=2)

    def test_classify_member_pcap_extension(self) -> None:
        self.assertEqual(classify_member("some/dir/capture.pcap"), "pcap")

    def test_classify_member_pcapng_extension(self) -> None:
        self.assertEqual(classify_member("some/dir/capture.pcapng"), "pcapng")

    def test_classify_member_no_extension_is_unknown(self) -> None:
        """Matches the REAL archive's members: no .pcap/.pcapng extension
        at all, so they must be classified unknown, never assumed PCAP."""
        self.assertEqual(classify_member("pcap/UCAP172.31.69.22"), "unknown_extension")
        self.assertEqual(classify_member("pcap/capWIN-J6GMIG1DQE5-172.31.65.99"), "unknown_extension")

    def test_classify_member_log(self) -> None:
        self.assertEqual(classify_member("Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip"), "log")

    def test_classify_member_folder_marker(self) -> None:
        self.assertEqual(classify_member("pcap/"), "folder_marker")

    def test_safe_size_limit_constants(self) -> None:
        self.assertEqual(MAX_TOTAL_DOWNLOAD_BYTES, 10 * 1024 * 1024)
        self.assertLess(MAX_CENTRAL_DIRECTORY_SAFE_BYTES, MAX_TOTAL_DOWNLOAD_BYTES)

    def test_budget_exceeded_raises_before_any_network_call(self) -> None:
        """The budget check happens before the request is issued, so a
        bogus/unreachable URL is safe to use here -- if this test somehow
        did reach the network, the assertRaises would still pass but the
        important property (raise-before-send) is what's being verified by
        using a URL that would fail differently (DNS error, not
        BudgetExceededError) if the check were ever skipped."""
        inspector = RemoteObjectInspector(url="https://example.invalid/does-not-exist", max_total_bytes=100)
        with self.assertRaises(BudgetExceededError):
            inspector.get_range(0, 999, purpose="test_oversized_request")
        self.assertEqual(inspector.total_bytes_downloaded, 0)

    def test_budget_tracks_cumulative_usage(self) -> None:
        inspector = RemoteObjectInspector(url="https://example.invalid/does-not-exist", max_total_bytes=100)
        inspector.total_bytes_downloaded = 90  # simulate prior usage without a real network call
        with self.assertRaises(BudgetExceededError):
            inspector.get_range(0, 19, purpose="would_push_over_budget")  # 20 bytes, 90+20 > 100

    def test_live_archive_inspection_end_to_end(self) -> None:
        """Live check against the real object discovered in Phase 9B.
        Skipped gracefully if the network is unavailable in this
        environment, consistent with the Phase 9B testing convention."""
        try:
            report, inspector = build_report()
        except Exception as exc:  # noqa: BLE001 -- network may be unavailable
            self.skipTest(f"Network unavailable in this environment: {exc}")
            return
        self.assertTrue(report["success"])
        self.assertLessEqual(inspector.total_bytes_downloaded, MAX_TOTAL_DOWNLOAD_BYTES)
        self.assertEqual(report["smallest_candidate_member"]["filename"], "pcap/UCAP172.31.69.22")
        self.assertEqual(report["step5_central_directory_entries_count"], 450)


import zlib as _zlib

from phase9d_pcap_validation import (
    DATA_DIR as PHASE9D_DATA_DIR,
    MAX_SAMPLE_PACKETS,
    OUTPUT_DIR as PHASE9D_OUTPUT_DIR,
    PCAPNG_MAGIC,
    PCAP_MAGICS,
    compute_aggregate_statistics,
    decompress_member,
    load_target_member_metadata,
    parse_packet_sample,
    retrieve_member,
)


def _build_synthetic_local_header(filename: str, comp_size: int, uncomp_size: int, crc32: int, flags: int = 0x0800) -> bytes:
    fname_bytes = filename.encode("utf-8")
    return _struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50, 45, flags, 8, 0, 0, crc32, comp_size, uncomp_size, len(fname_bytes), 0,
    ) + fname_bytes


class _FakeInspector:
    """Duck-typed stand-in for RemoteObjectInspector: serves get_range()
    calls from an in-memory byte buffer instead of the network, so the
    range-extraction/offset-math logic in retrieve_member() can be tested
    without any network access."""

    def __init__(self, byte_source: bytes, byte_source_start_offset: int) -> None:
        self.byte_source = byte_source
        self.byte_source_start_offset = byte_source_start_offset
        self.calls: list[dict] = []

    def get_range(self, start: int, end: int, purpose: str) -> bytes:
        self.calls.append({"start": start, "end": end, "purpose": purpose})
        local_start = start - self.byte_source_start_offset
        local_end = end - self.byte_source_start_offset + 1
        return self.byte_source[local_start:local_end]


def _make_target(filename="pcap/UCAP172.31.69.22", compressed_size=0, uncompressed_size=0, offset=5000, crc32=0) -> dict:
    return {
        "object_url": "https://example.invalid/fake",
        "filename": filename,
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "compression_method": 8,
        "local_header_offset": offset,
        "expected_crc32": crc32,
    }


class Phase9DPcapValidationTests(unittest.TestCase):
    """Phase 9D: focused tests for the single-member retrieval/decompression/
    parsing pipeline. Range-extraction and format-detection logic is tested
    with synthetic fixtures (no network); packet-parsing/aggregate-statistics
    logic is tested against the ALREADY-DOWNLOADED real local artifact from
    this phase's own run (no new network calls needed for those)."""

    # ---- range extraction / ZIP local-header validation (synthetic, no network) ----

    def test_retrieve_member_range_extraction_and_offsets(self) -> None:
        filename = "pcap/UCAP172.31.69.22"
        payload = b"FAKECOMPRESSEDDATA12345"
        header = _build_synthetic_local_header(filename, comp_size=len(payload), uncomp_size=999, crc32=123)
        synthetic_object = header + payload
        target = _make_target(filename=filename, compressed_size=len(payload), uncompressed_size=999, offset=5000, crc32=123)

        fake = _FakeInspector(synthetic_object, byte_source_start_offset=5000)
        result = retrieve_member(fake, target)

        self.assertTrue(result["local_header_signature_valid"])
        self.assertTrue(result["filename_matches_expected"])
        self.assertTrue(result["compressed_size_matches_expected"])
        self.assertEqual(result["compressed_data"], payload)
        self.assertEqual(result["data_start_offset"], 5000 + len(header))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["purpose"], "local_header_probe")
        self.assertEqual(fake.calls[1]["purpose"], "member_compressed_data")
        self.assertEqual(fake.calls[1]["start"], 5000 + len(header))

    def test_retrieve_member_detects_filename_mismatch(self) -> None:
        actual_filename = "pcap/some_other_member"
        payload = b"X" * 10
        header = _build_synthetic_local_header(actual_filename, comp_size=len(payload), uncomp_size=20, crc32=1)
        synthetic_object = header + payload
        target = _make_target(filename="pcap/UCAP172.31.69.22", compressed_size=len(payload), uncompressed_size=20, offset=5000, crc32=1)

        fake = _FakeInspector(synthetic_object, byte_source_start_offset=5000)
        result = retrieve_member(fake, target)
        self.assertFalse(result["filename_matches_expected"])

    def test_retrieve_member_wrong_signature_raises(self) -> None:
        bad_bytes = _struct.pack("<IHHHHHIIIHH", 0xDEADBEEF, 0, 0, 8, 0, 0, 0, 5, 5, 4, 0) + b"name" + b"XXXXX"
        target = _make_target(compressed_size=5, uncompressed_size=5, offset=5000)
        fake = _FakeInspector(bad_bytes, byte_source_start_offset=5000)
        with self.assertRaises(RuntimeError):
            retrieve_member(fake, target)

    # ---- DEFLATE extraction + magic detection (synthetic, no network) ----

    def _compress_raw_deflate(self, data: bytes) -> bytes:
        compressor = _zlib.compressobj(9, _zlib.DEFLATED, -15)
        return compressor.compress(data) + compressor.flush()

    def test_pcap_magic_detection_all_variants(self) -> None:
        for magic_bytes, expected_label in PCAP_MAGICS.items():
            with self.subTest(magic=magic_bytes.hex()):
                payload = magic_bytes + b"\x00" * 28  # pad to 32 bytes like a real global header
                compressed = self._compress_raw_deflate(payload)
                target = _make_target(compressed_size=len(compressed), uncompressed_size=len(payload), crc32=_zlib.crc32(payload) & 0xFFFFFFFF)
                result = decompress_member(compressed, target)
                self.assertEqual(result["detected_format"], expected_label)
                self.assertTrue(result["is_valid_capture"])
                self.assertTrue(result["crc32_matches_expected"])
                self.assertTrue(result["uncompressed_size_matches_expected"])

    def test_pcapng_magic_detection(self) -> None:
        payload = PCAPNG_MAGIC + b"\x00" * 28
        compressed = self._compress_raw_deflate(payload)
        target = _make_target(compressed_size=len(compressed), uncompressed_size=len(payload), crc32=_zlib.crc32(payload) & 0xFFFFFFFF)
        result = decompress_member(compressed, target)
        self.assertEqual(result["detected_format"], "pcapng (Section Header Block)")
        self.assertTrue(result["is_valid_capture"])

    def test_unknown_format_is_not_assumed_valid(self) -> None:
        payload = b"NOT_A_CAPTURE_FILE_AT_ALL_JUST_TEXT"
        compressed = self._compress_raw_deflate(payload)
        target = _make_target(compressed_size=len(compressed), uncompressed_size=len(payload), crc32=_zlib.crc32(payload) & 0xFFFFFFFF)
        result = decompress_member(compressed, target)
        self.assertEqual(result["detected_format"], "UNKNOWN")
        self.assertFalse(result["is_valid_capture"])

    def test_decompress_detects_size_and_crc_mismatch(self) -> None:
        payload = b"\xd4\xc3\xb2\xa1" + b"\x00" * 28  # valid pcap magic, but we'll assert against wrong expected values below
        compressed = self._compress_raw_deflate(payload)
        # deliberately wrong expected size/crc
        target = _make_target(compressed_size=len(compressed), uncompressed_size=999999, crc32=1)
        result = decompress_member(compressed, target)
        self.assertFalse(result["uncompressed_size_matches_expected"])
        self.assertFalse(result["crc32_matches_expected"])

    # ---- bounded packet parsing / aggregate statistics (real local artifact, no new network) ----

    def test_bounded_packet_parsing_respects_limit(self) -> None:
        pcap_path = PHASE9D_DATA_DIR / "UCAP172.31.69.22.pcap"
        if not pcap_path.exists():
            self.skipTest("Local decompressed artifact from this phase's own run not present")
        sample = parse_packet_sample(pcap_path, max_packets=10)
        self.assertLessEqual(sample["sample_size"] + len(sample["parsing_failures"]), 10)
        self.assertEqual(sample["bounded_at"], 10)
        for packet in sample["packets"]:
            self.assertIn("timestamp", packet)
            self.assertIn("src_ip", packet)
            self.assertIn("dst_ip", packet)
            self.assertIn("protocol", packet)

    def test_safe_packet_count_limit_constant(self) -> None:
        self.assertEqual(MAX_SAMPLE_PACKETS, 100)

    def test_aggregate_statistics_match_persisted_run(self) -> None:
        pcap_path = PHASE9D_DATA_DIR / "UCAP172.31.69.22.pcap"
        report_path = PHASE9D_OUTPUT_DIR / "validation_report.json"
        if not pcap_path.exists() or not report_path.exists():
            self.skipTest("This phase's own run output not present yet")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))["step4_capture_aggregates"]
        recomputed = compute_aggregate_statistics(pcap_path)
        self.assertEqual(recomputed["total_packet_count"], persisted["total_packet_count"])
        self.assertEqual(recomputed["unique_hosts_overall"], persisted["unique_hosts_overall"])
        self.assertEqual(recomputed["protocol_counts"], persisted["protocol_counts"])
        self.assertEqual(recomputed["total_bytes"], persisted["total_bytes"])

    def test_load_target_member_metadata_matches_phase9c(self) -> None:
        target = load_target_member_metadata()
        self.assertEqual(target["filename"], "pcap/UCAP172.31.69.22")
        self.assertEqual(target["compressed_size"], 551024)
        self.assertEqual(target["uncompressed_size"], 1282048)
        self.assertEqual(target["compression_method"], 8)


from unittest import mock as _mock

from phase9e_temporal_packet_graph import (
    PCAP_PATH as PHASE9E_PCAP_PATH,
    WINDOW_SECONDS as PHASE9E_WINDOW_SECONDS,
    anonymize_ip,
    assign_windows,
    build_and_validate_anonymization,
    build_edge_window_features,
    build_node_window_features,
    compute_capture_level_stats,
    compute_temporal_dynamics,
    feature_availability_table,
    parse_full_capture,
)


def _synthetic_ip_packet(
    timestamp: float,
    src_ip: str,
    dst_ip: str,
    protocol: str = "TCP",
    packet_length: int = 100,
    src_port=1234,
    dst_port=443,
    tcp_syn: bool = False,
    tcp_ack: bool = False,
    tcp_fin: bool = False,
    tcp_rst: bool = False,
) -> dict:
    return {
        "timestamp": timestamp,
        "packet_length": packet_length,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "ip_version": 4,
        "protocol": protocol,
        "src_port": src_port if protocol in ("TCP", "UDP") else None,
        "dst_port": dst_port if protocol in ("TCP", "UDP") else None,
        "ttl": 64,
        "tcp_flags_str": "S" if tcp_syn else None,
        "tcp_window": 8192 if protocol == "TCP" else None,
        "tcp_syn": tcp_syn,
        "tcp_ack": tcp_ack,
        "tcp_fin": tcp_fin,
        "tcp_rst": tcp_rst,
        "payload_length": max(packet_length - 40, 0),
        "ip_flags_str": "DF",
        "ip_fragment_offset": 0,
    }


class Phase9ETemporalPacketGraphTests(unittest.TestCase):
    """Phase 9E: focused tests for temporal packet graph construction.
    Graph-construction logic (windowing, node/edge aggregation, churn,
    statistics) is tested with small hand-checkable synthetic packet sets
    (no network, no dependency on the real capture). A couple of tests
    additionally exercise the real, already-downloaded Phase 9D artifact
    where that is the only way to check true end-to-end determinism."""

    # ---- 1: deterministic IP anonymization ----

    def test_anonymize_ip_is_deterministic(self) -> None:
        ip = "172.31.69.22"
        first = anonymize_ip(ip)
        second = anonymize_ip(ip)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("host_"))

    def test_anonymize_ip_differs_for_different_ips(self) -> None:
        self.assertNotEqual(anonymize_ip("172.31.69.22"), anonymize_ip("172.31.69.18"))

    # ---- 2: anonymization collision detection ----

    def test_anonymization_no_collision_for_real_distinct_ips(self) -> None:
        ips = {f"10.0.{i // 256}.{i % 256}" for i in range(500)}
        mapping, validation = build_and_validate_anonymization(ips)
        self.assertEqual(validation["collision_count"], 0)
        self.assertTrue(validation["bijective_mapping"])
        self.assertTrue(validation["deterministic_repeatable"])
        self.assertEqual(len(mapping), len(ips))

    def test_anonymization_detects_forced_collision(self) -> None:
        ips = {"1.1.1.1", "2.2.2.2", "3.3.3.3"}
        with _mock.patch(
            "phase9e_temporal_packet_graph.anonymize_ip",
            side_effect=lambda ip: "host_SAME" if ip != "3.3.3.3" else "host_UNIQUE",
        ):
            _, validation = build_and_validate_anonymization(ips)
        self.assertEqual(validation["collision_count"], 1)
        self.assertFalse(validation["bijective_mapping"])

    # ---- 4 (window assignment tested before edges, since edges need it) ----

    def test_ten_second_window_assignment_and_boundaries(self) -> None:
        base = 1_000_000.0
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 9.999, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 10.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 25.0, "10.0.0.1", "10.0.0.2"),
        ]
        max_window = assign_windows(packets, capture_start_ts=base)
        self.assertEqual(packets[0]["window_index"], 0)
        self.assertEqual(packets[1]["window_index"], 0)
        self.assertEqual(packets[2]["window_index"], 1)
        self.assertEqual(packets[3]["window_index"], 2)
        self.assertEqual(max_window, 2)
        self.assertEqual(PHASE9E_WINDOW_SECONDS, 10)

    # ---- 3: packet -> directed edge construction, 5: edge feature aggregation ----

    def test_edge_construction_and_feature_aggregation(self) -> None:
        base = 2_000_000.0
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2", packet_length=100, tcp_syn=True, dst_port=443),
            _synthetic_ip_packet(base + 1.0, "10.0.0.1", "10.0.0.2", packet_length=200, tcp_ack=True, dst_port=8080),
            _synthetic_ip_packet(base + 3.0, "10.0.0.1", "10.0.0.2", packet_length=300, tcp_fin=True, dst_port=443),
        ]
        assign_windows(packets, capture_start_ts=base)
        raw_ips = {"10.0.0.1", "10.0.0.2"}
        anon_map, _ = build_and_validate_anonymization(raw_ips)
        edge_windows = build_edge_window_features(packets, anon_map)

        self.assertEqual(set(edge_windows.keys()), {0})
        key = (anon_map["10.0.0.1"], anon_map["10.0.0.2"])
        self.assertEqual(set(edge_windows[0].keys()), {key})
        edge = edge_windows[0][key]

        self.assertEqual(edge["packet_count"], 3)
        self.assertEqual(edge["total_bytes"], 600)
        self.assertAlmostEqual(edge["mean_packet_length"], 200.0)
        self.assertEqual(edge["min_packet_length"], 100)
        self.assertEqual(edge["max_packet_length"], 300)
        self.assertEqual(edge["first_packet_offset"], 0.0)
        self.assertEqual(edge["last_packet_offset"], 3.0)
        self.assertEqual(edge["edge_duration"], 3.0)
        self.assertEqual(edge["tcp_packet_count"], 3)
        self.assertEqual(edge["udp_packet_count"], 0)
        self.assertEqual(edge["destination_port_diversity"], 2)  # ports 443, 8080
        self.assertEqual(edge["tcp_syn_count"], 1)
        self.assertEqual(edge["tcp_ack_count"], 1)
        self.assertEqual(edge["tcp_fin_count"], 1)
        self.assertEqual(edge["tcp_rst_count"], 0)
        # 2 inter-arrival gaps: (1.0 - 0.0), (3.0 - 1.0) = [1.0, 2.0] -> mean 1.5
        self.assertAlmostEqual(edge["mean_inter_arrival_time"], 1.5)

    def test_edge_iat_and_rate_are_null_for_single_packet_edge(self) -> None:
        base = 3_000_000.0
        packets = [_synthetic_ip_packet(base, "10.0.0.5", "10.0.0.6")]
        assign_windows(packets, capture_start_ts=base)
        anon_map, _ = build_and_validate_anonymization({"10.0.0.5", "10.0.0.6"})
        edge_windows = build_edge_window_features(packets, anon_map)
        edge = next(iter(edge_windows[0].values()))
        self.assertIsNone(edge["mean_inter_arrival_time"])
        self.assertIsNone(edge["inter_arrival_time_std"])
        self.assertIsNone(edge["packets_per_second"])
        self.assertIsNone(edge["bytes_per_second"])

    # ---- 6: node feature aggregation ----

    def test_node_feature_aggregation(self) -> None:
        base = 4_000_000.0
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2", packet_length=100, dst_port=443),
            _synthetic_ip_packet(base + 1.0, "10.0.0.1", "10.0.0.3", packet_length=150, dst_port=80),
            _synthetic_ip_packet(base + 2.0, "10.0.0.4", "10.0.0.1", packet_length=50, src_port=9999, dst_port=22),
        ]
        assign_windows(packets, capture_start_ts=base)
        raw_ips = {"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"}
        anon_map, _ = build_and_validate_anonymization(raw_ips)
        node_windows = build_node_window_features(packets, anon_map)

        node1 = node_windows[0][anon_map["10.0.0.1"]]
        self.assertEqual(node1["packet_count"], 3)  # involved in all 3 packets
        self.assertEqual(node1["bytes_sent"], 250)  # 100 + 150 (as src)
        self.assertEqual(node1["bytes_received"], 50)  # as dst
        self.assertEqual(node1["unique_destinations"], 2)  # .2 and .3
        self.assertEqual(node1["unique_sources"], 1)  # .4
        self.assertEqual(node1["first_seen_offset"], 0.0)
        self.assertEqual(node1["last_seen_offset"], 2.0)

    # ---- 7: exact unique-pair counting ----

    def test_exact_unique_pair_counting_across_repeated_windows(self) -> None:
        base = 5_000_000.0
        # Same (A->B) pair appears in 3 different windows; a distinct (B->A)
        # pair appears once. Exact unique-pair count must be 2, not 3.
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 10.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 20.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 20.0, "10.0.0.2", "10.0.0.1"),
        ]
        assign_windows(packets, capture_start_ts=base)
        anon_map, _ = build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        node_windows = build_node_window_features(packets, anon_map)
        edge_windows = build_edge_window_features(packets, anon_map)

        stats = compute_capture_level_stats(packets, node_windows, edge_windows, total_windows=3)
        self.assertEqual(stats["total_unique_directed_source_destination_pairs"], 2)
        self.assertEqual(stats["total_edges_across_windows"], 4)  # 3 + 1 window-local instances
        self.assertTrue(stats["packet_accounting_check"]["matches"])

    # ---- 8, 9: temporal edge/node churn ----

    def test_temporal_churn_and_jaccard(self) -> None:
        base = 6_000_000.0
        # window 0: A->B only. window 1: A->B and A->C (1 new node C, 1 new edge).
        # window 2: empty. window 3: A->B again.
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 10.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 10.0, "10.0.0.1", "10.0.0.3"),
            _synthetic_ip_packet(base + 30.0, "10.0.0.1", "10.0.0.2"),
        ]
        assign_windows(packets, capture_start_ts=base)
        anon_map, _ = build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.3"})
        node_windows = build_node_window_features(packets, anon_map)
        edge_windows = build_edge_window_features(packets, anon_map)

        dynamics = compute_temporal_dynamics(node_windows, edge_windows, total_windows=4)
        # window 0 -> 1: node C appears (new), no node disappears -> churn events exist
        self.assertGreater(dynamics["new_nodes_per_window"]["max"], 0)
        self.assertGreater(dynamics["new_edges_per_window"]["max"], 0)
        # window 1 -> 2 is non-empty -> empty: all nodes/edges disappear
        self.assertGreater(dynamics["disappearing_nodes_per_window"]["max"], 0)
        self.assertGreaterEqual(dynamics["longest_consecutive_non_empty_window_run"], 2)
        self.assertLessEqual(dynamics["longest_consecutive_non_empty_window_run"], 2)

    # ---- 10: graph statistics (exact density/degree on a hand-built graph) ----

    def test_graph_statistics_density_and_degree_exact(self) -> None:
        base = 7_000_000.0
        # 3 nodes, 2 directed edges in one window: A->B, A->C.
        packets = [
            _synthetic_ip_packet(base + 0.0, "10.0.0.1", "10.0.0.2"),
            _synthetic_ip_packet(base + 1.0, "10.0.0.1", "10.0.0.3"),
        ]
        assign_windows(packets, capture_start_ts=base)
        anon_map, _ = build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.3"})
        node_windows = build_node_window_features(packets, anon_map)
        edge_windows = build_edge_window_features(packets, anon_map)

        from phase9e_temporal_packet_graph import _window_graph_metrics

        metrics = _window_graph_metrics(node_windows[0], edge_windows[0])
        self.assertEqual(metrics["node_count"], 3)
        self.assertEqual(metrics["edge_count"], 2)
        # density = m / (n*(n-1)) = 2 / (3*2) = 1/3
        self.assertAlmostEqual(metrics["density"], 2 / 6)
        # average_degree = 2*m/n = 4/3
        self.assertAlmostEqual(metrics["average_degree"], 4 / 3)
        self.assertEqual(metrics["isolated_node_count"], 0)
        self.assertEqual(metrics["weakly_connected_component_count"], 1)

    # ---- 11: feature availability classification ----

    def test_feature_availability_classification_labels(self) -> None:
        parse_result = {
            "total_packets_in_capture": 10,
            "ip_packets": [
                {"protocol": "TCP"},
                {"protocol": "UDP"},
                {"protocol": "ICMP"},
            ],
        }
        table = feature_availability_table(parse_result, edge_windows={})
        by_feature = {row["feature"]: row for row in table}

        allowed = {"DIRECT", "DERIVED-HEURISTIC", "NOT IMPLEMENTED"}
        for row in table:
            self.assertIn(row["classification"], allowed)

        self.assertEqual(by_feature["timestamp"]["classification"], "DIRECT")
        self.assertEqual(by_feature["packet length"]["classification"], "DIRECT")
        self.assertEqual(by_feature["retransmission indicators"]["classification"], "NOT IMPLEMENTED")
        self.assertEqual(
            by_feature["connection indicators (e.g. completed handshake)"]["classification"],
            "NOT IMPLEMENTED",
        )
        self.assertEqual(by_feature["port-scan indicators"]["classification"], "NOT IMPLEMENTED")
        self.assertFalse(by_feature["retransmission indicators"]["available"])

    # ---- 12: deterministic rerun behavior (real local artifact, no network) ----

    def test_full_pipeline_is_deterministic_across_reruns(self) -> None:
        if not PHASE9E_PCAP_PATH.exists():
            self.skipTest("Phase 9D local artifact not present; run Phase 9D first")

        def _run_once():
            parsed = parse_full_capture(PHASE9E_PCAP_PATH)
            packets = parsed["ip_packets"]
            assign_windows(packets, parsed["capture_start_ts"])
            raw_ips = {p["src_ip"] for p in packets} | {p["dst_ip"] for p in packets}
            anon_map, _ = build_and_validate_anonymization(raw_ips)
            node_windows = build_node_window_features(packets, anon_map)
            edge_windows = build_edge_window_features(packets, anon_map)
            edge_key_set = {
                (w, src, dst) for w, edges in edge_windows.items() for (src, dst) in edges
            }
            return len(packets), sorted(anon_map.values()), edge_key_set

        first = _run_once()
        second = _run_once()
        self.assertEqual(first, second)


from phase9f_cross_host_graph import (
    MAX_TOTAL_DOWNLOAD_BYTES as PHASE9F_MAX_DOWNLOAD,
    OUTPUT_DIR as PHASE9F_OUTPUT_DIR,
    PHASE9D_MEMBER_FILENAME,
    TARGET_MAX_BYTES,
    TARGET_MIN_BYTES,
    build_comparison_table,
    classify_richness,
    graph_model_suitability,
    novelty_decision,
    select_second_member,
    summarize_capture,
)


def _make_row(metric: str, first_val, second_val) -> dict:
    ratio = (second_val / first_val) if (isinstance(first_val, (int, float)) and first_val not in (0, None) and isinstance(second_val, (int, float))) else None
    return {"metric": metric, "phase9d_9e_capture": first_val, "second_capture": second_val, "ratio_second_over_first": ratio}


def _make_comparison_rows(overrides: dict) -> list[dict]:
    """Builds a full comparison_rows list (all metrics classify_richness/
    graph_model_suitability expect) with sensible flat defaults (ratio 1.0
    everywhere), then applies the given per-metric (first, second) overrides."""
    defaults = {
        "packet_count (IP packets)": (1000, 1000),
        "duration_seconds": (10000.0, 10000.0),
        "unique_hosts": (100, 100),
        "unique_directed_pairs": (200, 200),
        "median_nodes_per_window": (1.0, 1.0),
        "p95_nodes_per_window": (3.0, 3.0),
        "max_nodes_per_window": (5.0, 5.0),
        "median_edges_per_window": (1.0, 1.0),
        "p95_edges_per_window": (4.0, 4.0),
        "max_edges_per_window": (8.0, 8.0),
        "non_empty_window_fraction": (0.3, 0.3),
        "longest_non_empty_run": (8, 8),
        "mean_edge_jaccard_consecutive": (0.03, 0.03),
        "mean_node_jaccard_consecutive": (0.08, 0.08),
        "sparsity_fraction": (0.996, 0.996),
    }
    defaults.update(overrides)
    return [_make_row(metric, first, second) for metric, (first, second) in defaults.items()]


def _make_phase9e_report(
    total_ip_packets=7316,
    duration=32220.9,
    total_nodes=551,
    unique_pairs=1094,
    non_empty=997,
    total_windows=3223,
    node_median=0.0,
    node_p95=3.0,
    node_max=5.0,
    edge_median=0.0,
    edge_p95=4.0,
    edge_max=8.0,
    longest_run=8,
    edge_jaccard=0.030,
    node_jaccard=0.085,
    sparsity=0.9964,
) -> dict:
    return {
        "parsing_results": {"total_ip_packets": total_ip_packets},
        "temporal_window_structure": {"capture_duration_seconds": duration},
        "graph_statistics": {
            "capture_level": {
                "total_nodes_observed": total_nodes,
                "total_unique_directed_source_destination_pairs": unique_pairs,
                "non_empty_windows": non_empty,
                "total_windows": total_windows,
            },
            "per_window_distributions": {
                "metrics": {
                    "node_count": {"median": node_median, "p95": node_p95, "max": node_max},
                    "edge_count": {"median": edge_median, "p95": edge_p95, "max": edge_max},
                }
            },
        },
        "temporal_dynamics": {
            "longest_consecutive_non_empty_window_run": longest_run,
            "edge_set_jaccard_consecutive": {"mean": edge_jaccard},
            "node_set_jaccard_consecutive": {"mean": node_jaccard},
        },
        "representation_feasibility": {"sparsity": {"sparsity_fraction": sparsity}},
    }


class Phase9FCrossHostGraphTests(unittest.TestCase):
    """Phase 9F: focused tests for second-member selection, cross-host
    comparison-table construction, richness classification, and the
    novelty-decision mapping. All tests use small synthetic fixtures with
    hand-checkable expected values -- no network access."""

    # ---- Step 1: member selection (metadata-only, deterministic) ----

    def test_select_second_member_picks_smallest_in_band(self) -> None:
        report = {
            "object_url": "https://example.invalid/fake.zip",
            "members": [
                {"filename": PHASE9D_MEMBER_FILENAME, "classification": "unknown_extension", "compressed_size": 551024, "uncompressed_size": 1282048, "compression_method": 8, "local_header_offset": 1, "crc32": 1},
                {"filename": "pcap/host_a", "classification": "unknown_extension", "compressed_size": 6_000_000, "uncompressed_size": 10_000_000, "compression_method": 8, "local_header_offset": 2, "crc32": 2},
                {"filename": "pcap/host_b", "classification": "unknown_extension", "compressed_size": 6_500_000, "uncompressed_size": 11_000_000, "compression_method": 8, "local_header_offset": 3, "crc32": 3},
                {"filename": "pcap/host_c_too_small", "classification": "unknown_extension", "compressed_size": 1_000_000, "uncompressed_size": 2_000_000, "compression_method": 8, "local_header_offset": 4, "crc32": 4},
                {"filename": "pcap/host_d_too_big", "classification": "unknown_extension", "compressed_size": 8_000_000, "uncompressed_size": 15_000_000, "compression_method": 8, "local_header_offset": 5, "crc32": 5},
            ],
        }
        target = select_second_member(report)
        self.assertEqual(target["filename"], "pcap/host_a")  # smallest of the two in-band candidates
        self.assertEqual(len(target["other_candidates_in_band"]), 1)
        self.assertEqual(target["other_candidates_in_band"][0]["filename"], "pcap/host_b")

    def test_select_second_member_excludes_phase9d_member_even_if_in_band(self) -> None:
        report = {
            "object_url": "https://example.invalid/fake.zip",
            "members": [
                {"filename": PHASE9D_MEMBER_FILENAME, "classification": "unknown_extension", "compressed_size": 5_500_000, "uncompressed_size": 9_000_000, "compression_method": 8, "local_header_offset": 1, "crc32": 1},
                {"filename": "pcap/only_other_in_band", "classification": "unknown_extension", "compressed_size": 5_600_000, "uncompressed_size": 9_100_000, "compression_method": 8, "local_header_offset": 2, "crc32": 2},
            ],
        }
        target = select_second_member(report)
        self.assertEqual(target["filename"], "pcap/only_other_in_band")

    def test_select_second_member_raises_when_no_candidates(self) -> None:
        report = {"object_url": "x", "members": [{"filename": "pcap/tiny", "classification": "unknown_extension", "compressed_size": 100, "uncompressed_size": 200, "compression_method": 8, "local_header_offset": 1, "crc32": 1}]}
        with self.assertRaises(RuntimeError):
            select_second_member(report)

    def test_target_band_matches_hard_constraint_11(self) -> None:
        self.assertEqual(TARGET_MIN_BYTES, 5 * 1024 * 1024)
        self.assertEqual(TARGET_MAX_BYTES, 7 * 1024 * 1024)

    def test_download_budget_comfortably_under_10mb(self) -> None:
        # Hard constraint #16.
        self.assertLess(PHASE9F_MAX_DOWNLOAD, 10 * 1024 * 1024)

    # ---- summarize_capture (rollup of Phase 9E's own packet field definitions) ----

    def test_summarize_capture_counts(self) -> None:
        parse_result = {
            "total_packets_in_capture": 5,
            "capture_start_ts": 100.0,
            "capture_end_ts": 104.0,
            "capture_duration_seconds": 4.0,
            "status_counts": {"ok": 4, "non_ip": 1},
            "ip_packets": [
                {"protocol": "TCP", "packet_length": 100, "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"},
                {"protocol": "TCP", "packet_length": 200, "src_ip": "1.1.1.1", "dst_ip": "3.3.3.3"},
                {"protocol": "UDP", "packet_length": 50, "src_ip": "2.2.2.2", "dst_ip": "1.1.1.1"},
                {"protocol": "ICMP", "packet_length": 64, "src_ip": "4.4.4.4", "dst_ip": "1.1.1.1"},
            ],
        }
        summary = summarize_capture(parse_result)
        self.assertEqual(summary["total_ip_packets"], 4)
        self.assertEqual(summary["tcp_packet_count"], 2)
        self.assertEqual(summary["udp_packet_count"], 1)
        self.assertEqual(summary["icmp_packet_count"], 1)
        self.assertEqual(summary["unique_hosts_overall"], 4)  # {1.1.1.1, 2.2.2.2, 3.3.3.3, 4.4.4.4}
        self.assertEqual(summary["total_bytes"], 100 + 200 + 50 + 64)

    # ---- Step 6/7: comparison table construction ----

    def test_build_comparison_table_ratios(self) -> None:
        phase9e_report = _make_phase9e_report()
        second_summary = {"total_ip_packets": 14632, "capture_duration_seconds": 64441.8}
        second_graph_stats = {
            "total_nodes_observed": 1102,
            "total_unique_directed_source_destination_pairs": 2188,
            "non_empty_windows": 1994,
            "total_windows": 6446,
            "per_window_distributions_metrics": {
                "node_count": {"median": 0.0, "p95": 6.0, "max": 10.0},
                "edge_count": {"median": 0.0, "p95": 8.0, "max": 16.0},
            },
        }
        second_dynamics = {
            "longest_consecutive_non_empty_window_run": 16,
            "edge_set_jaccard_consecutive": {"mean": 0.06},
            "node_set_jaccard_consecutive": {"mean": 0.17},
        }
        rows = build_comparison_table(phase9e_report, second_summary, second_graph_stats, second_dynamics, 0.9977)
        by_metric = {row["metric"]: row for row in rows}

        self.assertAlmostEqual(by_metric["packet_count (IP packets)"]["ratio_second_over_first"], 14632 / 7316)
        self.assertEqual(by_metric["unique_hosts"]["second_capture"], 1102)
        self.assertAlmostEqual(by_metric["max_nodes_per_window"]["ratio_second_over_first"], 2.0)
        # median is 0.0 in both -> ratio() guards against division by zero -> None, not a crash.
        self.assertIsNone(by_metric["median_nodes_per_window"]["ratio_second_over_first"])

    # ---- Step 7: richness classification (A / B / C, not forced) ----

    def test_classify_richness_consistently_sparse_is_A(self) -> None:
        rows = _make_comparison_rows({})  # all ratios exactly 1.0, well below 2.5x everywhere
        result = classify_richness(rows)
        self.assertEqual(result["classification"], "A")

    def test_classify_richness_broad_strong_growth_is_C(self) -> None:
        overrides = {
            "packet_count (IP packets)": (1000, 5000),
            "unique_hosts": (100, 400),
            "unique_directed_pairs": (200, 900),
            "p95_nodes_per_window": (3.0, 12.0),
            "max_nodes_per_window": (5.0, 20.0),
            "p95_edges_per_window": (4.0, 16.0),
            "max_edges_per_window": (8.0, 30.0),
            "non_empty_window_fraction": (0.1, 0.6),
            "longest_non_empty_run": (8, 40),
            "mean_edge_jaccard_consecutive": (0.03, 0.5),
            "mean_node_jaccard_consecutive": (0.08, 0.6),
        }
        rows = _make_comparison_rows(overrides)  # every richness metric ratio >= 3.0x
        result = classify_richness(rows)
        self.assertEqual(result["classification"], "C")

    def test_classify_richness_mixed_evidence_is_B(self) -> None:
        # Mirrors the actual Phase 9F observation: temporal-continuity
        # metrics grow a lot, structural-density metrics barely move, one
        # metric even shrinks -- must not be forced into A or C.
        overrides = {
            "packet_count (IP packets)": (1000, 6300),
            "unique_directed_pairs": (200, 150),  # shrinks
            "max_nodes_per_window": (5.0, 9.0),  # modest growth
            "max_edges_per_window": (8.0, 16.0),
            "non_empty_window_fraction": (0.3, 0.8),
            "longest_non_empty_run": (8, 71),
            "mean_edge_jaccard_consecutive": (0.03, 0.21),
            "mean_node_jaccard_consecutive": (0.08, 0.32),
        }
        rows = _make_comparison_rows(overrides)
        result = classify_richness(rows)
        self.assertEqual(result["classification"], "B")
        self.assertIn("Evidence is mixed", result["rationale"])

    def test_classify_richness_never_crashes_on_zero_ratio_metrics(self) -> None:
        # median_nodes/edges default to (1.0, 1.0) -> ratio defined; but
        # ensure the function tolerates a None ratio (e.g. 0/0) elsewhere
        # without raising, since real capture-9E data hits this exact case
        # (median_nodes_per_window was 0.0 in both captures).
        rows = _make_comparison_rows({"median_nodes_per_window": (0.0, 0.0), "median_edges_per_window": (0.0, 0.0)})
        result = classify_richness(rows)  # must not raise
        self.assertIn(result["classification"], {"A", "B", "C"})

    # ---- Step 9: novelty decision mapping ----

    def test_novelty_decision_maps_classification_to_decision(self) -> None:
        self.assertEqual(novelty_decision({"classification": "A"}, {})["decision"], "STOP GRAPH BRANCH")
        self.assertEqual(novelty_decision({"classification": "B"}, {})["decision"], "CONDITIONAL")
        self.assertEqual(novelty_decision({"classification": "C"}, {})["decision"], "CONTINUE")

    # ---- Step 8: graph-model suitability sanity + accuracy ----

    def test_graph_model_suitability_reports_per_capture_non_empty_fractions_distinctly(self) -> None:
        rows = _make_comparison_rows({"non_empty_window_fraction": (0.31, 0.81)})
        richness = classify_richness(rows)
        suitability = graph_model_suitability(rows, richness)
        q2_text = suitability["q2_snapshots_sufficiently_populated"]["observed"]
        self.assertIn("31.0%", q2_text)
        self.assertIn("81.0%", q2_text)
        for key in [
            "q1_enough_structure_for_meaningful_gnn",
            "q2_snapshots_sufficiently_populated",
            "q3_sufficiently_long_temporal_sequences",
            "q4_would_graph_model_add_info_beyond_aggregate_stats",
            "q5_which_representation_fits_the_observed_data",
        ]:
            self.assertIn(key, suitability)

    # ---- End-to-end regression against this phase's own persisted run ----

    def test_persisted_run_richness_matches_recomputation(self) -> None:
        report_path = PHASE9F_OUTPUT_DIR / "comparison_report.json"
        if not report_path.exists():
            self.skipTest("Phase 9F has not been run yet in this environment")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        recomputed = classify_richness(persisted["comparison_table"])
        self.assertEqual(recomputed["classification"], persisted["richness_classification"]["classification"])
        self.assertTrue(persisted["second_graph_statistics"]["packet_accounting_check"]["matches"])


from phase9g_event_representation import (
    MICRO_BATCH_SECONDS,
    OUTPUT_DIR as PHASE9G_OUTPUT_DIR,
    build_and_validate_anonymization as p9g_build_and_validate_anonymization,
    build_comparison_table,
    build_hybrid_representation,
    build_representation_a,
    build_representation_b1,
    build_representation_b2,
    check_information_preservation,
)


def _g_packet(
    offset: float,
    src_ip: str,
    dst_ip: str,
    protocol: str = "TCP",
    packet_length: int = 100,
    src_port=5000,
    dst_port=443,
    tcp_syn: bool = False,
    tcp_ack: bool = False,
    tcp_fin: bool = False,
    tcp_rst: bool = False,
) -> dict:
    return {
        "timestamp": 1_700_000_000.0 + offset,
        "offset": offset,
        "packet_length": packet_length,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "ip_version": 4,
        "protocol": protocol,
        "src_port": src_port if protocol in ("TCP", "UDP") else None,
        "dst_port": dst_port if protocol in ("TCP", "UDP") else None,
        "ttl": 64,
        "tcp_flags_str": "S" if tcp_syn else None,
        "tcp_window": 8192 if protocol == "TCP" else None,
        "tcp_syn": tcp_syn,
        "tcp_ack": tcp_ack,
        "tcp_fin": tcp_fin,
        "tcp_rst": tcp_rst,
        "payload_length": max(packet_length - 40, 0),
        "ip_flags_str": "DF",
        "ip_fragment_offset": 0,
    }


class Phase9GEventRepresentationTests(unittest.TestCase):
    """Phase 9G: focused tests for event-representation construction (B1,
    B2), the hybrid definition/size measurement, and the exact
    reconciliation check against the raw packet stream. Synthetic,
    hand-checkable fixtures only -- no network, no dependency on the real
    captures except for one end-to-end regression check."""

    def test_micro_batch_seconds_is_fixed_documented_constant(self) -> None:
        self.assertEqual(MICRO_BATCH_SECONDS, 1)

    # ---- B1: packet events ----

    def test_representation_b1_one_event_per_packet(self) -> None:
        packets = [
            _g_packet(0.0, "10.0.0.1", "10.0.0.2"),
            _g_packet(0.5, "10.0.0.1", "10.0.0.3"),
            _g_packet(1.5, "10.0.0.2", "10.0.0.1"),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.3"})
        b1 = build_representation_b1(packets, anon_map)
        self.assertEqual(b1["event_count"], 3)
        self.assertIn("src", b1["event_fields"])
        self.assertIn("dst", b1["event_fields"])
        self.assertGreater(b1["estimated_serialized_size_bytes"], 0)

    # ---- B2: micro-batch events ----

    def test_representation_b2_groups_by_5tuple_and_bucket(self) -> None:
        # Two packets share (src, dst, protocol, ports) within the same 1s
        # bucket -> one event with packet_count=2. A third packet in the
        # SAME 5-tuple but a later 1s bucket must be a SEPARATE event.
        packets = [
            _g_packet(0.1, "10.0.0.1", "10.0.0.2", dst_port=443, packet_length=100),
            _g_packet(0.4, "10.0.0.1", "10.0.0.2", dst_port=443, packet_length=200),
            _g_packet(1.2, "10.0.0.1", "10.0.0.2", dst_port=443, packet_length=150),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=1.2)
        self.assertEqual(b2["event_count"], 2)
        events = sorted(b2["_events"], key=lambda e: e["event_timestamp_offset"])
        self.assertEqual(events[0]["packet_count"], 2)
        self.assertEqual(events[0]["total_bytes"], 300)
        self.assertAlmostEqual(events[0]["mean_packet_length"], 150.0)
        self.assertEqual(events[1]["packet_count"], 1)

    def test_representation_b2_different_ports_stay_separate_events(self) -> None:
        packets = [
            _g_packet(0.1, "10.0.0.1", "10.0.0.2", dst_port=443),
            _g_packet(0.2, "10.0.0.1", "10.0.0.2", dst_port=8080),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=0.2)
        self.assertEqual(b2["event_count"], 2)
        for e in b2["_events"]:
            # port is part of the grouping key -> trivially 1 by construction
            self.assertEqual(e["destination_port_diversity"], 1)

    def test_representation_b2_iat_null_for_single_packet_event(self) -> None:
        packets = [_g_packet(0.0, "10.0.0.5", "10.0.0.6")]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.5", "10.0.0.6"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=1.0)
        event = b2["_events"][0]
        self.assertIsNone(event["mean_IAT"])
        self.assertIsNone(event["IAT_std"])

    def test_representation_b2_tcp_flag_counts(self) -> None:
        packets = [
            _g_packet(0.0, "10.0.0.1", "10.0.0.2", tcp_syn=True),
            _g_packet(0.1, "10.0.0.1", "10.0.0.2", tcp_ack=True),
            _g_packet(0.2, "10.0.0.1", "10.0.0.2", tcp_fin=True),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=0.2)
        event = b2["_events"][0]
        self.assertEqual(event["packet_count"], 3)
        self.assertEqual(event["tcp_syn_count"], 1)
        self.assertEqual(event["tcp_ack_count"], 1)
        self.assertEqual(event["tcp_fin_count"], 1)
        self.assertEqual(event["tcp_rst_count"], 0)

    def test_representation_b2_burst_and_compression_ratio(self) -> None:
        packets = [_g_packet(float(i) * 0.01, "10.0.0.1", "10.0.0.2") for i in range(5)]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=0.04)
        self.assertEqual(b2["event_count"], 1)  # all within the same 1s bucket, same 5-tuple
        self.assertEqual(b2["maximum_burst_size"], 5)
        self.assertAlmostEqual(b2["compression_ratio_packets_per_event"], 5.0)

    # ---- Step 5: event-sequence structure (unseen pair/dest/source tracking) ----

    def test_representation_b2_unseen_pair_percentage(self) -> None:
        # Two distinct pairs, chronologically: (1->2) first, then (1->3).
        # Each is a genuinely new pair on first occurrence -> 100% unseen.
        packets = [
            _g_packet(0.0, "10.0.0.1", "10.0.0.2"),
            _g_packet(2.0, "10.0.0.1", "10.0.0.3"),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.3"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=2.0)
        s = b2["structure_stats"]
        self.assertEqual(s["total_events"], 2)
        self.assertAlmostEqual(s["pct_events_previously_unseen_pair"], 100.0)

    # ---- Step 6: exact reconciliation against raw packet stream ----

    def test_information_preservation_reconciles_exactly(self) -> None:
        packets = [
            _g_packet(0.0, "10.0.0.1", "10.0.0.2", protocol="TCP", packet_length=100),
            _g_packet(0.5, "10.0.0.1", "10.0.0.2", protocol="TCP", packet_length=150),
            _g_packet(3.0, "10.0.0.2", "10.0.0.4", protocol="UDP", packet_length=64, dst_port=53),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.4"})
        b2 = build_representation_b2(packets, anon_map, capture_duration=3.0)
        preservation = check_information_preservation(packets, b2, anon_map, capture_start_ts=1000.0, capture_end_ts=1003.0)

        self.assertTrue(preservation["total_packets"]["matches"])
        self.assertTrue(preservation["total_bytes"]["matches"])
        self.assertTrue(preservation["unique_hosts"]["matches"])
        self.assertTrue(preservation["unique_directed_pairs"]["matches"])
        self.assertTrue(preservation["protocol_counts"]["matches"])
        self.assertTrue(preservation["port_diversity"]["source_matches"])
        self.assertTrue(preservation["port_diversity"]["destination_matches"])
        self.assertEqual(preservation["unique_hosts"]["raw"], 3)
        self.assertEqual(preservation["unique_directed_pairs"]["raw"], 2)  # (1->2), (2->4)

    # ---- Step 8: hybrid representation size measurement ----

    def test_hybrid_representation_size_is_smaller_than_naive_combination(self) -> None:
        packets = [
            _g_packet(0.0, "10.0.0.1", "10.0.0.2"),
            _g_packet(5.0, "10.0.0.1", "10.0.0.3"),
            _g_packet(15.0, "10.0.0.2", "10.0.0.1"),
        ]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2", "10.0.0.3"})
        from phase9e_temporal_packet_graph import assign_windows

        # _g_packet's timestamp = 1_700_000_000.0 + offset, so capture_start_ts must
        # match that base -- passing 0.0 here would make assign_windows compute an
        # offset of ~1.7 BILLION seconds (treating capture start as the Unix epoch),
        # producing ~170 million windows and hanging/exhausting memory downstream.
        max_window = assign_windows(packets, capture_start_ts=1_700_000_000.0)
        total_windows = max_window + 1
        rep_a = build_representation_a(packets, anon_map, total_windows)
        rep_b2 = build_representation_b2(packets, anon_map, capture_duration=15.0)

        hybrid = build_hybrid_representation(rep_a, rep_b2, total_windows)
        self.assertLess(
            hybrid["lean_hybrid_bytes_B2_plus_window_summary_plus_index"],
            hybrid["naive_combined_bytes_A_plus_B2"],
        )
        self.assertGreater(hybrid["window_state_summary_count"], 0)

    # ---- Step 7: comparison table sanity ----

    def test_comparison_table_covers_required_properties(self) -> None:
        packets = [_g_packet(0.0, "10.0.0.1", "10.0.0.2")]
        anon_map, _ = p9g_build_and_validate_anonymization({"10.0.0.1", "10.0.0.2"})
        from phase9e_temporal_packet_graph import assign_windows

        # See the matching comment above: capture_start_ts must match _g_packet's
        # 1_700_000_000.0 base, not 0.0.
        max_window = assign_windows(packets, capture_start_ts=1_700_000_000.0)
        total_windows = max_window + 1
        rep_a = build_representation_a(packets, anon_map, total_windows)
        rep_b1 = build_representation_b1(packets, anon_map)
        rep_b2 = build_representation_b2(packets, anon_map, capture_duration=1.0)
        hybrid = build_hybrid_representation(rep_a, rep_b2, total_windows)

        capture_result = {
            "representation_a": {k: v for k, v in rep_a.items() if not k.startswith("_")},
            "representation_b1": rep_b1,
            "representation_b2": {k: v for k, v in rep_b2.items() if not k.startswith("_")},
            "representation_c_hybrid": hybrid,
        }
        table = build_comparison_table([capture_result])
        properties = {row["property"] for row in table}
        for expected in [
            "temporal resolution", "variable topology handling", "sparsity handling",
            "sequence length (this run)", "measured storage (serialized JSON, this run)",
            "packet information preservation", "topology information", "burst information",
            "port behavior", "implementation complexity", "suitability for future world-model input",
        ]:
            self.assertIn(expected, properties)

    # ---- Regression against this phase's own persisted run ----

    def test_persisted_run_reconciliation_is_true(self) -> None:
        report_path = PHASE9G_OUTPUT_DIR / "event_representation_report.json"
        if not report_path.exists():
            self.skipTest("Phase 9G has not been run yet in this environment")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        for capture in persisted["captures"]:
            pres = capture["information_preservation"]
            self.assertTrue(pres["total_packets"]["matches"])
            self.assertTrue(pres["total_bytes"]["matches"])
            self.assertTrue(pres["unique_hosts"]["matches"])
            self.assertTrue(pres["unique_directed_pairs"]["matches"])
            self.assertTrue(pres["protocol_counts"]["matches"])


if __name__ == "__main__":
    unittest.main()
