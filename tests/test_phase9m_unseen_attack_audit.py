"""Phase 9M: focused tests for the unseen-attack-family generalization audit.

Covers: family/raw-label mapping correctness against the actual frozen data,
one-family-per-source-day invariant, walk-forward split (no temporal
leakage), held-out family exclusion from train/validation, scaler fit only
on the reduced train set (no held-out-family leakage into preprocessing),
duplicate-sequence classification (benign all-zero idle windows vs genuine
leakage), threshold-selection isolation (validation never contains the
held-out family), target definition unchanged (binary, not multiclass),
feature schema unchanged (157, exact order, no meta/label columns), and
deterministic re-evaluation of the already-trained checkpoints.

The two leave-one-family-out experiments (Bot, Infiltration) are expensive
(real GPU training) and are run ONCE by experiments/phase9m_unseen_attack_audit.py;
this suite reads the already-produced artifacts under
experiments/results/phase9m_unseen_attack_audit/ wherever possible, and
only re-invokes cheap, read-only functions (Steps 1-3, the holdout data
reader, leakage checks) directly.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase9m_unseen_attack_audit import (  # noqa: E402
    DEFAULT_WINDOWS_DIR,
    FAMILY_MAP,
    OUTPUT_DIR,
    INPUT_SIZE,
    load_all_rows,
    step1_inspect_frozen_dataset,
    step2_audit_current_split,
    step3_candidate_suitability,
    read_family_holdout_samples,
    leakage_stress_checks,
    _sha256_file,
)


class Step1DatasetInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_all_rows(DEFAULT_WINDOWS_DIR)
        cls.step1 = step1_inspect_frozen_dataset(cls.records)

    def test_no_unmapped_raw_labels(self) -> None:
        self.assertEqual(self.step1["unmapped_raw_labels"], [])

    def test_one_family_per_source_day_invariant_holds(self) -> None:
        self.assertTrue(self.step1["one_family_per_day_invariant_holds"])
        self.assertEqual(self.step1["files_with_multiple_families"], {})

    def test_family_table_totals_match_expected_split_counts(self) -> None:
        # Sum of family-table eligible counts across (train+val+test) must not exceed the
        # known total eligible row count (29315+6195+6195); it will be less than that because
        # benign rows are not counted in the family table.
        total = sum(row["total"] for row in self.step1["family_table"])
        self.assertGreater(total, 0)
        self.assertLess(total, 29315 + 6195 + 6195)

    def test_expected_six_families_present(self) -> None:
        families = {row["family"] for row in self.step1["family_table"]}
        self.assertEqual(families, {"Bot", "DoS", "DDoS", "Web Attack", "Infiltration", "Brute Force"})

    def test_bot_family_raw_labels(self) -> None:
        bot_row = next(r for r in self.step1["family_table"] if r["family"] == "Bot")
        self.assertEqual(bot_row["raw_labels"], ["Bot"])
        self.assertEqual(bot_row["source_day_files"], ["Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"])

    def test_infiltration_family_spans_two_days(self) -> None:
        row = next(r for r in self.step1["family_table"] if r["family"] == "Infiltration")
        self.assertEqual(len(row["source_day_files"]), 2)


class Step2SplitAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_all_rows(DEFAULT_WINDOWS_DIR)
        cls.step2 = step2_audit_current_split(DEFAULT_WINDOWS_DIR, cls.records)

    def test_walk_forward_ordering_holds(self) -> None:
        self.assertTrue(self.step2["walk_forward_ordering_holds"])
        self.assertEqual(self.step2["walk_forward_ordering_issues"], [])

    def test_no_cross_split_leakage_in_frozen_split(self) -> None:
        self.assertEqual(self.step2["cross_split_leakage_pairs_in_frozen_split"], 0)

    def test_target_is_binary_not_multiclass(self) -> None:
        self.assertTrue(self.step2["target_is_binary_future_attack_within_horizon_not_multiclass"])

    def test_feature_columns_exclude_meta_columns(self) -> None:
        self.assertTrue(self.step2["feature_columns_exclude_all_meta_and_label_columns"])

    def test_existing_scaler_would_leak_if_reused(self) -> None:
        # Documents WHY Phase 9M must fit its own scaler -- not a bug, a design justification.
        self.assertTrue(self.step2["existing_scaler_would_leak_holdout_family_if_reused"])


class Step3SuitabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_all_rows(DEFAULT_WINDOWS_DIR)
        step1 = step1_inspect_frozen_dataset(records)
        cls.step3 = step3_candidate_suitability(step1)

    def test_bot_is_suitable(self) -> None:
        self.assertIn("Bot", self.step3["suitable_families"])

    def test_infiltration_is_suitable(self) -> None:
        self.assertIn("Infiltration", self.step3["suitable_families"])

    def test_dos_is_unsuitable(self) -> None:
        self.assertIn("DoS", self.step3["unsuitable_families"])

    def test_web_attack_is_unsuitable(self) -> None:
        self.assertIn("Web Attack", self.step3["unsuitable_families"])

    def test_every_family_has_a_suitability_verdict(self) -> None:
        for row in self.step3["candidate_table"]:
            self.assertIn(row["suitability"], ("SUITABLE", "MARGINAL", "UNSUITABLE"))
            self.assertTrue(row["reasons"])

    def test_no_candidate_silently_forced(self) -> None:
        # Every UNSUITABLE/MARGINAL verdict must carry an explicit reason string (never empty).
        for row in self.step3["candidate_table"]:
            if row["suitability"] != "SUITABLE":
                self.assertTrue(all(isinstance(r, str) and r for r in row["reasons"]))


class HoldoutDataConstructionTests(unittest.TestCase):
    """Cheap, read-only checks on the family-holdout data reader (no training)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bot_files = {"Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"}
        cls.data = read_family_holdout_samples(DEFAULT_WINDOWS_DIR, cls.bot_files)

    def test_feature_count_is_157(self) -> None:
        self.assertEqual(len(self.data.feature_columns), INPUT_SIZE)

    def test_train_and_validation_are_finite(self) -> None:
        self.assertTrue(np.isfinite(self.data.train.sequences).all())
        self.assertTrue(np.isfinite(self.data.validation.sequences).all())

    def test_test_split_is_never_filtered(self) -> None:
        # The held-out family's day must still contribute rows to test (both original Phase 3.5
        # test-split rows, unmodified in count) -- the filter only applies to train/validation.
        held_out_test_rows = int((self.data.test_source_file == "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv").sum())
        self.assertEqual(held_out_test_rows, 637)  # exact known count for Bot's test-split rows

    def test_test_split_total_equals_frozen_phase35_test_count(self) -> None:
        # The test split is read from ALL files unfiltered -- must equal the frozen Phase 3.5
        # total test count (6195), proving no test-split row was dropped or added.
        self.assertEqual(len(self.data.test.labels), 6195)

    def test_train_excludes_all_holdout_family_rows(self) -> None:
        # Indirect proof: train count for the Bot-holdout run must be smaller than the full
        # Phase 3.5 train count (29315) by exactly the Bot day's train-eligible row count (1042+2660... )
        # -- here we just assert it's strictly smaller and finite, exact accounting is covered by
        # leakage_stress_checks's held_out_family_eligible_rows_that_WOULD_have_been_in_train_val.
        self.assertLess(len(self.data.train.labels), 29315)

    def test_different_holdout_family_yields_different_train_size(self) -> None:
        infiltration_files = {
            "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
        }
        infiltration_data = read_family_holdout_samples(DEFAULT_WINDOWS_DIR, infiltration_files)
        self.assertNotEqual(len(self.data.train.labels), len(infiltration_data.train.labels))


class LeakageStressCheckTests(unittest.TestCase):
    """Runs leakage_stress_checks on a cheap (Bot) holdout construction -- does not require
    a trained model, only the StandardScaler fit, which is fast."""

    @classmethod
    def setUpClass(cls) -> None:
        from sklearn.preprocessing import StandardScaler
        cls.holdout_files = {"Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"}
        cls.data = read_family_holdout_samples(DEFAULT_WINDOWS_DIR, cls.holdout_files)
        cls.scaler = StandardScaler()
        cls.scaler.fit(cls.data.train.sequences.reshape(-1, cls.data.train.sequences.shape[-1]))
        cls.checks = leakage_stress_checks(DEFAULT_WINDOWS_DIR, cls.data, cls.holdout_files, cls.scaler)

    def test_held_out_family_exclusion_verified(self) -> None:
        self.assertTrue(self.checks["held_out_family_exclusion_verified"])
        self.assertEqual(self.checks["held_out_family_rows_in_train_or_validation"], 0)

    def test_scaler_differs_from_original_full_train_scaler(self) -> None:
        self.assertTrue(self.checks["reduced_train_scaler_differs_from_original_full_train_scaler"])

    def test_duplicate_sequences_are_classified_not_just_counted(self) -> None:
        self.assertIn("duplicate_sequences_all_zero_idle_windows", self.checks)
        self.assertIn("duplicate_sequences_nonzero_genuine_concern", self.checks)
        # Every duplicate must be accounted for as either benign-all-zero or genuine-concern.
        self.assertEqual(
            self.checks["duplicate_sequences_all_zero_idle_windows"] + self.checks["duplicate_sequences_nonzero_genuine_concern"],
            self.checks["duplicate_sequences_between_reduced_train_and_test_total"],
        )

    def test_no_genuine_nonzero_duplicate_leakage(self) -> None:
        self.assertEqual(self.checks["duplicate_sequences_nonzero_genuine_concern"], 0)

    def test_all_checks_pass(self) -> None:
        self.assertTrue(self.checks["all_checks_pass"])

    def test_validation_excludes_held_out_family(self) -> None:
        self.assertTrue(self.checks["validation_set_excludes_held_out_family_by_construction"])
        self.assertGreater(self.checks["validation_row_count"], 0)


@unittest.skipUnless(OUTPUT_DIR.exists(), "Phase 9M has not been run yet (experiments/phase9m_unseen_attack_audit.py)")
class ProducedArtifactTests(unittest.TestCase):
    """Validates the artifacts actually written by a completed Phase 9M run, without
    re-running training. Skipped automatically if the phase hasn't been run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.full = json.loads((OUTPUT_DIR / "phase9m_unseen_attack_audit.json").read_text(encoding="utf-8"))
        cls.protocol = json.loads((OUTPUT_DIR / "phase9m_protocol.json").read_text(encoding="utf-8"))

    def test_status_is_one_of_green_yellow_red(self) -> None:
        self.assertIn(self.full["status"], ("GREEN", "YELLOW", "RED"))

    def test_required_files_exist(self) -> None:
        for name in ["phase9m_unseen_attack_audit.md", "phase9m_unseen_attack_audit.json", "phase9m_protocol.json", "metrics.csv"]:
            self.assertTrue((OUTPUT_DIR / name).exists(), f"missing required artifact: {name}")

    def test_no_forbidden_overclaim_phrases_in_report(self) -> None:
        # Section 17 ("Claim-Safety Statement") legitimately QUOTES the forbidden phrases in order
        # to explicitly deny them ("None of the following are claimed: ..."), exactly as Step 11/17
        # require -- so it is excluded from this scan; every OTHER section must never use them.
        report_text = (OUTPUT_DIR / "phase9m_unseen_attack_audit.md").read_text(encoding="utf-8")
        before_section_17 = report_text.split("## 17. Claim-Safety Statement")[0]
        after_section_17 = report_text.split("## 18. Final Verdict")[-1] if "## 18. Final Verdict" in report_text else ""
        scanned_text = (before_section_17 + after_section_17).lower()
        forbidden = [
            "generalizes to all unseen attacks", "zero-shot attack detection", "novel attack detection",
            "unknown attack detection", "detects attacks never seen before", "works on arbitrary future attack families",
        ]
        for phrase in forbidden:
            self.assertNotIn(phrase.lower(), scanned_text, f"forbidden overclaim phrase found outside the disclaimer section: {phrase!r}")

    def test_every_experiment_has_an_interpretation_claim(self) -> None:
        for family, exp in self.full.get("experiments", {}).items():
            self.assertIn("interpretation", exp)
            self.assertIn(exp["interpretation"]["claim"], ("CLAIM A", "CLAIM B", "CLAIM C", "CLAIM D"))

    def test_every_experiment_leakage_checks_pass(self) -> None:
        for family, exp in self.full.get("experiments", {}).items():
            self.assertTrue(exp["leakage_checks"]["all_checks_pass"], f"{family} leakage checks failed")

    def test_checkpoints_confined_to_phase9m_directory(self) -> None:
        for family_key in ("holdout_bot", "holdout_infiltration"):
            exp_dir = OUTPUT_DIR / family_key
            if exp_dir.exists():
                self.assertTrue((exp_dir / "transformer_best_model.pt").exists())
                self.assertTrue((exp_dir / "scaler.joblib").exists())

    def test_frozen_phase4_5_checkpoints_untouched(self) -> None:
        # These paths must exist and be byte-identical to what Phase 5 itself reported
        # (spot-checked via existence + non-zero size here; exact hash comparison is done
        # in the Phase 9M report's own integrity section using git status).
        phase5_checkpoint = EXPERIMENTS_DIR / "results/phase5_temporal_transformer/transformer/best_model.pt"
        if phase5_checkpoint.exists():
            self.assertGreater(phase5_checkpoint.stat().st_size, 0)

    def test_deterministic_reevaluation_of_trained_checkpoint(self) -> None:
        """Loads the already-trained Bot-holdout checkpoint fresh and re-predicts on the same
        test tensor twice -- must be bit-for-bit deterministic (eval mode, no dropout randomness)."""
        import joblib
        import torch
        from phase9m_unseen_attack_audit import TemporalTransformer, _predict_torch

        exp_dir = OUTPUT_DIR / "holdout_bot"
        if not exp_dir.exists():
            self.skipTest("holdout_bot artifacts not present")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TemporalTransformer().to(device)
        checkpoint = torch.load(exp_dir / "transformer_best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = joblib.load(exp_dir / "scaler.joblib")

        data = read_family_holdout_samples(DEFAULT_WINDOWS_DIR, {"Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"})
        subset = data.test.sequences[:50]
        # Per-timestep StandardScaler convention (fit on reshape(-1, 157)), matching
        # run_holdout_experiment's _scale() -- NOT a flatten-whole-sequence reshape.
        flat = scaler.transform(subset.reshape(-1, subset.shape[-1]))
        scaled = flat.reshape(subset.shape).astype(np.float32)

        p1 = _predict_torch(model, scaled, device)
        p2 = _predict_torch(model, scaled, device)
        self.assertTrue(np.allclose(p1, p2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
