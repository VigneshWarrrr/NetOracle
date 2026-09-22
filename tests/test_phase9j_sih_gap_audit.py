"""Phase 9J: focused tests for the SIH compliance + scientific gap audit.

Covers requirement-matrix schema, valid status/priority/verdict values,
roadmap priority values, claims-audit verdict values, JSON/CSV/Markdown
consistency, and no-prior-artifact-mutation. Pure data-structure and
regression checks -- no network access, no training.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase9j_sih_gap_audit import (  # noqa: E402
    CLAIMS_AUDIT,
    NOVELTY_MATRIX,
    OUTPUT_DIR,
    REQUIREMENT_MATRIX,
    ROADMAP,
    VALID_CLAIM_VERDICTS,
    VALID_NOVELTY_CLASSES,
    VALID_REQUIREMENT_STATUSES,
    VALID_ROADMAP_PRIORITIES,
    overall_verdict,
    status_counts,
    verify_no_prior_artifact_changed,
)

REQUIREMENT_MATRIX_REQUIRED_COLUMNS = {
    "ID", "PS Requirement", "Exact/faithful PS interpretation",
    "Current NetOracle implementation", "Evidence/artifact", "Status",
    "Scientific confidence", "Missing work", "Priority",
}
ROADMAP_REQUIRED_COLUMNS = {
    "Priority", "Task", "Purpose", "Expected evidence", "Dependencies", "Risk", "Requires new data",
}
CLAIMS_AUDIT_REQUIRED_COLUMNS = {"CLAIM", "CAN WE SAY THIS?", "EVIDENCE", "CONFIDENCE", "REQUIRED QUALIFIER"}


class RequirementMatrixSchemaTests(unittest.TestCase):
    def test_every_row_has_required_columns(self) -> None:
        for row in REQUIREMENT_MATRIX:
            missing = REQUIREMENT_MATRIX_REQUIRED_COLUMNS - set(row.keys())
            self.assertFalse(missing, f"row {row.get('ID')} missing columns: {missing}")

    def test_every_status_is_valid(self) -> None:
        for row in REQUIREMENT_MATRIX:
            self.assertIn(row["Status"], VALID_REQUIREMENT_STATUSES, f"{row['ID']}: {row['Status']!r}")

    def test_every_priority_is_valid_roadmap_priority(self) -> None:
        for row in REQUIREMENT_MATRIX:
            self.assertIn(row["Priority"], VALID_ROADMAP_PRIORITIES, f"{row['ID']}: {row['Priority']!r}")

    def test_ids_are_unique(self) -> None:
        ids = [row["ID"] for row in REQUIREMENT_MATRIX]
        self.assertEqual(len(ids), len(set(ids)), "duplicate requirement IDs found")

    def test_no_row_is_empty_or_placeholder(self) -> None:
        for row in REQUIREMENT_MATRIX:
            for key, value in row.items():
                self.assertTrue(str(value).strip(), f"{row['ID']}.{key} is empty")
                self.assertNotIn("TODO", str(value))
                self.assertNotIn("TBD", str(value))


class NoveltyMatrixTests(unittest.TestCase):
    def test_every_recommendation_starts_with_a_valid_class_letter(self) -> None:
        for row in NOVELTY_MATRIX:
            rec = row["RECOMMENDATION"]
            first_token = rec.split(" ", 1)[0].rstrip(".")
            self.assertIn(first_token, VALID_NOVELTY_CLASSES, f"{row['FEATURE']}: {rec!r}")

    def test_no_recommendation_is_A_merely_because_it_sounds_advanced(self) -> None:
        # Regression guard: none of the explicitly novelty-theater-risk items
        # (RL, LLM chatbot, GAN, federated learning) may be classified A or B.
        risky_features = {"Reinforcement learning", "LLM chatbot", "GAN / synthetic data", "Federated learning"}
        for row in NOVELTY_MATRIX:
            if row["FEATURE"] in risky_features:
                self.assertTrue(
                    row["RECOMMENDATION"].startswith("D"),
                    f"{row['FEATURE']} should be classified D (reject), got: {row['RECOMMENDATION']!r}",
                )

    def test_world_model_and_transformer_are_not_rejected(self) -> None:
        # Regression guard: the two capabilities that ARE genuinely validated
        # and PS-required must never be classified D.
        core_features = {"Temporal Transformer", "World Model (learned state-transition dynamics)"}
        for row in NOVELTY_MATRIX:
            if row["FEATURE"] in core_features:
                self.assertFalse(row["RECOMMENDATION"].startswith("D"), row["FEATURE"])


class RoadmapSchemaTests(unittest.TestCase):
    def test_every_row_has_required_columns(self) -> None:
        for row in ROADMAP:
            missing = ROADMAP_REQUIRED_COLUMNS - set(row.keys())
            self.assertFalse(missing, f"row {row.get('Task')} missing columns: {missing}")

    def test_every_priority_is_valid(self) -> None:
        for row in ROADMAP:
            self.assertIn(row["Priority"], VALID_ROADMAP_PRIORITIES, f"{row['Task']}: {row['Priority']!r}")

    def test_requires_new_data_is_boolean(self) -> None:
        for row in ROADMAP:
            self.assertIsInstance(row["Requires new data"], bool, row["Task"])

    def test_packet_flow_fusion_is_rejected(self) -> None:
        # Regression guard: this was explicitly tested and falsified twice
        # (Phase 9H RED, Phase 9I RED) -- the roadmap must not silently
        # re-propose it at any P0/P1/P2 priority.
        fusion_rows = [r for r in ROADMAP if "fusion" in r["Task"].lower() or "gnn" in r["Task"].lower()]
        self.assertTrue(fusion_rows, "expected at least one fusion/GNN roadmap row")
        for row in fusion_rows:
            self.assertEqual(row["Priority"], "REJECT", row["Task"])

    def test_p0_tasks_do_not_require_new_data(self) -> None:
        # P0 = must-fix-now items; if they required new data acquisition they
        # could not actually be executed immediately, contradicting P0.
        for row in ROADMAP:
            if row["Priority"] == "P0":
                self.assertFalse(row["Requires new data"], row["Task"])


class ClaimsAuditTests(unittest.TestCase):
    def test_every_row_has_required_columns(self) -> None:
        for row in CLAIMS_AUDIT:
            missing = CLAIMS_AUDIT_REQUIRED_COLUMNS - set(row.keys())
            self.assertFalse(missing, f"row {row.get('CLAIM')} missing columns: {missing}")

    def test_every_verdict_is_valid(self) -> None:
        for row in CLAIMS_AUDIT:
            self.assertIn(row["CAN WE SAY THIS?"], VALID_CLAIM_VERDICTS, f"{row['CLAIM']}: {row['CAN WE SAY THIS?']!r}")

    def test_known_overclaims_are_not_supported(self) -> None:
        # Regression guard: these specific claims were explicitly flagged as
        # false/unsupported by frozen prior-phase evidence (Phase 7B/7C/7D/8B/9H/9I).
        must_not_be_supported = {
            "Uses SHAP",
            "Uses packet-level information",
            "Uses a network graph",
            "Learns attacker kill-chain progression",
            "Provides calibrated probabilities",
            "Generalizes to unseen attacks",
        }
        by_claim = {row["CLAIM"]: row for row in CLAIMS_AUDIT}
        for claim in must_not_be_supported:
            self.assertIn(claim, by_claim)
            self.assertEqual(by_claim[claim]["CAN WE SAY THIS?"], "NOT SUPPORTED", claim)

    def test_offline_claim_is_supported(self) -> None:
        by_claim = {row["CLAIM"]: row for row in CLAIMS_AUDIT}
        self.assertEqual(by_claim["Works offline"]["CAN WE SAY THIS?"], "SUPPORTED")

    def test_outperforms_baselines_requires_naming_the_model(self) -> None:
        by_claim = {row["CLAIM"]: row for row in CLAIMS_AUDIT}
        qualifier = by_claim["Outperforms baselines"]["REQUIRED QUALIFIER"].lower()
        self.assertIn("transformer", qualifier)


class VerdictLogicTests(unittest.TestCase):
    def test_status_counts_sum_to_total(self) -> None:
        counts = status_counts(REQUIREMENT_MATRIX)
        self.assertEqual(sum(counts.values()), len(REQUIREMENT_MATRIX))

    def test_overall_verdict_is_valid(self) -> None:
        counts = status_counts(REQUIREMENT_MATRIX)
        verdict = overall_verdict(counts)
        self.assertIn(verdict, VALID_REQUIREMENT_STATUSES)

    def test_overall_verdict_red_when_reds_dominate(self) -> None:
        self.assertEqual(overall_verdict({"GREEN": 1, "YELLOW": 1, "RED": 8}), "RED")

    def test_overall_verdict_green_when_all_green(self) -> None:
        self.assertEqual(overall_verdict({"GREEN": 10, "YELLOW": 0, "RED": 0}), "GREEN")

    def test_overall_verdict_yellow_for_mixed_evidence(self) -> None:
        self.assertEqual(overall_verdict({"GREEN": 4, "YELLOW": 4, "RED": 2}), "YELLOW")


class NoArtifactMutationTests(unittest.TestCase):
    def test_verify_no_prior_artifact_changed_detects_mismatch(self) -> None:
        baseline = {"prior_result_directory_file_counts": {"phase4_baseline": 999999}}
        audit = verify_no_prior_artifact_changed(baseline)
        self.assertFalse(audit["unchanged"])
        self.assertIn("phase4_baseline", audit["mismatches"])


class PersistedRunRegressionTests(unittest.TestCase):
    """Schema, budget, and consistency checks against this phase's own
    persisted output. Skipped entirely if the phase has not been run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.json_path = OUTPUT_DIR / "sih_gap_audit_report.json"
        cls.md_path = OUTPUT_DIR / "sih_gap_audit_report.md"
        cls.req_csv = OUTPUT_DIR / "requirement_matrix.csv"
        cls.roadmap_csv = OUTPUT_DIR / "roadmap.csv"
        cls.claims_csv = OUTPUT_DIR / "claims_audit.csv"
        cls.available = all(
            p.exists() for p in (cls.json_path, cls.md_path, cls.req_csv, cls.roadmap_csv, cls.claims_csv)
        )
        if cls.available:
            cls.report = json.loads(cls.json_path.read_text(encoding="utf-8"))
            cls.md_text = cls.md_path.read_text(encoding="utf-8")

    def setUp(self) -> None:
        if not self.available:
            self.skipTest("Phase 9J has not been run yet in this environment")

    def test_json_verdict_is_valid(self) -> None:
        self.assertIn(self.report["overall_verdict"], VALID_REQUIREMENT_STATUSES)

    def test_json_verdict_appears_in_markdown(self) -> None:
        self.assertIn(f"Overall project status: **{self.report['overall_verdict']}**", self.md_text)

    def test_csv_row_count_matches_json(self) -> None:
        with self.req_csv.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), len(self.report["requirement_matrix"]))
        csv_ids = {row["ID"] for row in csv_rows}
        json_ids = {row["ID"] for row in self.report["requirement_matrix"]}
        self.assertEqual(csv_ids, json_ids)

    def test_roadmap_csv_matches_json(self) -> None:
        with self.roadmap_csv.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), len(self.report["roadmap"]))

    def test_claims_csv_matches_json(self) -> None:
        with self.claims_csv.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), len(self.report["claims_audit"]))

    def test_no_prior_artifact_mutation_recorded(self) -> None:
        audit = verify_no_prior_artifact_changed(self.report["repo_facts"])
        self.assertTrue(audit["unchanged"], f"prior artifacts changed: {audit['mismatches']}")

    def test_audited_checkpoints_confirmed_present(self) -> None:
        for path, present in self.report["repo_facts"]["audited_checkpoints_present"].items():
            self.assertTrue(present, f"expected audited checkpoint present: {path}")

    def test_unaudited_checkpoints_confirmed_absent(self) -> None:
        for path, absent in self.report["repo_facts"]["unaudited_track_checkpoints_absent"].items():
            self.assertTrue(absent, f"expected unaudited checkpoint absent: {path}")

    def test_dual_implementation_finding_present_in_report(self) -> None:
        self.assertIn("dual_implementation_finding", self.report)
        self.assertIn("Track A", self.md_text.replace("**", ""))
        self.assertIn("Track B", self.md_text.replace("**", ""))


if __name__ == "__main__":
    unittest.main()
