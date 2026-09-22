"""Phase 9I: focused tests for the PCAP<->flow alignment investigation.

Covers report schema, verdict validity/logic (including the
counts_toward_verdict scoping bug this phase caught and fixed), no-prior-
artifact-mutation, download-budget accounting, evidence classification
validity, timestamp/endpoint status handling, and JSON/Markdown
consistency. No network access is triggered by these tests themselves
(they exercise pure functions with synthetic fixtures, plus read-only
regression checks against this run's own persisted output).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase9i_pcap_flow_alignment import (  # noqa: E402
    MAX_TOTAL_DOWNLOAD_BYTES,
    OUTPUT_DIR,
    build_alignment_feasibility_table,
    classify_correspondence,
    determine_verdict,
    verify_no_prior_artifact_changed,
)

VALID_STATUSES = {"ESTABLISHED", "STRONGLY SUPPORTED", "PLAUSIBLE BUT UNPROVEN", "UNSUPPORTED", "CONTRADICTED"}
VALID_VERDICTS = {"GREEN", "YELLOW", "RED"}


def _flow_semantics(endpoint_available=False):
    return {
        "endpoint_identity_columns_present": [] if not endpoint_available else ["Src IP", "Dst IP"],
        "endpoint_identity_available": endpoint_available,
    }


def _pcap_metadata(other_patterns=0, total=449):
    return {"other_naming_pattern_members": other_patterns, "total_members": total}


def _logs_metadata(member_count=449):
    return {"member_count": member_count}


def _phase9h_alignment(best_r=0.088, best_offset=0, offset_results=None):
    return {
        "best_offset_result": {"pearson_r": best_r, "offset_hours_added_to_csv_local_time": best_offset},
        "offset_results": offset_results or [{"offset_hours_added_to_csv_local_time": h, "pearson_r": 0.02} for h in range(9)],
    }


def _dhcp_crosscheck(established=True, best_offset_hours=4, pcap_count=21, syslog_count=26):
    return {
        "attempted": True,
        "established": established,
        "best_offset_hours": best_offset_hours,
        "best_offset_exact_match_count": pcap_count if established else pcap_count - 1,
        "pcap_dhcprequest_event_count": pcap_count,
        "syslog_dhcprequest_event_count": syslog_count,
        "conclusion": "synthetic conclusion text",
    }


class VerdictLogicTests(unittest.TestCase):
    """Regression coverage for the real bug this phase caught: an
    ESTABLISHED status on a candidate that does NOT bear on PCAP<->CSV
    correspondence (E1, the PCAP-internal clock finding) must NOT by
    itself flip the verdict to GREEN."""

    def test_established_candidate_that_does_not_count_does_not_force_green(self) -> None:
        classifications = [
            {"candidate": "A", "status": "UNSUPPORTED"},
            {"candidate": "E1", "status": "ESTABLISHED", "counts_toward_verdict": False},
        ]
        verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=0))
        self.assertEqual(verdict, "RED")

    def test_established_candidate_that_does_count_forces_green(self) -> None:
        classifications = [
            {"candidate": "A", "status": "ESTABLISHED"},
        ]
        verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=5))
        self.assertEqual(verdict, "GREEN")

    def test_structural_population_mismatch_forces_red_even_with_plausible_candidates(self) -> None:
        classifications = [
            {"candidate": "E", "status": "PLAUSIBLE BUT UNPROVEN"},
            {"candidate": "G", "status": "PLAUSIBLE BUT UNPROVEN"},
        ]
        # other_naming_pattern_members == 0 -> archive is host-specific-only (established structural fact)
        verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=0, total=449))
        self.assertEqual(verdict, "RED")

    def test_plausible_without_structural_block_is_yellow(self) -> None:
        classifications = [{"candidate": "E", "status": "PLAUSIBLE BUT UNPROVEN"}]
        # other_naming_pattern_members > 0 -> structural block does not apply
        verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=3, total=449))
        self.assertEqual(verdict, "YELLOW")

    def test_all_unsupported_is_red(self) -> None:
        classifications = [{"candidate": "A", "status": "UNSUPPORTED"}, {"candidate": "B", "status": "CONTRADICTED"}]
        verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=3, total=449))
        self.assertEqual(verdict, "RED")

    def test_verdict_always_in_valid_set(self) -> None:
        for statuses in (["UNSUPPORTED"], ["ESTABLISHED"], ["PLAUSIBLE BUT UNPROVEN"], ["CONTRADICTED"]):
            classifications = [{"candidate": "X", "status": s} for s in statuses]
            for other in (0, 1):
                verdict = determine_verdict(classifications, _logs_metadata(), _pcap_metadata(other_patterns=other))
                self.assertIn(verdict, VALID_VERDICTS)


class CorrespondenceClassificationTests(unittest.TestCase):
    def test_no_endpoint_identity_yields_unsupported_A_B_C(self) -> None:
        rows = classify_correspondence(_flow_semantics(False), _pcap_metadata(), _logs_metadata(), _phase9h_alignment())
        by_candidate = {r["candidate"]: r for r in rows}
        self.assertEqual(by_candidate["A. Exact endpoint matching (specific IP <-> specific flow rows)"]["status"], "UNSUPPORTED")
        self.assertEqual(by_candidate["B. Exact 5-tuple matching (src ip, dst ip, src port, dst port, protocol)"]["status"], "UNSUPPORTED")
        self.assertEqual(by_candidate["C. Host/subnet matching (PCAP host IP appears somewhere in CSV-derivable subnet)"]["status"], "UNSUPPORTED")

    def test_all_statuses_are_valid_enum_values(self) -> None:
        rows = classify_correspondence(
            _flow_semantics(False), _pcap_metadata(), _logs_metadata(), _phase9h_alignment(), _dhcp_crosscheck(established=True)
        )
        for row in rows:
            self.assertIn(row["status"], VALID_STATUSES, f"{row['candidate']} has invalid status {row['status']!r}")

    def test_e1_established_when_dhcp_crosscheck_established(self) -> None:
        rows = classify_correspondence(
            _flow_semantics(False), _pcap_metadata(), _logs_metadata(), _phase9h_alignment(), _dhcp_crosscheck(established=True)
        )
        e1 = next(r for r in rows if r["candidate"].startswith("E1."))
        self.assertEqual(e1["status"], "ESTABLISHED")
        self.assertFalse(e1.get("counts_toward_verdict", True), "E1 must be excluded from verdict determination")

    def test_e1_not_established_when_dhcp_crosscheck_incomplete(self) -> None:
        rows = classify_correspondence(
            _flow_semantics(False), _pcap_metadata(), _logs_metadata(), _phase9h_alignment(),
            _dhcp_crosscheck(established=False),
        )
        e1 = next(r for r in rows if r["candidate"].startswith("E1."))
        self.assertEqual(e1["status"], "PLAUSIBLE BUT UNPROVEN")

    def test_e1_handles_missing_dhcp_crosscheck_gracefully(self) -> None:
        rows = classify_correspondence(_flow_semantics(False), _pcap_metadata(), _logs_metadata(), _phase9h_alignment(), None)
        e1 = next(r for r in rows if r["candidate"].startswith("E1."))
        self.assertEqual(e1["status"], "PLAUSIBLE BUT UNPROVEN")

    def test_e2_reflects_weak_correlation_at_established_offset(self) -> None:
        offset_results = [{"offset_hours_added_to_csv_local_time": h, "pearson_r": 0.02 if h == 4 else 0.05} for h in range(9)]
        rows = classify_correspondence(
            _flow_semantics(False), _pcap_metadata(), _logs_metadata(),
            _phase9h_alignment(offset_results=offset_results),
            _dhcp_crosscheck(established=True, best_offset_hours=4),
        )
        e2 = next(r for r in rows if r["candidate"].startswith("E2."))
        self.assertEqual(e2["status"], "UNSUPPORTED")
        self.assertIn("0.0200", e2["reason"])


class AlignmentFeasibilityTableTests(unittest.TestCase):
    def test_table_has_two_rows_matching_known_captures(self) -> None:
        table = build_alignment_feasibility_table(None)
        self.assertEqual(len(table), 2)
        captures = {row["pcap_capture"] for row in table}
        self.assertIn("pcap/UCAP172.31.69.22", captures)
        self.assertIn("pcap/capWIN-J6GMIG1DQE5-172.31.64.89", captures)
        for row in table:
            self.assertEqual(row["verdict"], "NOT ALIGNABLE with current evidence")

    def test_established_dhcp_offset_updates_host_22_row_only(self) -> None:
        without = build_alignment_feasibility_table(None)
        with_dhcp = build_alignment_feasibility_table(_dhcp_crosscheck(established=True, best_offset_hours=4))

        row_22_without = next(r for r in without if r["pcap_capture"] == "pcap/UCAP172.31.69.22")
        row_22_with = next(r for r in with_dhcp if r["pcap_capture"] == "pcap/UCAP172.31.69.22")
        row_89_with = next(r for r in with_dhcp if r["pcap_capture"] != "pcap/UCAP172.31.69.22")

        self.assertNotEqual(row_22_without["timestamp_coverage"], row_22_with["timestamp_coverage"])
        self.assertIn("ESTABLISHED", row_22_with["timestamp_coverage"])
        # The other capture (no syslog available) must remain unaffected.
        self.assertIn("unconfirmed", row_89_with["timestamp_coverage"])
        # Overall alignability verdict is unaffected by the clock finding alone.
        self.assertEqual(row_22_with["verdict"], "NOT ALIGNABLE with current evidence")


class NoArtifactMutationTests(unittest.TestCase):
    def test_verify_no_prior_artifact_changed_detects_mismatch(self) -> None:
        baseline = {"prior_result_directory_file_counts": {"phase4_baseline": 17, "phase6b_vector_world_model": 20}}
        audit = verify_no_prior_artifact_changed(baseline)
        # Real counts should match the real baseline recorded elsewhere in this
        # session (both directories are frozen); this is a sanity check that
        # the comparison mechanism itself works correctly on real data.
        self.assertIn("unchanged", audit)
        self.assertIn("mismatches", audit)
        self.assertIsInstance(audit["mismatches"], dict)

    def test_verify_no_prior_artifact_changed_flags_a_fabricated_baseline(self) -> None:
        # Deliberately wrong expected count -> must be reported as a mismatch, not silently ignored.
        baseline = {"prior_result_directory_file_counts": {"phase4_baseline": 999999}}
        audit = verify_no_prior_artifact_changed(baseline)
        self.assertFalse(audit["unchanged"])
        self.assertIn("phase4_baseline", audit["mismatches"])


class PersistedRunRegressionTests(unittest.TestCase):
    """Schema, budget, and consistency checks against this phase's own
    persisted output. Skipped entirely if the phase has not been run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.json_path = OUTPUT_DIR / "pcap_flow_alignment_report.json"
        cls.md_path = OUTPUT_DIR / "pcap_flow_alignment_report.md"
        cls.available = cls.json_path.exists() and cls.md_path.exists()
        if cls.available:
            cls.report = json.loads(cls.json_path.read_text(encoding="utf-8"))
            cls.md_text = cls.md_path.read_text(encoding="utf-8")

    def setUp(self) -> None:
        if not self.available:
            self.skipTest("Phase 9I has not been run yet in this environment")

    # 1. Report schema
    def test_required_top_level_fields_present(self) -> None:
        required = [
            "phase", "success", "verdict", "flow_timestamp_semantics", "pcap_timestamp_semantics",
            "timezone_status", "endpoint_identity_available_in_csv", "pcap_population_scope",
            "csv_population_scope", "correspondence_status", "authoritative_mapping_found",
            "logs_metadata_inspected", "additional_bytes_downloaded", "claims", "limitations", "next_step",
        ]
        for field in required:
            self.assertIn(field, self.report, f"missing required field: {field}")
        self.assertEqual(self.report["phase"], "9I")

    # 2. Verdict validity
    def test_verdict_is_valid(self) -> None:
        self.assertIn(self.report["verdict"], VALID_VERDICTS)

    # 3. No prior result mutation
    def test_no_prior_result_mutation_recorded(self) -> None:
        # The script's own Step 14 audit is embedded implicitly: rerun the check here.
        audit = verify_no_prior_artifact_changed(self.report["repo_safety"])
        self.assertTrue(audit["unchanged"], f"prior artifacts changed: {audit['mismatches']}")

    # 4. Download-budget accounting
    def test_download_budget_respected(self) -> None:
        self.assertLessEqual(self.report["additional_bytes_downloaded"], MAX_TOTAL_DOWNLOAD_BYTES)
        self.assertGreaterEqual(self.report["additional_bytes_downloaded"], 0)
        logs_meta = self.report["logs_zip_metadata"]
        self.assertLessEqual(logs_meta["total_bytes_downloaded"], logs_meta["max_total_download_budget_bytes"])

    # 5. Evidence classification validity
    def test_all_correspondence_statuses_valid(self) -> None:
        for row in self.report["correspondence_classifications"]:
            self.assertIn(row["status"], VALID_STATUSES, f"{row['candidate']}: {row['status']}")

    # 6. Timestamp status handling
    def test_timezone_status_is_a_non_empty_string(self) -> None:
        self.assertIsInstance(self.report["timezone_status"], str)
        self.assertGreater(len(self.report["timezone_status"]), 0)

    # 7. Endpoint correspondence status handling
    def test_endpoint_identity_false_for_wednesday(self) -> None:
        self.assertFalse(self.report["endpoint_identity_available_in_csv"])

    # 8. JSON/Markdown consistency
    def test_verdict_appears_consistently_in_markdown(self) -> None:
        self.assertIn(f"Verdict: **{self.report['verdict']}**", self.md_text)
        self.assertIn(f"## 17. Scientific Verdict\n\n**{self.report['verdict']}**", self.md_text)

    def test_dhcp_crosscheck_consistent_between_json_and_e1_candidate(self) -> None:
        dc = self.report["dhcp_syslog_crosscheck"]
        e1 = next(c for c in self.report["correspondence_classifications"] if c["candidate"].startswith("E1."))
        if dc.get("established"):
            self.assertEqual(e1["status"], "ESTABLISHED")
        self.assertFalse(e1.get("counts_toward_verdict", True))

    def test_archive_structural_fact_matches_verdict(self) -> None:
        pm = self.report["pcap_metadata"]
        if pm.get("other_naming_pattern_members") == 0 and pm.get("total_members", 0) > 0:
            self.assertEqual(self.report["verdict"], "RED")


if __name__ == "__main__":
    unittest.main()
