"""Phase 7A: focused tests for forecasting.mitre_mapping.

Plain unittest.TestCase (not Django DB-backed) is used deliberately:
MitreMapper has zero ORM/Django dependency, so these tests run without any
database or migration setup, while remaining fully discoverable by Django's
own test runner (`manage.py test forecasting`) if ever invoked that way.

Coverage:
  - every real CIC-IDS2018 attack label currently present in the temporal
    dataset (per data/audit label documentation), including the two
    confirmed Phase 7A defect fixes (SQL Injection, Brute Force -XSS);
  - the explicitly-documented Infiltration uncertainty;
  - shorthand/reordered aliases of already-covered real labels;
  - regression coverage proving pre-existing generic keyword mappings were
    NOT altered by the Phase 7A changes;
  - the unknown-label fallback;
  - the from_label_set() deterministic aggregation policy.
"""

from __future__ import annotations

import unittest

from forecasting.mitre_mapping import MitreMapper, MitreStage


class RealDatasetLabelMappingTests(unittest.TestCase):
    """Every real, non-header CIC-IDS2018 label in data/windows/*.csv."""

    # (raw dataset label, expected stage, expected technique or None)
    REAL_LABELS = [
        ("Benign", MitreStage.BENIGN, None),
        ("Bot", MitreStage.COMMAND_AND_CONTROL, None),
        ("DDoS attack-HOIC", MitreStage.IMPACT, "T1498"),
        ("DDoS attacks-LOIC-HTTP", MitreStage.IMPACT, "T1498"),
        ("DDoS attack-LOIC-UDP", MitreStage.IMPACT, "T1498"),
        ("DoS attacks-Hulk", MitreStage.IMPACT, "T1499"),
        ("DoS attacks-SlowHTTPTest", MitreStage.IMPACT, "T1499"),
        ("DoS attacks-GoldenEye", MitreStage.IMPACT, "T1499"),
        ("DoS attacks-Slowloris", MitreStage.IMPACT, "T1499"),
        ("FTP-BruteForce", MitreStage.CREDENTIAL_ACCESS, "T1110"),
        ("SSH-Bruteforce", MitreStage.CREDENTIAL_ACCESS, "T1110"),
        ("Infiltration", MitreStage.LATERAL_MOVEMENT, None),
        ("Brute Force -Web", MitreStage.CREDENTIAL_ACCESS, "T1110"),
        ("Brute Force -XSS", MitreStage.INITIAL_ACCESS, "T1190"),
        ("SQL Injection", MitreStage.INITIAL_ACCESS, "T1190"),
    ]

    def test_every_real_label_maps_as_expected(self) -> None:
        for label, expected_stage, expected_technique in self.REAL_LABELS:
            with self.subTest(label=label):
                prediction = MitreMapper.from_attack_label(label)
                self.assertEqual(prediction.stage, expected_stage)
                self.assertEqual(prediction.technique, expected_technique)

    def test_no_real_label_falls_back_to_unknown(self) -> None:
        """None of the 15 real labels should ever hit the 'Unknown attack type' fallback."""
        for label, _, _ in self.REAL_LABELS:
            with self.subTest(label=label):
                prediction = MitreMapper.from_attack_label(label)
                self.assertNotEqual(prediction.description, "Unknown attack type")

    def test_infilteration_typo_spelling_matches_infiltration(self) -> None:
        """build_temporal_dataset.py normalizes this typo upstream, but the
        mapper must also handle the raw dataset spelling defensively."""
        typo = MitreMapper.from_attack_label("Infilteration")
        fixed = MitreMapper.from_attack_label("Infiltration")
        self.assertEqual(typo.stage, fixed.stage)
        self.assertEqual(typo.confidence, fixed.confidence)
        self.assertEqual(typo.evidence_type, fixed.evidence_type)


class ConfirmedDefectFixTests(unittest.TestCase):
    """Explicit regression tests for the two defects confirmed in Phase 7 inspection."""

    def test_sql_injection_is_not_benign(self) -> None:
        prediction = MitreMapper.from_attack_label("SQL Injection")
        self.assertNotEqual(prediction.stage, MitreStage.BENIGN)
        self.assertGreater(prediction.confidence, 0.0)
        self.assertEqual(prediction.stage, MitreStage.INITIAL_ACCESS)
        self.assertEqual(prediction.technique, "T1190")

    def test_brute_force_xss_is_not_credential_access(self) -> None:
        prediction = MitreMapper.from_attack_label("Brute Force -XSS")
        self.assertNotEqual(prediction.stage, MitreStage.CREDENTIAL_ACCESS)
        self.assertNotEqual(prediction.technique, "T1110")
        self.assertEqual(prediction.stage, MitreStage.INITIAL_ACCESS)
        self.assertEqual(prediction.technique, "T1190")

    def test_brute_force_web_is_unaffected_and_still_credential_access(self) -> None:
        """The fix for Brute Force -XSS must not change the genuine web-login
        brute-force label."""
        prediction = MitreMapper.from_attack_label("Brute Force -Web")
        self.assertEqual(prediction.stage, MitreStage.CREDENTIAL_ACCESS)
        self.assertEqual(prediction.technique, "T1110")


class InfiltrationUncertaintyDocumentationTests(unittest.TestCase):
    def test_infiltration_is_explicitly_flagged_uncertain(self) -> None:
        prediction = MitreMapper.from_attack_label("Infiltration")
        self.assertEqual(prediction.evidence_type, "heuristic_uncertain")
        self.assertIsNotNone(prediction.notes)
        self.assertGreater(len(prediction.notes or ""), 0)

    def test_infiltration_stage_retained(self) -> None:
        """Phase 7A documents the uncertainty; it does not silently change the stage."""
        prediction = MitreMapper.from_attack_label("Infiltration")
        self.assertEqual(prediction.stage, MitreStage.LATERAL_MOVEMENT)


class AliasConsistencyTests(unittest.TestCase):
    """Shorthand/reordered aliases of an already-covered real label must
    resolve identically to their canonical counterpart -- no new mapping is
    invented for these, only recognition of an existing one."""

    ALIAS_PAIRS = [
        ("LOIC-HTTP", "DDoS attacks-LOIC-HTTP"),
        ("LOIC-UDP", "DDoS attack-LOIC-UDP"),
        ("Web Brute Force", "Brute Force -Web"),
        ("XSS", "Brute Force -XSS"),
    ]

    def test_aliases_match_canonical_label(self) -> None:
        for alias, canonical in self.ALIAS_PAIRS:
            with self.subTest(alias=alias, canonical=canonical):
                alias_prediction = MitreMapper.from_attack_label(alias)
                canonical_prediction = MitreMapper.from_attack_label(canonical)
                self.assertEqual(alias_prediction.stage, canonical_prediction.stage)
                self.assertEqual(alias_prediction.technique, canonical_prediction.technique)
                self.assertEqual(alias_prediction.confidence, canonical_prediction.confidence)


class UnrelatedMappingRegressionTests(unittest.TestCase):
    """Proves the Phase 7A change did not alter any pre-existing generic
    keyword mapping outside the confirmed-defect scope. Expected values are
    the exact values captured during Phase 7 inspection, before any code
    change was made."""

    UNCHANGED = [
        ("portscan", MitreStage.RECONNAISSANCE, 0.90, "T1046"),
        ("scan", MitreStage.RECONNAISSANCE, 0.85, "T1046"),
        ("reconnaissance", MitreStage.RECONNAISSANCE, 0.90, None),
        ("bruteforce", MitreStage.CREDENTIAL_ACCESS, 0.90, "T1110"),
        ("brute force", MitreStage.CREDENTIAL_ACCESS, 0.90, "T1110"),
        ("credential attack", MitreStage.CREDENTIAL_ACCESS, 0.85, None),
        ("web attack", MitreStage.INITIAL_ACCESS, 0.80, None),
        ("botnet", MitreStage.COMMAND_AND_CONTROL, 0.85, None),
        ("c2", MitreStage.COMMAND_AND_CONTROL, 0.85, None),
        ("ddos", MitreStage.IMPACT, 0.95, None),
        ("dos", MitreStage.IMPACT, 0.90, None),
        ("exfiltration", MitreStage.EXFILTRATION, 0.95, "T1041"),
        ("lateral movement", MitreStage.LATERAL_MOVEMENT, 0.90, None),
    ]

    def test_unrelated_mappings_unchanged(self) -> None:
        for label, expected_stage, expected_confidence, expected_technique in self.UNCHANGED:
            with self.subTest(label=label):
                prediction = MitreMapper.from_attack_label(label)
                self.assertEqual(prediction.stage, expected_stage)
                self.assertEqual(prediction.confidence, expected_confidence)
                self.assertEqual(prediction.technique, expected_technique)

    def test_ftp_and_ssh_bruteforce_still_resolve_via_existing_keyword(self) -> None:
        """These were already correct before Phase 7A and must remain so."""
        for label in ("FTP-BruteForce", "SSH-Bruteforce"):
            with self.subTest(label=label):
                prediction = MitreMapper.from_attack_label(label)
                self.assertEqual(prediction.stage, MitreStage.CREDENTIAL_ACCESS)
                self.assertEqual(prediction.technique, "T1110")


class UnknownLabelFallbackTests(unittest.TestCase):
    def test_unmapped_label_falls_back_to_benign_zero_confidence(self) -> None:
        prediction = MitreMapper.from_attack_label("totally unknown attack type xyz")
        self.assertEqual(prediction.stage, MitreStage.BENIGN)
        self.assertEqual(prediction.confidence, 0.0)
        self.assertEqual(prediction.evidence_type, "not_applicable")


class LabelSetAggregationTests(unittest.TestCase):
    """Deterministic aggregation policy for multiple future-horizon labels."""

    def test_none_input_is_benign(self) -> None:
        self.assertEqual(MitreMapper.from_label_set(None).stage, MitreStage.BENIGN)

    def test_empty_string_is_benign(self) -> None:
        self.assertEqual(MitreMapper.from_label_set("").stage, MitreStage.BENIGN)

    def test_empty_list_is_benign(self) -> None:
        self.assertEqual(MitreMapper.from_label_set([]).stage, MitreStage.BENIGN)

    def test_single_label_matches_from_attack_label(self) -> None:
        direct = MitreMapper.from_attack_label("SSH-Bruteforce")
        aggregated = MitreMapper.from_label_set("SSH-Bruteforce")
        self.assertEqual(direct.stage, aggregated.stage)
        self.assertEqual(direct.technique, aggregated.technique)

    def test_semicolon_joined_string_picks_highest_stage(self) -> None:
        """Mirrors the dataset's own future_attack_types format."""
        result = MitreMapper.from_label_set("SSH-Bruteforce;DDoS attack-HOIC")
        # CREDENTIAL_ACCESS(7) vs IMPACT(13) -> IMPACT wins
        self.assertEqual(result.stage, MitreStage.IMPACT)

    def test_list_input_picks_highest_stage(self) -> None:
        result = MitreMapper.from_label_set(["SQL Injection", "Brute Force -Web"])
        # INITIAL_ACCESS(2) vs CREDENTIAL_ACCESS(7) -> CREDENTIAL_ACCESS wins
        self.assertEqual(result.stage, MitreStage.CREDENTIAL_ACCESS)

    def test_tie_break_is_deterministic_and_alphabetical(self) -> None:
        """Both labels map to IMPACT (a tie); the alphabetically-first
        NORMALIZED label ('dos attacks goldeneye' < 'dos attacks hulk')
        must always win, deterministically."""
        expected = MitreMapper.from_attack_label("DoS attacks-GoldenEye")
        result_forward = MitreMapper.from_label_set(["DoS attacks-Hulk", "DoS attacks-GoldenEye"])
        result_reversed = MitreMapper.from_label_set(["DoS attacks-GoldenEye", "DoS attacks-Hulk"])
        self.assertEqual(result_forward.technique, expected.technique)
        self.assertEqual(result_forward.description, expected.description)
        self.assertEqual(result_forward.description, result_reversed.description)

    def test_never_used_as_model_input_is_a_pure_function(self) -> None:
        """Sanity check that the aggregation function is side-effect-free and
        deterministic (same input -> same output), which is required for it
        to be safely usable only as a label/target, never as live model state."""
        first = MitreMapper.from_label_set("Bot;Infiltration")
        second = MitreMapper.from_label_set("Bot;Infiltration")
        self.assertEqual(first.stage, second.stage)
        self.assertEqual(first.confidence, second.confidence)


class EvidenceTypeCoverageTests(unittest.TestCase):
    """Every mapping must be traceable to a declared evidence category --
    nothing should silently claim authority it does not have."""

    VALID_EVIDENCE_TYPES = {
        "authoritative_generic",
        "heuristic",
        "heuristic_reasoned",
        "heuristic_uncertain",
        "not_applicable",
    }

    def test_all_attack_mapping_entries_have_valid_evidence_type(self) -> None:
        for key, prediction in MitreMapper.ATTACK_MAPPING.items():
            with self.subTest(key=key):
                self.assertIn(prediction.evidence_type, self.VALID_EVIDENCE_TYPES)

    def test_all_attack_mapping_technique_ids_are_well_formed_or_none(self) -> None:
        import re

        pattern = re.compile(r"^T\d{4}$")
        for key, prediction in MitreMapper.ATTACK_MAPPING.items():
            with self.subTest(key=key):
                if prediction.technique is not None:
                    self.assertRegex(prediction.technique, pattern)

    def test_no_authoritative_claim_without_a_technique_or_explicit_rationale(self) -> None:
        """'authoritative_generic' entries should carry either a technique ID
        or a clear note -- never a bare unexplained authority claim."""
        for key, prediction in MitreMapper.ATTACK_MAPPING.items():
            if prediction.evidence_type == "authoritative_generic":
                with self.subTest(key=key):
                    self.assertTrue(prediction.technique is not None or prediction.notes)


if __name__ == "__main__":
    unittest.main()
