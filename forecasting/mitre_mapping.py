from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence

import numpy as np


class MitreStage(IntEnum):
    BENIGN = 0

    RECONNAISSANCE = 1
    INITIAL_ACCESS = 2
    EXECUTION = 3
    PERSISTENCE = 4
    PRIVILEGE_ESCALATION = 5
    DEFENSE_EVASION = 6
    CREDENTIAL_ACCESS = 7
    DISCOVERY = 8
    LATERAL_MOVEMENT = 9
    COLLECTION = 10
    COMMAND_AND_CONTROL = 11
    EXFILTRATION = 12
    IMPACT = 13


@dataclass
class MitrePrediction:
    stage: MitreStage
    confidence: float
    technique: Optional[str] = None
    description: Optional[str] = None
    # "authoritative_generic": a standard, widely-cited ATT&CK technique/tactic
    #     association for this attack class (e.g. Brute Force -> T1110). Not a
    #     per-flow verified citation, but not a guess either.
    # "heuristic": stage/technique chosen by categorical judgment, with no
    #     specific ATT&CK technique asserted.
    # "heuristic_reasoned": a corrected heuristic judgment made with an
    #     explicit rationale (see `notes`), not a textbook citation.
    # "heuristic_uncertain": explicitly flagged as uncertain/ambiguous; do not
    #     treat as authoritative (see Infiltration below).
    # "not_applicable": not an attack (Benign).
    evidence_type: str = "heuristic"
    notes: Optional[str] = None


class MitreMapper:
    """
    Maps attack labels and model predictions to MITRE ATT&CK stages.

    This is intentionally separated from the ML model so that dataset
    labels, heuristic predictions and model outputs can all use the
    same interface.
    """

    ATTACK_MAPPING: Dict[str, MitrePrediction] = {
        "benign": MitrePrediction(
            stage=MitreStage.BENIGN,
            confidence=1.0,
            description="Normal network behaviour",
        ),

        "portscan": MitrePrediction(
            stage=MitreStage.RECONNAISSANCE,
            confidence=0.90,
            technique="T1046",
            description="Network Service Discovery",
        ),

        "scan": MitrePrediction(
            stage=MitreStage.RECONNAISSANCE,
            confidence=0.85,
            technique="T1046",
            description="Network Service Discovery",
        ),

        "reconnaissance": MitrePrediction(
            stage=MitreStage.RECONNAISSANCE,
            confidence=0.90,
            description="Reconnaissance activity",
        ),

        "bruteforce": MitrePrediction(
            stage=MitreStage.CREDENTIAL_ACCESS,
            confidence=0.90,
            technique="T1110",
            description="Brute Force",
        ),

        "brute force": MitrePrediction(
            stage=MitreStage.CREDENTIAL_ACCESS,
            confidence=0.90,
            technique="T1110",
            description="Brute Force",
        ),

        "credential attack": MitrePrediction(
            stage=MitreStage.CREDENTIAL_ACCESS,
            confidence=0.85,
            description="Credential access attempt",
        ),

        "web attack": MitrePrediction(
            stage=MitreStage.INITIAL_ACCESS,
            confidence=0.80,
            description="Possible exploitation of public-facing application",
        ),

        "infiltration": MitrePrediction(
            stage=MitreStage.LATERAL_MOVEMENT,
            confidence=0.80,
            description="Possible internal compromise or lateral movement",
            evidence_type="heuristic_uncertain",
            notes=(
                "Reviewed in Phase 7A. CSE-CIC-IDS2018's 'Infiltration' scenario documents a "
                "dropped-file execution phase (Initial Access / Execution) followed by internal "
                "network probing (Discovery / Lateral Movement); the dataset provides one flow-level "
                "label regardless of which sub-phase produced a given flow, so the specific stage "
                "cannot be disambiguated from the label alone. LATERAL_MOVEMENT is retained as the "
                "predominant network-visible signature per public dataset documentation, but this "
                "mapping is explicitly uncertain and must not be treated as authoritative."
            ),
        ),

        "botnet": MitrePrediction(
            stage=MitreStage.COMMAND_AND_CONTROL,
            confidence=0.85,
            description="Possible command and control activity",
        ),

        "c2": MitrePrediction(
            stage=MitreStage.COMMAND_AND_CONTROL,
            confidence=0.85,
            description="Command and control activity",
        ),

        "ddos": MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            description="Distributed denial of service activity",
        ),

        "dos": MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.90,
            description="Denial of service activity",
        ),

        "exfiltration": MitrePrediction(
            stage=MitreStage.EXFILTRATION,
            confidence=0.95,
            technique="T1041",
            description="Exfiltration over C2 channel",
        ),

        "lateral movement": MitrePrediction(
            stage=MitreStage.LATERAL_MOVEMENT,
            confidence=0.90,
            description="Possible lateral movement",
        ),
    }

    STAGE_DESCRIPTIONS = {
        MitreStage.BENIGN: "Normal network behaviour",

        MitreStage.RECONNAISSANCE:
            "Attacker is gathering information about the target network",

        MitreStage.INITIAL_ACCESS:
            "Attacker is attempting to gain initial access",

        MitreStage.EXECUTION:
            "Potential malicious execution activity",

        MitreStage.PERSISTENCE:
            "Potential persistence establishment",

        MitreStage.PRIVILEGE_ESCALATION:
            "Potential privilege escalation",

        MitreStage.DEFENSE_EVASION:
            "Potential security control evasion",

        MitreStage.CREDENTIAL_ACCESS:
            "Potential credential theft or brute-force activity",

        MitreStage.DISCOVERY:
            "Network or system discovery activity",

        MitreStage.LATERAL_MOVEMENT:
            "Potential movement between internal systems",

        MitreStage.COLLECTION:
            "Potential collection of sensitive information",

        MitreStage.COMMAND_AND_CONTROL:
            "Potential attacker command and control communication",

        MitreStage.EXFILTRATION:
            "Potential extraction of data from the network",

        MitreStage.IMPACT:
            "Potential impact on network availability or systems",
    }

    @classmethod
    def normalize_label(cls, label: str) -> str:
        return (
            str(label)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

    @classmethod
    def from_attack_label(
        cls,
        label: str,
    ) -> MitrePrediction:

        normalized = cls.normalize_label(label)

        if normalized in cls.ATTACK_MAPPING:
            prediction = cls.ATTACK_MAPPING[normalized]

            return MitrePrediction(
                stage=prediction.stage,
                confidence=prediction.confidence,
                technique=prediction.technique,
                description=prediction.description,
                evidence_type=prediction.evidence_type,
                notes=prediction.notes,
            )

        for key, prediction in cls.ATTACK_MAPPING.items():

            if key in normalized or normalized in key:

                return MitrePrediction(
                    stage=prediction.stage,
                    confidence=prediction.confidence,
                    technique=prediction.technique,
                    description=prediction.description,
                    evidence_type=prediction.evidence_type,
                    notes=prediction.notes,
                )

        return MitrePrediction(
            stage=MitreStage.BENIGN,
            confidence=0.0,
            description="Unknown attack type",
            evidence_type="not_applicable",
            notes="No exact or keyword match found for this label; defaulted to Benign/confidence 0.0.",
        )

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Sequence[float],
    ) -> MitrePrediction:
        """
        Expects probabilities ordered according to MitreStage values.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.size == 0:
            return MitrePrediction(
                stage=MitreStage.BENIGN,
                confidence=0.0,
                description="No prediction available",
            )

        index = int(np.argmax(probabilities))

        index = min(
            index,
            max(stage.value for stage in MitreStage),
        )

        stage = MitreStage(index)

        return MitrePrediction(
            stage=stage,
            confidence=float(probabilities[index]),
            description=cls.STAGE_DESCRIPTIONS.get(
                stage,
                "Unknown stage",
            ),
        )

    @classmethod
    def progression_score(
        cls,
        stage: MitreStage,
    ) -> float:
        """
        Returns a normalized progression score.
        """

        maximum = max(
            item.value
            for item in MitreStage
        )

        if maximum == 0:
            return 0.0

        return float(stage.value / maximum)

    @classmethod
    def is_progression(
        cls,
        previous: MitreStage,
        current: MitreStage,
    ) -> bool:
        return current.value >= previous.value

    @classmethod
    def get_description(
        cls,
        stage: MitreStage,
    ) -> str:

        return cls.STAGE_DESCRIPTIONS.get(
            stage,
            "Unknown MITRE stage",
        )

    @classmethod
    def from_label_set(
        cls,
        labels,
    ) -> MitrePrediction:
        """
        Deterministic aggregation policy for a set of attack-type labels
        spanning multiple windows (e.g. the dataset's `future_attack_types`
        column, a ';'-joined set of labels covering a forecast horizon).

        Policy (Phase 7A):
          1. Parse `labels` into individual non-empty label strings. A single
             string is split on ';' (the dataset's own join separator); an
             iterable of strings is used as-is.
          2. Map each individual label independently via `from_attack_label`.
          3. Select the mapping with the HIGHEST `MitreStage` value -- the
             most advanced kill-chain stage present is treated as the
             dominant future risk signal for the horizon.
          4. Ties (multiple labels reaching the same maximum stage value) are
             broken by the alphabetically-first NORMALIZED label, so the
             result is fully deterministic for a given input set.
          5. No labels (None, empty string, or an empty/blank iterable) ->
             BENIGN / confidence 0.0.

        This method builds a LABEL/TARGET value only. It must never be used
        to feed information back into a model's input -- only into a loss or
        evaluation target, the same way `future_attack_within_horizon` is
        used elsewhere in this codebase.
        """

        if labels is None:
            individual: List[str] = []
        elif isinstance(labels, str):
            individual = [
                part.strip()
                for part in labels.split(";")
                if part.strip()
            ]
        else:
            individual = [
                str(label).strip()
                for label in labels
                if str(label).strip()
            ]

        if not individual:
            return MitrePrediction(
                stage=MitreStage.BENIGN,
                confidence=0.0,
                description="No attack labels in horizon",
                evidence_type="not_applicable",
                notes="from_label_set received no non-empty labels.",
            )

        mapped = [
            (label, cls.from_attack_label(label))
            for label in individual
        ]

        best_stage_value = max(
            prediction.stage.value
            for _, prediction in mapped
        )

        candidates = [
            (label, prediction)
            for label, prediction in mapped
            if prediction.stage.value == best_stage_value
        ]

        candidates.sort(
            key=lambda item: cls.normalize_label(item[0])
        )

        return candidates[0][1]


# ============================================================
# PHASE 7A: exact-normalized-label entries for real CIC-IDS2018 dataset
# labels (plus known shorthand/reordered aliases of the SAME real labels).
#
# These are added as EXACT keys into MitreMapper.ATTACK_MAPPING, computed
# via MitreMapper.normalize_label() rather than hand-typed, so the key is
# guaranteed to match what from_attack_label() actually looks up at runtime
# (hand-typed keys risk a silent whitespace/typo mismatch that would fall
# through to the fragile substring matcher instead of matching exactly).
#
# from_attack_label() already checks ATTACK_MAPPING for an EXACT normalized
# match before ever reaching the substring-containment fallback loop, so
# every entry registered here takes priority over that fallback for the
# labels named here, without changing the fallback's behavior for any other
# input. No pre-existing key in the ATTACK_MAPPING class-body dict literal
# above is modified by this block (only the "infiltration" entry, reviewed
# per Phase 7A instruction 2, was edited in place).
#
# Two confirmed defects fixed here:
#   - "SQL Injection" previously matched nothing and fell back to
#     BENIGN / confidence 0.0 ("Unknown attack type").
#   - "Brute Force -XSS" previously matched the generic "brute force"
#     keyword via substring containment and was misclassified as
#     Credential Access / T1110, despite being a Cross-Site Scripting
#     attack, not a brute-force credential attack.
# ============================================================

_PHASE_7A_CANONICAL_ENTRIES = [
    # -- DDoS (volumetric / network flood tools) -> Impact, T1498 Network Denial of Service --
    (
        "DDoS attack-HOIC",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            technique="T1498",
            description="High Orbit Ion Cannon volumetric flood",
            evidence_type="authoritative_generic",
            notes="HOIC is a volumetric/network-flood DDoS tool. T1498 Network Denial of "
            "Service is the standard ATT&CK technique for this attack class.",
        ),
    ),
    (
        "DDoS attacks-LOIC-HTTP",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            technique="T1498",
            description="Low Orbit Ion Cannon HTTP-flood mode",
            evidence_type="authoritative_generic",
            notes="Same standard DDoS technique (T1498) as the other LOIC/HOIC flood tools.",
        ),
    ),
    (
        "DDoS attack-LOIC-UDP",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            technique="T1498",
            description="Low Orbit Ion Cannon UDP-flood mode",
            evidence_type="authoritative_generic",
            notes="Same standard DDoS technique (T1498) as the other LOIC/HOIC flood tools.",
        ),
    ),
    # Shorthand aliases used for the two LOIC modes above -- same tool, same
    # mapping, not a new attack class.
    (
        "LOIC-HTTP",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            technique="T1498",
            description="Low Orbit Ion Cannon HTTP-flood mode (shorthand alias)",
            evidence_type="authoritative_generic",
            notes="Alias of 'DDoS attacks-LOIC-HTTP'.",
        ),
    ),
    (
        "LOIC-UDP",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.95,
            technique="T1498",
            description="Low Orbit Ion Cannon UDP-flood mode (shorthand alias)",
            evidence_type="authoritative_generic",
            notes="Alias of 'DDoS attack-LOIC-UDP'.",
        ),
    ),
    # -- Single-target application-layer DoS tools -> Impact, T1499 Endpoint Denial of Service --
    # (distinguished from the network-flood DDoS tools above: these exhaust
    # one web service rather than flooding network bandwidth.)
    (
        "DoS attacks-Hulk",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.90,
            technique="T1499",
            description="HULK single-target HTTP application-layer DoS",
            evidence_type="authoritative_generic",
            notes="Hulk exhausts a single web server's resources rather than flooding network "
            "bandwidth; T1499 Endpoint Denial of Service is the more precise standard technique "
            "than the network-flood T1498 used for HOIC/LOIC.",
        ),
    ),
    (
        "DoS attacks-SlowHTTPTest",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.90,
            technique="T1499",
            description="SlowHTTPTest single-target application-layer DoS",
            evidence_type="authoritative_generic",
            notes="Same rationale as DoS attacks-Hulk: single-service exhaustion, T1499.",
        ),
    ),
    (
        "DoS attacks-GoldenEye",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.90,
            technique="T1499",
            description="GoldenEye single-target application-layer DoS",
            evidence_type="authoritative_generic",
            notes="Same rationale as DoS attacks-Hulk: single-service exhaustion, T1499.",
        ),
    ),
    (
        "DoS attacks-Slowloris",
        MitrePrediction(
            stage=MitreStage.IMPACT,
            confidence=0.90,
            technique="T1499",
            description="Slowloris single-target application-layer DoS",
            evidence_type="authoritative_generic",
            notes="Same rationale as DoS attacks-Hulk: single-service exhaustion, T1499.",
        ),
    ),
    # -- Web application exploitation (Initial Access, T1190) --
    (
        "SQL Injection",
        MitrePrediction(
            stage=MitreStage.INITIAL_ACCESS,
            confidence=0.85,
            technique="T1190",
            description="SQL injection against a public-facing web application",
            evidence_type="authoritative_generic",
            notes="FIXED (Phase 7A): previously matched no entry and fell back to "
            "BENIGN / confidence 0.0. SQL injection against a public-facing web application is a "
            "standard, widely-cited example of MITRE ATT&CK T1190 Exploit Public-Facing Application.",
        ),
    ),
    (
        "Brute Force -XSS",
        MitrePrediction(
            stage=MitreStage.INITIAL_ACCESS,
            confidence=0.70,
            technique="T1190",
            description="Cross-Site Scripting against a public-facing web application",
            evidence_type="heuristic_reasoned",
            notes="FIXED (Phase 7A): despite the dataset's label containing the words 'Brute "
            "Force', this is a Cross-Site Scripting attack, not credential brute-forcing. It "
            "previously matched the generic 'brute force' keyword via substring containment and "
            "was misclassified as Credential Access / T1110. Now resolved via an exact canonical "
            "match to Initial Access / T1190 Exploit Public-Facing Application, the closest "
            "standard ATT&CK technique for exploiting a web app via injected script. This is a "
            "corrected heuristic judgment, not a definitive per-flow-verified citation -- lower "
            "confidence than SQL Injection reflects that.",
        ),
    ),
    (
        "XSS",
        MitrePrediction(
            stage=MitreStage.INITIAL_ACCESS,
            confidence=0.70,
            technique="T1190",
            description="Cross-Site Scripting against a public-facing web application (shorthand alias)",
            evidence_type="heuristic_reasoned",
            notes="Alias of 'Brute Force -XSS' under its common generic name.",
        ),
    ),
    # -- Aliases of already-correct entries (no new mapping invented) --
    (
        "Infilteration",
        MitrePrediction(
            stage=MitreStage.LATERAL_MOVEMENT,
            confidence=0.80,
            description="Possible internal compromise or lateral movement",
            evidence_type="heuristic_uncertain",
            notes="Raw pre-normalization spelling ('Infilteration') of 'Infiltration'; see the "
            "'infiltration' entry above for the full Phase 7A review note. build_temporal_dataset.py "
            "already normalizes this typo to 'Infiltration' upstream, but this alias is registered "
            "defensively for any caller passing the raw dataset spelling directly.",
        ),
    ),
    (
        "Web Brute Force",
        MitrePrediction(
            stage=MitreStage.CREDENTIAL_ACCESS,
            confidence=0.90,
            technique="T1110",
            description="Brute Force",
            evidence_type="authoritative_generic",
            notes="Reordered alias of 'Brute Force -Web'; same genuine login brute-force mapping.",
        ),
    ),
]

for _raw_label, _prediction in _PHASE_7A_CANONICAL_ENTRIES:
    MitreMapper.ATTACK_MAPPING[MitreMapper.normalize_label(_raw_label)] = _prediction