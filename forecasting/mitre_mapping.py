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
            )

        for key, prediction in cls.ATTACK_MAPPING.items():

            if key in normalized or normalized in key:

                return MitrePrediction(
                    stage=prediction.stage,
                    confidence=prediction.confidence,
                    technique=prediction.technique,
                    description=prediction.description,
                )

        return MitrePrediction(
            stage=MitreStage.BENIGN,
            confidence=0.0,
            description="Unknown attack type",
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