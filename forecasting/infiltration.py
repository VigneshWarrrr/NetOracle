from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .mitre_mapping import (
    MitreMapper,
    MitreStage,
)


@dataclass
class InfiltrationForecast:
    probability: float

    risk_level: str

    predicted_stage: MitreStage

    confidence: float

    reasons: list[str]


class InfiltrationForecaster:
    """
    Estimates probability that suspicious network activity
    is progressing toward internal infiltration or
    lateral movement.

    The class can consume model predictions and optional
    network state features.
    """

    def __init__(
        self,
        infiltration_threshold: float = 0.65,
        high_risk_threshold: float = 0.80,
    ):

        self.infiltration_threshold = (
            infiltration_threshold
        )

        self.high_risk_threshold = (
            high_risk_threshold
        )

    def forecast(
        self,
        attack_probability: float,
        mitre_stage: MitreStage,
        network_state: Optional[
            Dict[str, Any]
        ] = None,
        stage_confidence: float = 1.0,
    ) -> InfiltrationForecast:

        attack_probability = self._clamp(
            attack_probability
        )

        stage_confidence = self._clamp(
            stage_confidence
        )

        network_score = (
            self._calculate_network_score(
                network_state or {}
            )
        )

        stage_score = (
            self._calculate_stage_score(
                mitre_stage
            )
        )

        probability = (
            0.50 * attack_probability
            +
            0.30 * stage_score
            +
            0.20 * network_score
        )

        probability *= (
            0.75
            +
            0.25 * stage_confidence
        )

        probability = self._clamp(
            probability
        )

        reasons = self._generate_reasons(
            attack_probability=attack_probability,
            mitre_stage=mitre_stage,
            network_state=network_state or {},
            network_score=network_score,
        )

        return InfiltrationForecast(
            probability=probability,

            risk_level=self._risk_level(
                probability
            ),

            predicted_stage=mitre_stage,

            confidence=stage_confidence,

            reasons=reasons,
        )

    def _calculate_stage_score(
        self,
        stage: MitreStage,
    ) -> float:

        high_risk_stages = {
            MitreStage.INITIAL_ACCESS: 0.55,

            MitreStage.CREDENTIAL_ACCESS: 0.70,

            MitreStage.DISCOVERY: 0.60,

            MitreStage.LATERAL_MOVEMENT: 1.00,

            MitreStage.COMMAND_AND_CONTROL: 0.90,

            MitreStage.EXFILTRATION: 0.95,
        }

        if stage in high_risk_stages:
            return high_risk_stages[stage]

        return MitreMapper.progression_score(
            stage
        )

    def _calculate_network_score(
        self,
        state: Dict[str, Any],
    ) -> float:

        signals = []

        unique_ports = self._get(
            state,
            "unique_destination_ports",
            "unique_dst_ports",
            "unique_ports",
        )

        if unique_ports is not None:

            signals.append(
                min(
                    float(unique_ports) / 100.0,
                    1.0,
                )
            )

        new_hosts = self._get(
            state,
            "new_hosts",
            "new_nodes",
            "new_destination_hosts",
        )

        if new_hosts is not None:

            signals.append(
                min(
                    float(new_hosts) / 20.0,
                    1.0,
                )
            )

        failed_connections = self._get(
            state,
            "failed_connections",
            "failed_logins",
            "failed_authentication",
        )

        if failed_connections is not None:

            signals.append(
                min(
                    float(failed_connections) / 50.0,
                    1.0,
                )
            )

        internal_connections = self._get(
            state,
            "internal_connections",
            "east_west_connections",
        )

        if internal_connections is not None:

            signals.append(
                min(
                    float(internal_connections)
                    / 1000.0,
                    1.0,
                )
            )

        anomaly_score = self._get(
            state,
            "anomaly_score",
        )

        if anomaly_score is not None:

            signals.append(
                self._clamp(
                    float(anomaly_score)
                )
            )

        if not signals:
            return 0.0

        return float(
            np.mean(signals)
        )

    def _generate_reasons(
        self,
        attack_probability: float,
        mitre_stage: MitreStage,
        network_state: Dict[str, Any],
        network_score: float,
    ) -> list[str]:

        reasons = []

        if attack_probability >= 0.70:

            reasons.append(
                "The world model predicts a high probability "
                "of malicious activity in the next network state."
            )

        if mitre_stage in {
            MitreStage.INITIAL_ACCESS,
            MitreStage.CREDENTIAL_ACCESS,
            MitreStage.DISCOVERY,
            MitreStage.LATERAL_MOVEMENT,
            MitreStage.COMMAND_AND_CONTROL,
        }:

            reasons.append(
                "The predicted MITRE stage is consistent with "
                "possible attack progression or internal compromise."
            )

        if network_score >= 0.60:

            reasons.append(
                "Network-state indicators show elevated "
                "behavioural risk."
            )

        if not reasons:

            reasons.append(
                "No strong infiltration indicators were "
                "detected in the current forecast."
            )

        return reasons

    def _risk_level(
        self,
        probability: float,
    ) -> str:

        if probability >= self.high_risk_threshold:
            return "CRITICAL"

        if probability >= self.infiltration_threshold:
            return "HIGH"

        if probability >= 0.40:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _get(
        data: Dict[str, Any],
        *keys: str,
    ) -> Optional[Any]:

        for key in keys:

            if key in data:
                return data[key]

        return None

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return float(
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )
        )