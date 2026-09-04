from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

from .mitre_mapping import (
    MitreMapper,
    MitrePrediction,
    MitreStage,
)


@dataclass
class AttackChainStage:
    stage: MitreStage
    confidence: float

    timestamp: Optional[datetime] = None

    technique: Optional[str] = None
    description: Optional[str] = None

    risk_score: float = 0.0


@dataclass
class AttackChain:
    stages: List[AttackChainStage] = field(
        default_factory=list
    )

    overall_risk: float = 0.0

    is_active: bool = False

    predicted_next_stage: Optional[MitreStage] = None

    def add_stage(
        self,
        stage: AttackChainStage,
    ) -> None:

        self.stages.append(stage)

        self._update_state()

    def latest_stage(
        self,
    ) -> Optional[AttackChainStage]:

        if not self.stages:
            return None

        return self.stages[-1]

    def unique_stages(
        self,
    ) -> List[MitreStage]:

        seen = set()

        stages = []

        for item in self.stages:

            if item.stage not in seen:

                stages.append(item.stage)

                seen.add(item.stage)

        return stages

    def _update_state(
        self,
    ) -> None:

        if not self.stages:

            self.overall_risk = 0.0

            self.is_active = False

            return

        latest = self.stages[-1]

        progression = MitreMapper.progression_score(
            latest.stage
        )

        self.overall_risk = float(
            min(
                1.0,
                (
                    0.55 * latest.confidence
                    +
                    0.45 * progression
                ),
            )
        )

        self.is_active = (
            latest.stage != MitreStage.BENIGN
        )

    def summary(
        self,
    ) -> dict:

        latest = self.latest_stage()

        return {
            "active": self.is_active,

            "overall_risk": round(
                self.overall_risk,
                4,
            ),

            "stages": [
                stage.stage.name
                for stage in self.stages
            ],

            "latest_stage": (
                latest.stage.name
                if latest
                else None
            ),

            "predicted_next_stage": (
                self.predicted_next_stage.name
                if self.predicted_next_stage
                else None
            ),
        }


class AttackChainBuilder:
    """
    Builds attack progression from sequential MITRE predictions.

    The builder removes repeated states and smooths small
    backwards fluctuations from the model.
    """

    def __init__(
        self,
        allow_backward_progression: bool = False,
    ):

        self.allow_backward_progression = (
            allow_backward_progression
        )

    def build(
        self,
        predictions: Iterable[MitrePrediction],
    ) -> AttackChain:

        chain = AttackChain()

        previous_stage: Optional[
            MitreStage
        ] = None

        for prediction in predictions:

            stage = prediction.stage

            if (
                previous_stage is not None
                and stage == previous_stage
            ):
                continue

            if (
                not self.allow_backward_progression
                and previous_stage is not None
                and stage.value < previous_stage.value
            ):
                continue

            risk_score = self._calculate_risk(
                prediction
            )

            chain.add_stage(
                AttackChainStage(
                    stage=stage,
                    confidence=prediction.confidence,
                    technique=prediction.technique,
                    description=prediction.description,
                    timestamp=datetime.utcnow(),
                    risk_score=risk_score,
                )
            )

            previous_stage = stage

        chain.predicted_next_stage = (
            self.predict_next_stage(chain)
        )

        return chain

    def predict_next_stage(
        self,
        chain: AttackChain,
    ) -> Optional[MitreStage]:

        latest = chain.latest_stage()

        if latest is None:
            return None

        current = latest.stage

        if current == MitreStage.BENIGN:
            return MitreStage.RECONNAISSANCE

        next_value = min(
            current.value + 1,
            max(
                stage.value
                for stage in MitreStage
            ),
        )

        return MitreStage(next_value)

    def _calculate_risk(
        self,
        prediction: MitrePrediction,
    ) -> float:

        progression = (
            MitreMapper.progression_score(
                prediction.stage
            )
        )

        return float(
            min(
                1.0,
                (
                    0.6 * prediction.confidence
                    +
                    0.4 * progression
                ),
            )
        )