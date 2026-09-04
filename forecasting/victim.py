from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


@dataclass
class NextVictimPrediction:
    victim: Optional[str]

    probability: float

    ranked_targets: List[
        Dict[str, Any]
    ]

    reasons: List[str]


class NextVictimForecaster:
    """
    Predicts likely next victim nodes using network graph
    properties and optional risk information.

    Nodes with unusual connectivity, recent attacker
    interaction, centrality and existing risk are ranked
    higher.
    """

    def __init__(
        self,
        top_k: int = 5,
    ):

        self.top_k = top_k

    def predict(
        self,
        graph: nx.Graph,
        attacker_nodes: Optional[
            List[str]
        ] = None,
        node_risk: Optional[
            Dict[str, float]
        ] = None,
    ) -> NextVictimPrediction:

        if graph.number_of_nodes() == 0:

            return NextVictimPrediction(
                victim=None,
                probability=0.0,
                ranked_targets=[],
                reasons=[
                    "The network graph contains no nodes."
                ],
            )

        attacker_nodes = (
            attacker_nodes or []
        )

        node_risk = (
            node_risk or {}
        )

        degree_centrality = (
            nx.degree_centrality(graph)
        )

        try:

            betweenness = (
                nx.betweenness_centrality(
                    graph,
                    normalized=True,
                )
            )

        except Exception:

            betweenness = {
                node: 0.0
                for node in graph.nodes
            }

        scores = {}

        for node in graph.nodes:

            if node in attacker_nodes:
                continue

            score = 0.0

            # Structural importance
            score += (
                0.35
                * degree_centrality.get(
                    node,
                    0.0,
                )
            )

            # Bridge / strategic importance
            score += (
                0.25
                * betweenness.get(
                    node,
                    0.0,
                )
            )

            # Existing risk from anomaly or ML model
            score += (
                0.25
                * self._clamp(
                    node_risk.get(
                        node,
                        0.0,
                    )
                )
            )

            # Connectivity to suspected attackers
            attacker_connections = (
                self._attacker_connections(
                    graph,
                    node,
                    attacker_nodes,
                )
            )

            score += (
                0.15
                * attacker_connections
            )

            scores[node] = score

        if not scores:

            return NextVictimPrediction(
                victim=None,
                probability=0.0,
                ranked_targets=[],
                reasons=[
                    "No valid victim candidates found."
                ],
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        maximum = max(
            scores.values()
        )

        if maximum > 0:

            probabilities = {
                node: score / maximum
                for node, score
                in scores.items()
            }

        else:

            probabilities = {
                node: 0.0
                for node in scores
            }

        ranked_targets = []

        for node, score in ranked[:self.top_k]:

            ranked_targets.append(
                {
                    "node": str(node),

                    "risk_score": round(
                        float(score),
                        4,
                    ),

                    "probability": round(
                        float(
                            probabilities[node]
                        ),
                        4,
                    ),

                    "degree_centrality": round(
                        float(
                            degree_centrality.get(
                                node,
                                0.0,
                            )
                        ),
                        4,
                    ),

                    "betweenness_centrality": round(
                        float(
                            betweenness.get(
                                node,
                                0.0,
                            )
                        ),
                        4,
                    ),
                }
            )

        victim = ranked_targets[0]["node"]

        probability = (
            ranked_targets[0]["probability"]
        )

        reasons = self._generate_reasons(
            victim=victim,
            graph=graph,
            attacker_nodes=attacker_nodes,
            degree_centrality=degree_centrality,
            betweenness=betweenness,
            node_risk=node_risk,
        )

        return NextVictimPrediction(
            victim=victim,

            probability=float(
                probability
            ),

            ranked_targets=ranked_targets,

            reasons=reasons,
        )

    def _attacker_connections(
        self,
        graph: nx.Graph,
        node: str,
        attacker_nodes: List[str],
    ) -> float:

        if not attacker_nodes:
            return 0.0

        connections = 0

        for attacker in attacker_nodes:

            if (
                attacker in graph
                and graph.has_edge(
                    attacker,
                    node,
                )
            ):
                connections += 1

        return min(
            connections
            / len(attacker_nodes),
            1.0,
        )

    def _generate_reasons(
        self,
        victim: str,
        graph: nx.Graph,
        attacker_nodes: List[str],
        degree_centrality: Dict[
            str,
            float,
        ],
        betweenness: Dict[
            str,
            float,
        ],
        node_risk: Dict[
            str,
            float,
        ],
    ) -> List[str]:

        reasons = []

        if (
            degree_centrality.get(
                victim,
                0.0,
            ) >= 0.30
        ):

            reasons.append(
                "The target has high network connectivity."
            )

        if (
            betweenness.get(
                victim,
                0.0,
            ) >= 0.10
        ):

            reasons.append(
                "The target occupies an important network "
                "communication position."
            )

        if (
            node_risk.get(
                victim,
                0.0,
            ) >= 0.50
        ):

            reasons.append(
                "The target already has elevated anomaly "
                "or attack risk."
            )

        for attacker in attacker_nodes:

            if (
                attacker in graph
                and graph.has_edge(
                    attacker,
                    victim,
                )
            ):

                reasons.append(
                    "The target has direct communication "
                    "with a suspected attacker node."
                )

                break

        if not reasons:

            reasons.append(
                "The target was ranked highest based on "
                "its network topology."
            )

        return reasons

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )