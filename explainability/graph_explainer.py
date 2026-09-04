from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


@dataclass
class GraphExplanation:
    """
    Explanation of graph-based risk or victim prediction.
    """

    target_node: Optional[str]

    node_importance: List[
        Dict[str, Any]
    ]

    important_edges: List[
        Dict[str, Any]
    ]

    suspicious_paths: List[
        List[str]
    ]

    reasons: List[str]

    graph_summary: Dict[str, Any]


class GraphExplainer:
    """
    Explains graph-based attack forecasting.

    The explainer combines:

    - Degree centrality
    - Betweenness centrality
    - Closeness centrality
    - PageRank
    - Node risk
    - Connectivity to attacker nodes
    - Important communication edges

    This is intentionally model-agnostic so it can explain
    heuristic graph forecasting now and can later be extended
    to explain a Temporal GNN.
    """

    def __init__(
        self,
        top_k_nodes: int = 10,
        top_k_edges: int = 10,
    ):

        self.top_k_nodes = (
            top_k_nodes
        )

        self.top_k_edges = (
            top_k_edges
        )

    # --------------------------------------------------
    # MAIN EXPLANATION
    # --------------------------------------------------

    def explain(
        self,
        graph: nx.Graph,
        target_node: Optional[str] = None,
        attacker_nodes: Optional[
            Sequence[str]
        ] = None,
        node_risk: Optional[
            Dict[str, float]
        ] = None,
    ) -> GraphExplanation:

        attacker_nodes = list(
            attacker_nodes or []
        )

        node_risk = (
            node_risk or {}
        )

        if graph.number_of_nodes() == 0:

            return GraphExplanation(
                target_node=target_node,

                node_importance=[],

                important_edges=[],

                suspicious_paths=[],

                reasons=[
                    "The graph contains no nodes."
                ],

                graph_summary={
                    "nodes": 0,
                    "edges": 0,
                },
            )

        metrics = (
            self._calculate_metrics(
                graph
            )
        )

        node_importance = (
            self._rank_nodes(
                graph=graph,
                metrics=metrics,
                attacker_nodes=attacker_nodes,
                node_risk=node_risk,
                target_node=target_node,
            )
        )

        important_edges = (
            self._rank_edges(
                graph=graph,
                attacker_nodes=attacker_nodes,
                target_node=target_node,
                node_risk=node_risk,
            )
        )

        suspicious_paths = (
            self._find_suspicious_paths(
                graph=graph,
                attacker_nodes=attacker_nodes,
                target_node=target_node,
            )
        )

        reasons = (
            self._generate_reasons(
                graph=graph,
                target_node=target_node,
                attacker_nodes=attacker_nodes,
                node_risk=node_risk,
                metrics=metrics,
                suspicious_paths=suspicious_paths,
            )
        )

        summary = (
            self._graph_summary(
                graph=graph,
                attacker_nodes=attacker_nodes,
                node_risk=node_risk,
            )
        )

        return GraphExplanation(
            target_node=target_node,

            node_importance=node_importance,

            important_edges=important_edges,

            suspicious_paths=suspicious_paths,

            reasons=reasons,

            graph_summary=summary,
        )

    # --------------------------------------------------
    # GRAPH METRICS
    # --------------------------------------------------

    def _calculate_metrics(
        self,
        graph: nx.Graph,
    ) -> Dict[
        str,
        Dict[str, float]
    ]:

        degree = (
            nx.degree_centrality(
                graph
            )
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

        try:

            closeness = (
                nx.closeness_centrality(
                    graph
                )
            )

        except Exception:

            closeness = {
                node: 0.0
                for node in graph.nodes
            }

        try:

            pagerank = (
                nx.pagerank(
                    graph
                )
            )

        except Exception:

            pagerank = {
                node: 0.0
                for node in graph.nodes
            }

        return {
            "degree": degree,

            "betweenness": betweenness,

            "closeness": closeness,

            "pagerank": pagerank,
        }

    # --------------------------------------------------
    # NODE RANKING
    # --------------------------------------------------

    def _rank_nodes(
        self,
        graph: nx.Graph,
        metrics: Dict[
            str,
            Dict[str, float]
        ],
        attacker_nodes: List[str],
        node_risk: Dict[
            str,
            float
        ],
        target_node: Optional[str],
    ) -> List[
        Dict[str, Any]
    ]:

        ranked = []

        for node in graph.nodes:

            degree = (
                metrics["degree"].get(
                    node,
                    0.0,
                )
            )

            betweenness = (
                metrics["betweenness"].get(
                    node,
                    0.0,
                )
            )

            closeness = (
                metrics["closeness"].get(
                    node,
                    0.0,
                )
            )

            pagerank = (
                metrics["pagerank"].get(
                    node,
                    0.0,
                )
            )

            risk = (
                self._clamp(
                    node_risk.get(
                        node,
                        0.0,
                    )
                )
            )

            attacker_score = (
                self._attacker_connectivity(
                    graph,
                    node,
                    attacker_nodes,
                )
            )

            score = (
                0.25 * degree
                +
                0.25 * betweenness
                +
                0.15 * closeness
                +
                0.15 * pagerank
                +
                0.15 * risk
                +
                0.05 * attacker_score
            )

            ranked.append(
                {
                    "node": str(node),

                    "importance": float(
                        score
                    ),

                    "degree_centrality": float(
                        degree
                    ),

                    "betweenness_centrality": float(
                        betweenness
                    ),

                    "closeness_centrality": float(
                        closeness
                    ),

                    "pagerank": float(
                        pagerank
                    ),

                    "node_risk": float(
                        risk
                    ),

                    "attacker_connectivity": float(
                        attacker_score
                    ),

                    "is_target": (
                        str(node)
                        == str(target_node)
                    ),

                    "is_attacker": (
                        str(node)
                        in {
                            str(item)
                            for item
                            in attacker_nodes
                        }
                    ),
                }
            )

        ranked.sort(
            key=lambda item: item[
                "importance"
            ],
            reverse=True,
        )

        maximum = (
            ranked[0]["importance"]
            if ranked
            else 1.0
        )

        if maximum <= 0:
            maximum = 1.0

        for item in ranked:

            item[
                "relative_importance"
            ] = float(
                item["importance"]
                / maximum
            )

            item[
                "percentage"
            ] = round(
                item[
                    "relative_importance"
                ]
                * 100,
                2,
            )

        return ranked[
            :self.top_k_nodes
        ]

    # --------------------------------------------------
    # EDGE RANKING
    # --------------------------------------------------

    def _rank_edges(
        self,
        graph: nx.Graph,
        attacker_nodes: List[str],
        target_node: Optional[str],
        node_risk: Dict[
            str,
            float
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        ranked = []

        for source, destination, data in (
            graph.edges(
                data=True
            )
        ):

            weight = float(
                data.get(
                    "weight",
                    1.0,
                )
            )

            risk = (
                self._clamp(
                    node_risk.get(
                        source,
                        0.0,
                    )
                )
                +
                self._clamp(
                    node_risk.get(
                        destination,
                        0.0,
                    )
                )
            ) / 2.0

            attacker_connection = (
                float(
                    source
                    in attacker_nodes
                    or destination
                    in attacker_nodes
                )
            )

            target_connection = (
                float(
                    target_node is not None
                    and (
                        str(source)
                        == str(target_node)
                        or str(destination)
                        == str(target_node)
                    )
                )
            )

            score = (
                0.40
                * self._normalize_weight(
                    weight
                )
                +
                0.30
                * risk
                +
                0.20
                * attacker_connection
                +
                0.10
                * target_connection
            )

            ranked.append(
                {
                    "source": str(
                        source
                    ),

                    "destination": str(
                        destination
                    ),

                    "importance": float(
                        score
                    ),

                    "weight": float(
                        weight
                    ),

                    "risk": float(
                        risk
                    ),

                    "connected_to_attacker": bool(
                        attacker_connection
                    ),

                    "connected_to_target": bool(
                        target_connection
                    ),
                }
            )

        ranked.sort(
            key=lambda item: item[
                "importance"
            ],
            reverse=True,
        )

        return ranked[
            :self.top_k_edges
        ]

    # --------------------------------------------------
    # SUSPICIOUS PATHS
    # --------------------------------------------------

    def _find_suspicious_paths(
        self,
        graph: nx.Graph,
        attacker_nodes: List[str],
        target_node: Optional[str],
    ) -> List[
        List[str]
    ]:

        if (
            target_node is None
            or not attacker_nodes
        ):

            return []

        if target_node not in graph:

            return []

        paths = []

        for attacker in attacker_nodes:

            if attacker not in graph:
                continue

            try:

                path = (
                    nx.shortest_path(
                        graph,
                        source=attacker,
                        target=target_node,
                    )
                )

                paths.append(
                    [
                        str(node)
                        for node
                        in path
                    ]
                )

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound,
            ):

                continue

        return paths

    # --------------------------------------------------
    # HUMAN READABLE REASONS
    # --------------------------------------------------

    def _generate_reasons(
        self,
        graph: nx.Graph,
        target_node: Optional[str],
        attacker_nodes: List[str],
        node_risk: Dict[
            str,
            float
        ],
        metrics: Dict[
            str,
            Dict[str, float]
        ],
        suspicious_paths: List[
            List[str]
        ],
    ) -> List[str]:

        reasons = []

        if (
            target_node is None
            or target_node not in graph
        ):

            reasons.append(
                "The explanation is based on global "
                "network topology because no target node "
                "was specified."
            )

            return reasons

        degree = (
            metrics["degree"].get(
                target_node,
                0.0,
            )
        )

        betweenness = (
            metrics["betweenness"].get(
                target_node,
                0.0,
            )
        )

        closeness = (
            metrics["closeness"].get(
                target_node,
                0.0,
            )
        )

        risk = (
            node_risk.get(
                target_node,
                0.0,
            )
        )

        if degree >= 0.30:

            reasons.append(
                "The predicted target has high network "
                "connectivity, making it an important "
                "communication node."
            )

        if betweenness >= 0.10:

            reasons.append(
                "The predicted target acts as a bridge "
                "between multiple parts of the network."
            )

        if closeness >= 0.30:

            reasons.append(
                "The predicted target can communicate "
                "efficiently with many network nodes."
            )

        if risk >= 0.50:

            reasons.append(
                "The predicted target already has an "
                "elevated anomaly or attack-risk score."
            )

        direct_attacker_connection = False

        for attacker in attacker_nodes:

            if (
                attacker in graph
                and graph.has_edge(
                    attacker,
                    target_node,
                )
            ):

                direct_attacker_connection = True

                break

        if direct_attacker_connection:

            reasons.append(
                "The predicted target has direct "
                "communication with a suspected attacker."
            )

        if suspicious_paths:

            reasons.append(
                "A communication path exists between "
                "suspected attacker nodes and the "
                "predicted target."
            )

        if not reasons:

            reasons.append(
                "The target was selected primarily "
                "because of its relative structural "
                "importance in the network graph."
            )

        return reasons

    # --------------------------------------------------
    # GRAPH SUMMARY
    # --------------------------------------------------

    def _graph_summary(
        self,
        graph: nx.Graph,
        attacker_nodes: List[str],
        node_risk: Dict[
            str,
            float
        ],
    ) -> Dict[
        str,
        Any
    ]:

        node_count = (
            graph.number_of_nodes()
        )

        edge_count = (
            graph.number_of_edges()
        )

        try:

            density = (
                nx.density(
                    graph
                )
            )

        except Exception:

            density = 0.0

        try:

            components = (
                nx.number_connected_components(
                    graph.to_undirected()
                )
            )

        except Exception:

            components = 0

        high_risk_nodes = [
            str(node)
            for node, risk
            in node_risk.items()
            if risk >= 0.50
        ]

        return {
            "nodes": int(
                node_count
            ),

            "edges": int(
                edge_count
            ),

            "density": float(
                density
            ),

            "connected_components": int(
                components
            ),

            "attacker_nodes": [
                str(node)
                for node
                in attacker_nodes
            ],

            "high_risk_nodes": high_risk_nodes,
        }

    # --------------------------------------------------
    # ATTACKER CONNECTIVITY
    # --------------------------------------------------

    def _attacker_connectivity(
        self,
        graph: nx.Graph,
        node: Any,
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

        return float(
            min(
                connections
                / len(attacker_nodes),
                1.0,
            )
        )

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    @staticmethod
    def _normalize_weight(
        weight: float,
    ) -> float:

        if weight <= 0:
            return 0.0

        return float(
            weight
            / (
                weight + 1.0
            )
        )

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