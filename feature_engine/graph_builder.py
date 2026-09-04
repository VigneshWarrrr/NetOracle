from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import polars as pl

@dataclass
class GraphSnapshot:
    timestamp: Any
    graph: nx.DiGraph

def build_static_graph(
    self,
    flows: pl.DataFrame,
) -> nx.DiGraph:
    """
    Build one graph representing all flows.
    """

    self._validate_input(flows)

    graph = nx.DiGraph()

    if flows.is_empty():
        return graph

    edge_features = (
        flows
        .group_by(
            [
                "src_ip",
                "dst_ip",
            ]
        )
        .agg(
            [
                pl.len()
                .alias("flow_count"),

                pl.col("packet_count")
                .sum()
                .alias("packet_count"),

                pl.col("byte_count")
                .sum()
                .alias("byte_count"),

                pl.col("src_port")
                .n_unique()
                .alias("unique_source_ports"),

                pl.col("dst_port")
                .n_unique()
                .alias("unique_destination_ports"),

                pl.col("protocol")
                .n_unique()
                .alias("unique_protocols"),

                pl.col("flow_duration_seconds")
                .mean()
                .fill_null(0.0)
                .alias("mean_duration"),
            ]
        )
    )

    for row in edge_features.iter_rows(
        named=True
    ):

        src = row["src_ip"]
        dst = row["dst_ip"]

        if src is None or dst is None:
            continue

        graph.add_node(
            src,
            node_type="host",
        )

        graph.add_node(
            dst,
            node_type="host",
        )

        graph.add_edge(
            src,
            dst,
            **{
                key: value
                for key, value in row.items()
                if key not in (
                    "src_ip",
                    "dst_ip",
                )
            },
        )

    self._add_node_statistics(graph)

    return graph

def build_dynamic_graphs(
    self,
    flows: pl.DataFrame,
    window_seconds: int = 10,
) -> list[GraphSnapshot]:
    """
    Build time-based graph snapshots.

    Example:

    G(t1)
    G(t2)
    G(t3)

    These snapshots can later be used for
    Temporal GNN training.
    """

    if window_seconds <= 0:
        raise ValueError(
            "window_seconds must be greater than zero."
        )

    self._validate_input(flows)

    if flows.is_empty():
        return []

    window_size = f"{window_seconds}s"

    flows = (
        flows
        .sort("flow_start")
        .with_columns(
            pl.col("flow_start")
            .dt.truncate(window_size)
            .alias("graph_window")
        )
    )

    snapshots = []

    for window, window_flows in flows.group_by(
        "graph_window",
        maintain_order=True,
    ):

        graph = self.build_static_graph(
            window_flows.drop("graph_window")
        )

        timestamp = (
            window[0]
            if isinstance(window, tuple)
            else window
        )

        snapshots.append(
            GraphSnapshot(
                timestamp=timestamp,
                graph=graph,
            )
        )

    return snapshots

def build_node_features(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build numerical node features.

    These can later be converted into tensors for GNN models.

    Features:

    inbound flow count
    outbound flow count
    inbound bytes
    outbound bytes
    unique peers
    """

    self._validate_input(flows)

    outgoing = (
        flows
        .group_by("src_ip")
        .agg(
            [
                pl.len()
                .alias("out_flow_count"),

                pl.col("byte_count")
                .sum()
                .alias("out_bytes"),

                pl.col("dst_ip")
                .n_unique()
                .alias("out_unique_peers"),
            ]
        )
        .rename(
            {
                "src_ip": "node",
            }
        )
    )

    incoming = (
        flows
        .group_by("dst_ip")
        .agg(
            [
                pl.len()
                .alias("in_flow_count"),

                pl.col("byte_count")
                .sum()
                .alias("in_bytes"),

                pl.col("src_ip")
                .n_unique()
                .alias("in_unique_peers"),
            ]
        )
        .rename(
            {
                "dst_ip": "node",
            }
        )
    )

    node_features = outgoing.join(
        incoming,
        on="node",
        how="full",
        coalesce=True,
    )

    numeric_columns = [
        column
        for column in node_features.columns
        if column != "node"
    ]

    return node_features.with_columns(
        [
            pl.col(column)
            .fill_null(0)
            for column in numeric_columns
        ]
    )

def graph_to_edge_dataframe(
    self,
    graph: nx.DiGraph,
) -> pl.DataFrame:
    """
    Convert a NetworkX graph back into an edge DataFrame.

    Useful for debugging and visualization.
    """

    rows = []

    for source, destination, data in graph.edges(
        data=True
    ):

        row = {
            "src_ip": source,
            "dst_ip": destination,
            **data,
        }

        rows.append(row)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)

def get_suspicious_nodes(
    self,
    graph: nx.DiGraph,
    top_k: int = 10,
) -> list[dict]:
    """
    Return nodes with the highest communication activity.

    This is NOT the final AI-based suspiciousness score.

    It provides a simple graph heuristic useful for:

    - Debugging
    - Baseline comparison
    - Dashboard visualization
    """

    scores = []

    for node in graph.nodes:

        in_degree = graph.in_degree(node)

        out_degree = graph.out_degree(node)

        total_degree = (
            in_degree
            +
            out_degree
        )

        scores.append(
            {
                "node": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
            }
        )

    scores.sort(
        key=lambda item: item["total_degree"],
        reverse=True,
    )

    return scores[:top_k]

@staticmethod
def _add_node_statistics(
    graph: nx.DiGraph,
) -> None:
    """
    Add graph-based node statistics.
    """

    for node in graph.nodes:

        graph.nodes[node][
            "in_degree"
        ] = graph.in_degree(node)

        graph.nodes[node][
            "out_degree"
        ] = graph.out_degree(node)

        graph.nodes[node][
            "total_degree"
        ] = (
            graph.in_degree(node)
            +
            graph.out_degree(node)
        )

@staticmethod
def _validate_input(
    flows: pl.DataFrame,
) -> None:

    required = {
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "packet_count",
        "byte_count",
        "flow_duration_seconds",
        "flow_start",
    }

    missing = (
        required
        -
        set(flows.columns)
    )

    if missing:
        raise ValueError(
            "Flow data is missing required columns: "
            f"{sorted(missing)}"
        )


class NetworkGraphBuilder:
    """Build static and time-windowed network graphs from flow data."""

    def build_static_graph(
        self,
        flows: pl.DataFrame,
    ) -> nx.DiGraph:
        return build_static_graph(self, flows)

    def build_dynamic_graphs(
        self,
        flows: pl.DataFrame,
        window_seconds: int = 10,
    ) -> list[GraphSnapshot]:
        return build_dynamic_graphs(self, flows, window_seconds)

    def build_node_features(
        self,
        flows: pl.DataFrame,
    ) -> pl.DataFrame:
        return build_node_features(self, flows)

    def graph_to_edge_dataframe(
        self,
        graph: nx.DiGraph,
    ) -> pl.DataFrame:
        return graph_to_edge_dataframe(self, graph)

    def get_suspicious_nodes(
        self,
        graph: nx.DiGraph,
        top_k: int = 10,
    ) -> list[dict]:
        return get_suspicious_nodes(self, graph, top_k)

    @staticmethod
    def _add_node_statistics(
        graph: nx.DiGraph,
    ) -> None:
        _add_node_statistics(graph)

    @staticmethod
    def _validate_input(
        flows: pl.DataFrame,
    ) -> None:
        _validate_input(flows)

