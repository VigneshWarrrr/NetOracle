from __future__ import annotations

from typing import Sequence

import polars as pl


class FlowFeatureEngine:
    DEFAULT_FLOW_KEYS: Sequence[str] = (
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
    )

    def __init__(self, flow_timeout_seconds: int = 60) -> None:
        if flow_timeout_seconds <= 0:
            raise ValueError("flow_timeout_seconds must be greater than zero.")
        self.flow_timeout_seconds = flow_timeout_seconds

    def transform(self, events: pl.DataFrame) -> pl.DataFrame:
        """Convert canonical packet events into timeout-segmented flows."""
        self._validate_input(events)
        if events.is_empty():
            return self._empty_flow_dataframe()

        events = self._segment_flows(self._prepare_events(events))
        flow_keys = [*self.DEFAULT_FLOW_KEYS, "_flow_segment"]
        return (
            events.group_by(flow_keys)
            .agg(
                [
                    pl.col("timestamp").min().alias("flow_start"),
                    pl.col("timestamp").max().alias("flow_end"),
                    pl.col("packet_count").sum().alias("packet_count"),
                    pl.col("byte_count").sum().alias("byte_count"),
                    pl.col("byte_count").mean().alias("packet_size_mean"),
                    pl.col("byte_count").median().alias("packet_size_median"),
                    pl.col("byte_count").std().fill_null(0.0).alias("packet_size_std"),
                    pl.col("byte_count").min().alias("packet_size_min"),
                    pl.col("byte_count").max().alias("packet_size_max"),
                    pl.col("ttl").mean().alias("ttl_mean"),
                    pl.col("ttl").std().fill_null(0.0).alias("ttl_std"),
                    pl.col("tcp_window_size").mean().alias("tcp_window_mean"),
                    pl.col("tcp_window_size").std().fill_null(0.0).alias("tcp_window_std"),
                    pl.col("tcp_flags").drop_nulls().n_unique().alias("unique_tcp_flags"),
                    pl.col("source_type").first().alias("source_type"),
                ]
            )
            .drop("_flow_segment")
            .with_columns(
                (
                    (pl.col("flow_end").cast(pl.Int64) - pl.col("flow_start").cast(pl.Int64))
                    .cast(pl.Float64)
                    .truediv(1_000_000)
                    .alias("flow_duration_seconds")
                )
            )
            .with_columns(
                [
                    pl.when(pl.col("flow_duration_seconds") > 0)
                    .then(pl.col("packet_count") / pl.col("flow_duration_seconds"))
                    .otherwise(pl.col("packet_count"))
                    .alias("packets_per_second"),
                    pl.when(pl.col("flow_duration_seconds") > 0)
                    .then(pl.col("byte_count") / pl.col("flow_duration_seconds"))
                    .otherwise(pl.col("byte_count"))
                    .alias("bytes_per_second"),
                ]
            )
            .sort("flow_start")
        )

    def add_directional_features(self, flows: pl.DataFrame) -> pl.DataFrame:
        required = {"src_ip", "dst_ip", "src_port", "dst_port"}
        missing = required - set(flows.columns)
        if missing:
            raise ValueError(f"Missing required flow columns: {sorted(missing)}")

        source_features = flows.group_by("src_ip").agg(
            [
                pl.col("dst_ip").n_unique().alias("src_unique_destinations"),
                pl.col("dst_port").n_unique().alias("src_unique_destination_ports"),
                pl.col("src_port").n_unique().alias("src_unique_source_ports"),
                pl.len().alias("src_flow_count"),
                pl.col("byte_count").sum().alias("src_total_bytes"),
            ]
        )
        destination_features = flows.group_by("dst_ip").agg(
            [
                pl.col("src_ip").n_unique().alias("dst_unique_sources"),
                pl.col("dst_port").n_unique().alias("dst_unique_ports"),
                pl.len().alias("dst_flow_count"),
                pl.col("byte_count").sum().alias("dst_total_bytes"),
            ]
        )
        return flows.join(source_features, on="src_ip", how="left").join(
            destination_features, on="dst_ip", how="left"
        )

    def add_ratio_features(self, flows: pl.DataFrame) -> pl.DataFrame:
        return flows.with_columns(
            [
                pl.when(pl.col("packet_count") > 0)
                .then(pl.col("byte_count") / pl.col("packet_count"))
                .otherwise(0.0)
                .alias("bytes_per_packet"),
                pl.when(pl.col("src_flow_count") > 0)
                .then(pl.col("src_total_bytes") / pl.col("src_flow_count"))
                .otherwise(0.0)
                .alias("src_average_bytes_per_flow"),
                pl.when(pl.col("dst_flow_count") > 0)
                .then(pl.col("dst_total_bytes") / pl.col("dst_flow_count"))
                .otherwise(0.0)
                .alias("dst_average_bytes_per_flow"),
            ]
        )

    def build(self, events: pl.DataFrame) -> pl.DataFrame:
        flows = self.transform(events)
        return self.add_ratio_features(self.add_directional_features(flows))

    def _segment_flows(self, events: pl.DataFrame) -> pl.DataFrame:
        """Start a new flow when a same-key packet gap exceeds the timeout."""
        keys = list(self.DEFAULT_FLOW_KEYS)
        return (
            events.sort([*keys, "timestamp"])
            .with_columns(
                pl.col("timestamp")
                .diff()
                .over(keys)
                .dt.total_seconds()
                .fill_null(0.0)
                .gt(float(self.flow_timeout_seconds))
                .cast(pl.Int64)
                .alias("_flow_boundary")
            )
            .with_columns(
                pl.col("_flow_boundary").cum_sum().over(keys).alias("_flow_segment")
            )
            .drop("_flow_boundary")
        )

    @staticmethod
    def _prepare_events(events: pl.DataFrame) -> pl.DataFrame:
        return events.with_columns(
            [
                pl.col("timestamp")
                .cast(pl.String, strict=False)
                .str.to_datetime(time_zone="UTC", strict=False)
                .alias("timestamp"),
                pl.col("packet_count").cast(pl.Int64, strict=False).fill_null(1),
                pl.col("byte_count").cast(pl.Float64, strict=False).fill_null(0.0),
            ]
        )

    @classmethod
    def _validate_input(cls, events: pl.DataFrame) -> None:
        required = set(cls.DEFAULT_FLOW_KEYS) | {"timestamp", "packet_count", "byte_count"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"Input events are missing required columns: {sorted(missing)}")

    @staticmethod
    def _empty_flow_dataframe() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "src_ip": pl.String,
                "dst_ip": pl.String,
                "src_port": pl.Int64,
                "dst_port": pl.Int64,
                "protocol": pl.String,
                "flow_start": pl.Datetime("us", time_zone="UTC"),
                "flow_end": pl.Datetime("us", time_zone="UTC"),
                "packet_count": pl.Int64,
                "byte_count": pl.Float64,
                "flow_duration_seconds": pl.Float64,
            }
        )
