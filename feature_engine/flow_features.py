from future import annotations

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

def __init__(
    self,
    flow_timeout_seconds: int = 60,
) -> None:
    
    self.flow_timeout_seconds = flow_timeout_seconds

def transform(
    self,
    events: pl.DataFrame,
) -> pl.DataFrame:
    """
    Convert canonical network events into flow-level features.
    """

    self._validate_input(events)

    if events.is_empty():
        return self._empty_flow_dataframe()

    events = self._prepare_events(events)

    flow_features = (
        events
        .group_by(list(self.DEFAULT_FLOW_KEYS))
        .agg(
            [
                pl.col("timestamp")
                .min()
                .alias("flow_start"),

                pl.col("timestamp")
                .max()
                .alias("flow_end"),

                pl.col("packet_count")
                .sum()
                .alias("packet_count"),

                pl.col("byte_count")
                .sum()
                .alias("byte_count"),

                pl.col("byte_count")
                .mean()
                .alias("packet_size_mean"),

                pl.col("byte_count")
                .median()
                .alias("packet_size_median"),

                pl.col("byte_count")
                .std()
                .fill_null(0.0)
                .alias("packet_size_std"),

                pl.col("byte_count")
                .min()
                .alias("packet_size_min"),

                pl.col("byte_count")
                .max()
                .alias("packet_size_max"),

                pl.col("ttl")
                .mean()
                .alias("ttl_mean"),

                pl.col("ttl")
                .std()
                .fill_null(0.0)
                .alias("ttl_std"),

                pl.col("tcp_window_size")
                .mean()
                .alias("tcp_window_mean"),

                pl.col("tcp_window_size")
                .std()
                .fill_null(0.0)
                .alias("tcp_window_std"),

                pl.col("tcp_flags")
                .drop_nulls()
                .n_unique()
                .alias("unique_tcp_flags"),

                pl.col("source_type")
                .first()
                .alias("source_type"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("flow_end")
                    .cast(pl.Int64)
                    -
                    pl.col("flow_start")
                    .cast(pl.Int64)
                )
                .cast(pl.Float64)
                .truediv(1_000_000)
                .alias("flow_duration_seconds")
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("flow_duration_seconds") > 0
                )
                .then(
                    pl.col("packet_count")
                    /
                    pl.col("flow_duration_seconds")
                )
                .otherwise(
                    pl.col("packet_count")
                )
                .alias("packets_per_second"),

                pl.when(
                    pl.col("flow_duration_seconds") > 0
                )
                .then(
                    pl.col("byte_count")
                    /
                    pl.col("flow_duration_seconds")
                )
                .otherwise(
                    pl.col("byte_count")
                )
                .alias("bytes_per_second"),
            ]
        )
        .sort("flow_start")
    )

    return flow_features

def add_directional_features(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add source and destination behavioural features.

    Examples:

    - Number of unique destinations contacted by source
    - Number of unique source ports
    - Number of unique source IPs contacting destination
    - Destination connection popularity
    """

    required = {
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
    }

    missing = required - set(flows.columns)

    if missing:
        raise ValueError(
            f"Missing required flow columns: {sorted(missing)}"
        )

    source_features = (
        flows
        .group_by("src_ip")
        .agg(
            [
                pl.col("dst_ip")
                .n_unique()
                .alias("src_unique_destinations"),

                pl.col("dst_port")
                .n_unique()
                .alias("src_unique_destination_ports"),

                pl.col("src_port")
                .n_unique()
                .alias("src_unique_source_ports"),

                pl.len()
                .alias("src_flow_count"),

                pl.col("byte_count")
                .sum()
                .alias("src_total_bytes"),
            ]
        )
    )

    destination_features = (
        flows
        .group_by("dst_ip")
        .agg(
            [
                pl.col("src_ip")
                .n_unique()
                .alias("dst_unique_sources"),

                pl.col("dst_port")
                .n_unique()
                .alias("dst_unique_ports"),

                pl.len()
                .alias("dst_flow_count"),

                pl.col("byte_count")
                .sum()
                .alias("dst_total_bytes"),
            ]
        )
    )

    flows = flows.join(
        source_features,
        on="src_ip",
        how="left",
    )

    flows = flows.join(
        destination_features,
        on="dst_ip",
        how="left",
    )

    return flows

def add_ratio_features(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add useful ratios for ML models.
    """

    return flows.with_columns(
        [
            pl.when(
                pl.col("packet_count") > 0
            )
            .then(
                pl.col("byte_count")
                /
                pl.col("packet_count")
            )
            .otherwise(0.0)
            .alias("bytes_per_packet"),

            pl.when(
                pl.col("src_flow_count") > 0
            )
            .then(
                pl.col("src_total_bytes")
                /
                pl.col("src_flow_count")
            )
            .otherwise(0.0)
            .alias("src_average_bytes_per_flow"),

            pl.when(
                pl.col("dst_flow_count") > 0
            )
            .then(
                pl.col("dst_total_bytes")
                /
                pl.col("dst_flow_count")
            )
            .otherwise(0.0)
            .alias("dst_average_bytes_per_flow"),
        ]
    )

def build(
    self,
    events: pl.DataFrame,
) -> pl.DataFrame:
    """
    Full flow feature pipeline.

    Canonical Events
        ↓
    Flow Aggregation
        ↓
    Directional Features
        ↓
    Ratio Features
    """

    flows = self.transform(events)

    flows = self.add_directional_features(flows)

    flows = self.add_ratio_features(flows)

    return flows

@staticmethod
def _prepare_events(
    events: pl.DataFrame,
) -> pl.DataFrame:
    """
    Ensure expected datatypes.
    """

    return events.with_columns(
        [
            pl.col("timestamp")
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                ),
                strict=False,
            ),

            pl.col("packet_count")
            .cast(pl.Int64, strict=False)
            .fill_null(1),

            pl.col("byte_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
        ]
    )

@classmethod
def _validate_input(
    cls,
    events: pl.DataFrame,
) -> None:

    required = set(cls.DEFAULT_FLOW_KEYS) | {
        "timestamp",
        "packet_count",
        "byte_count",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            "Input events are missing required columns: "
            f"{sorted(missing)}"
        )

@staticmethod
def _empty_flow_dataframe() -> pl.DataFrame:

    return pl.DataFrame(
        schema={
            "src_ip": pl.String,
            "dst_ip": pl.String,
            "src_port": pl.Int64,
            "dst_port": pl.Int64,
            "protocol": pl.String,
            "flow_start": pl.Datetime(
                "us",
                time_zone="UTC",
            ),
            "flow_end": pl.Datetime(
                "us",
                time_zone="UTC",
            ),
            "packet_count": pl.Int64,
            "byte_count": pl.Float64,
            "flow_duration_seconds": pl.Float64,
        }
    )

