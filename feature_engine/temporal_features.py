from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl

class TemporalFeatureEngine:
    def __init__(self, window_seconds: int = 10) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")
        self.window_seconds = window_seconds

    def build_windows(self, flows: pl.DataFrame) -> pl.DataFrame:
        return build_windows(self, flows)

    def add_entropy_features(self, flows: pl.DataFrame) -> pl.DataFrame:
        return add_entropy_features(self, flows)

    def _add_rate_features(self, windows: pl.DataFrame) -> pl.DataFrame:
        return _add_rate_features(self, windows)

    def add_rolling_features(
        self,
        windows: pl.DataFrame,
        columns: Sequence[str] | None = None,
        rolling_windows: Sequence[int] = (3, 6, 12),
    ) -> pl.DataFrame:
        return add_rolling_features(self, windows, columns, rolling_windows)

    def add_change_features(
        self,
        windows: pl.DataFrame,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        return add_change_features(self, windows, columns)

    def create_sequences(
        self,
        windows: pl.DataFrame,
        feature_columns: Iterable[str],
        sequence_length: int = 10,
        prediction_horizon: int = 1,
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        return create_sequences(self, windows, feature_columns, sequence_length, prediction_horizon)

    def build(self, flows: pl.DataFrame) -> pl.DataFrame:
        return build(self, flows)

    @staticmethod
    def _validate_input(flows: pl.DataFrame) -> None:
        _validate_input(flows)

    @staticmethod
    def _empty_window_dataframe() -> pl.DataFrame:
        return _empty_window_dataframe()

def build_windows(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate flow data into fixed time windows.
    """

    self._validate_input(flows)

    if flows.is_empty():
        return self._empty_window_dataframe()

    window_size = f"{self.window_seconds}s"

    flows = flows.sort("flow_start")

    windowed = (
        flows
        .group_by_dynamic(
            index_column="flow_start",
            every=window_size,
            period=window_size,
            closed="left",
        )
        .agg(
            [
                pl.len()
                .alias("flow_count"),

                pl.col("src_ip")
                .n_unique()
                .alias("unique_sources"),

                pl.col("dst_ip")
                .n_unique()
                .alias("unique_destinations"),

                pl.col("src_port")
                .n_unique()
                .alias("unique_source_ports"),

                pl.col("dst_port")
                .n_unique()
                .alias("unique_destination_ports"),

                pl.col("protocol")
                .n_unique()
                .alias("unique_protocols"),

                pl.col("packet_count")
                .sum()
                .alias("total_packets"),

                pl.col("byte_count")
                .sum()
                .alias("total_bytes"),

                pl.col("flow_duration_seconds")
                .mean()
                .fill_null(0.0)
                .alias("mean_flow_duration"),

                pl.col("flow_duration_seconds")
                .std()
                .fill_null(0.0)
                .alias("std_flow_duration"),

                pl.col("packet_size_mean")
                .mean()
                .fill_null(0.0)
                .alias("mean_packet_size"),

                pl.col("packets_per_second")
                .mean()
                .fill_null(0.0)
                .alias("mean_packet_rate"),

                pl.col("bytes_per_second")
                .mean()
                .fill_null(0.0)
                .alias("mean_byte_rate"),
            ]
        )
        .rename(
            {
                "flow_start": "window_start",
            }
        )
        .sort("window_start")
    )

    return self._add_rate_features(windowed)

def _add_rate_features(
    self,
    windows: pl.DataFrame,
) -> pl.DataFrame:

    duration = float(self.window_seconds)

    return windows.with_columns(
        [
            (
                pl.col("total_packets") / duration
            ).alias("packets_per_window_second"),

            (
                pl.col("total_bytes") / duration
            ).alias("bytes_per_window_second"),

            pl.when(
                pl.col("flow_count") > 0
            )
            .then(
                pl.col("total_packets")
                /
                pl.col("flow_count")
            )
            .otherwise(0.0)
            .alias("average_packets_per_flow"),

            pl.when(
                pl.col("flow_count") > 0
            )
            .then(
                pl.col("total_bytes")
                /
                pl.col("flow_count")
            )
            .otherwise(0.0)
            .alias("average_bytes_per_flow"),
        ]
    )

def add_entropy_features(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate entropy-based behaviour features.

    High destination/port entropy can be useful for identifying:

    - Scanning
    - Distributed behaviour
    - Abnormal communication patterns
    """

    self._validate_input(flows)

    flows = flows.sort("flow_start")

    window_size = f"{self.window_seconds}s"

    return (
        flows
        .group_by_dynamic(
            index_column="flow_start",
            every=window_size,
            period=window_size,
            closed="left",
        )
        .agg(
            [
                pl.col("dst_ip")
                .n_unique()
                .alias("destination_diversity"),

                pl.col("dst_port")
                .n_unique()
                .alias("port_diversity"),

                pl.col("src_ip")
                .n_unique()
                .alias("source_diversity"),
            ]
        )
        .rename(
            {
                "flow_start": "window_start",
            }
        )
        .sort("window_start")
    )

def add_rolling_features(
    self,
    windows: pl.DataFrame,
    columns: Sequence[str] | None = None,
    rolling_windows: Sequence[int] = (3, 6, 12),
) -> pl.DataFrame:
    """
    Add rolling mean and standard deviation features.

    Example:

    Current flow count
    Previous 3-window average
    Previous 6-window average
    Previous 12-window average
    """

    if windows.is_empty():
        return windows

    if columns is None:
        columns = [
            "flow_count",
            "total_packets",
            "total_bytes",
            "unique_destinations",
            "unique_destination_ports",
        ]

    expressions = []

    for column in columns:

        if column not in windows.columns:
            continue

        for window_size in rolling_windows:

            expressions.append(
                pl.col(column)
                .rolling_mean(
                    window_size=window_size,
                    min_samples=1,
                )
                .alias(
                    f"{column}_rolling_mean_{window_size}"
                )
            )

            expressions.append(
                pl.col(column)
                .rolling_std(
                    window_size=window_size,
                    min_samples=1,
                )
                .fill_null(0.0)
                .alias(
                    f"{column}_rolling_std_{window_size}"
                )
            )

    return windows.with_columns(expressions)

def add_change_features(
    self,
    windows: pl.DataFrame,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Add window-to-window changes.

    Useful for detecting sudden escalation.

    Example:

    SYN/connection activity:

    10 → 12 → 11 → 80

    Change features highlight the sudden jump.
    """

    if windows.is_empty():
        return windows

    if columns is None:
        columns = [
            "flow_count",
            "total_packets",
            "total_bytes",
            "unique_destinations",
            "unique_destination_ports",
        ]

    expressions = []

    for column in columns:

        if column not in windows.columns:
            continue

        expressions.extend(
            [
                pl.col(column)
                .diff()
                .fill_null(0)
                .alias(
                    f"{column}_change"
                ),

                (
                    pl.col(column)
                    .pct_change()
                    .fill_null(0.0)
                )
                .alias(
                    f"{column}_pct_change"
                ),
            ]
        )

    return windows.with_columns(expressions)

def create_sequences(
    self,
    windows: pl.DataFrame,
    feature_columns: Iterable[str],
    sequence_length: int = 10,
    prediction_horizon: int = 1,
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """
    Convert windowed data into ML sequences.

    Example:

    sequence_length = 5

    Input:
    S1 S2 S3 S4 S5

    Target:
    S6

    Returns
    -------
    X:
        List of input sequences

    y:
        Corresponding future states
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if prediction_horizon <= 0:
        raise ValueError(
            "prediction_horizon must be greater than zero."
        )

    feature_columns = list(feature_columns)

    missing = (
        set(feature_columns)
        -
        set(windows.columns)
    )

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    matrix = (
        windows
        .select(feature_columns)
        .fill_null(0.0)
        .cast(pl.Float64)
        .to_numpy()
    )

    X = []
    y = []

    total = (
        len(matrix)
        -
        sequence_length
        -
        prediction_horizon
        +
        1
    )

    for start in range(max(total, 0)):

        input_end = (
            start + sequence_length
        )

        target_index = (
            input_end
            +
            prediction_horizon
            -
            1
        )

        X.append(
            matrix[
                start:input_end
            ].tolist()
        )

        y.append(
            matrix[
                target_index
            ].tolist()
        )

    return X, y

def build(
    self,
    flows: pl.DataFrame,
) -> pl.DataFrame:
    """
    Complete temporal feature pipeline.
    """

    windows = self.build_windows(flows)

    diversity = self.add_entropy_features(flows)

    if not diversity.is_empty():

        windows = windows.join(
            diversity,
            on="window_start",
            how="left",
        )

    windows = self.add_rolling_features(windows)

    windows = self.add_change_features(windows)

    return windows

@staticmethod
def _validate_input(
    flows: pl.DataFrame,
) -> None:

    required = {
        "flow_start",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "packet_count",
        "byte_count",
        "flow_duration_seconds",
    }

    missing = (
        required
        -
        set(flows.columns)
    )

    if missing:
        raise ValueError(
            "Input flows are missing required columns: "
            f"{sorted(missing)}"
        )

@staticmethod
def _empty_window_dataframe() -> pl.DataFrame:

    return pl.DataFrame(
        schema={
            "window_start": pl.Datetime(
                "us",
                time_zone="UTC",
            ),
            "flow_count": pl.Int64,
            "total_packets": pl.Float64,
            "total_bytes": pl.Float64,
        }
    )

