from future import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import polars as pl

CANONICAL_COLUMNS = [
"timestamp",
"src_ip",
"dst_ip",
"src_port",
"dst_port",
"protocol",
"packet_count",
"byte_count",
"duration",
"tcp_flags",
"ttl",
"tcp_window_size",
"source_type",
]

class CSVReaderService:

 DEFAULT_COLUMN_MAPPING = {
    "timestamp": [
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "ts",
    ],
    "src_ip": [
        "src_ip",
        "Source IP",
        "SourceIP",
        "source_ip",
    ],
    "dst_ip": [
        "dst_ip",
        "Destination IP",
        "DestinationIP",
        "destination_ip",
    ],
    "src_port": [
        "src_port",
        "Source Port",
        "SourcePort",
        "source_port",
    ],
    "dst_port": [
        "dst_port",
        "Destination Port",
        "DestinationPort",
        "destination_port",
    ],
    "protocol": [
        "protocol",
        "Protocol",
        "proto",
    ],
    "packet_count": [
        "packet_count",
        "Total Fwd Packets",
        "Total Packets",
        "Packets",
    ],
    "byte_count": [
        "byte_count",
        "Total Length of Fwd Packets",
        "Total Bytes",
        "Bytes",
    ],
    "duration": [
        "duration",
        "Flow Duration",
        "Duration",
    ],
    "tcp_flags": [
        "tcp_flags",
        "TCP Flags",
        "Flags",
    ],
    "ttl": [
        "ttl",
        "TTL",
    ],
    "tcp_window_size": [
        "tcp_window_size",
        "TCP Window Size",
        "Window Size",
    ],
}

def __init__(
    self,
    column_mapping: Optional[Dict[str, str]] = None,
    separator: str = ",",
) -> None:
    """
    column_mapping:
        Optional explicit mapping:

        {
            "timestamp": "Timestamp",
            "src_ip": "Source IP",
            ...
        }
    """

    self.column_mapping = column_mapping or {}
    self.separator = separator

def read(
    self,
    file_path: Union[str, Path],
) -> pl.DataFrame:
    """
    Read and normalize a CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pl.read_csv(
        file_path,
        separator=self.separator,
        ignore_errors=True,
        infer_schema_length=10_000,
        truncate_ragged_lines=True,
    )

    return self._normalize(df)

def scan(
    self,
    file_path: Union[str, Path],
) -> pl.LazyFrame:
    """
    Lazily scan a CSV file.

    Recommended for very large datasets.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    return pl.scan_csv(
        file_path,
        separator=self.separator,
        infer_schema_length=10_000,
        ignore_errors=True,
    )

def _normalize(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert arbitrary dataset columns to canonical columns.
    """

    expressions = []

    for canonical_column in CANONICAL_COLUMNS:

        if canonical_column == "source_type":
            expressions.append(
                pl.lit("csv").alias("source_type")
            )
            continue

        source_column = self._resolve_column(
            canonical_column,
            df.columns,
        )

        if source_column is None:
            expressions.append(
                pl.lit(None).alias(canonical_column)
            )

        else:
            expressions.append(
                pl.col(source_column).alias(canonical_column)
            )

    normalized = df.select(expressions)

    normalized = normalized.with_columns(
        [
            pl.col("timestamp")
            .cast(pl.String)
            .str.to_datetime(
                strict=False,
                time_zone="UTC",
            )
            .alias("timestamp"),

            pl.col("src_ip")
            .cast(pl.String, strict=False),

            pl.col("dst_ip")
            .cast(pl.String, strict=False),

            pl.col("src_port")
            .cast(pl.Int64, strict=False),

            pl.col("dst_port")
            .cast(pl.Int64, strict=False),

            pl.col("protocol")
            .cast(pl.String, strict=False),

            pl.col("packet_count")
            .cast(pl.Int64, strict=False),

            pl.col("byte_count")
            .cast(pl.Int64, strict=False),

            pl.col("duration")
            .cast(pl.Float64, strict=False),

            pl.col("tcp_flags")
            .cast(pl.String, strict=False),

            pl.col("ttl")
            .cast(pl.Int64, strict=False),

            pl.col("tcp_window_size")
            .cast(pl.Int64, strict=False),

            pl.col("source_type")
            .cast(pl.String),
        ]
    )

    return normalized

def _resolve_column(
    self,
    canonical_column: str,
    available_columns: list[str],
) -> Optional[str]:
    """
    Find the dataset column corresponding to a canonical field.
    """

    if canonical_column in self.column_mapping:
        mapped_column = self.column_mapping[canonical_column]

        if mapped_column in available_columns:
            return mapped_column

        raise ValueError(
            f"Mapped column '{mapped_column}' for "
            f"'{canonical_column}' was not found."
        )

    candidates = self.DEFAULT_COLUMN_MAPPING.get(
        canonical_column,
        [],
    )

    normalized_lookup = {
        self._normalize_name(column): column
        for column in available_columns
    }

    for candidate in candidates:

        if candidate in available_columns:
            return candidate

        normalized_candidate = self._normalize_name(candidate)

        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    return None

@staticmethod
def _normalize_name(name: str) -> str:
    """
    Normalize column names for flexible matching.
    """

    return (
        name.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )

