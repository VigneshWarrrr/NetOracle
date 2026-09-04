from __future__ import annotations

from pathlib import Path
from typing import Union

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

class ZeekReaderService:

 def read(
    self,
    file_path: Union[str, Path],
    log_type: str = "auto",
) -> pl.DataFrame:
    """
    Read a Zeek log.

    Parameters
    ----------
    file_path:
        Path to the Zeek log.

    log_type:
        "auto"
        "json"
        "tsv"
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Zeek log not found: {file_path}"
        )

    if log_type == "auto":
        log_type = self._detect_log_type(file_path)

    if log_type == "json":
        df = self._read_json(file_path)

    elif log_type == "tsv":
        df = self._read_tsv(file_path)

    else:
        raise ValueError(
            "log_type must be one of: auto, json, tsv"
        )

    return self._normalize(df)

def _detect_log_type(self, file_path: Path) -> str:
    """
    Detect whether the Zeek log is JSON or TSV.
    """

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:

            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                return "tsv"

            if stripped.startswith("{"):
                return "json"

            break

    raise ValueError(
        f"Unable to determine Zeek log format: {file_path}"
    )

def _read_json(
    self,
    file_path: Path,
) -> pl.DataFrame:
    """
    Read Zeek JSON lines.
    """

    return pl.read_ndjson(file_path)

def _read_tsv(
    self,
    file_path: Path,
) -> pl.DataFrame:
    """
    Read standard Zeek TSV logs.

    Zeek stores field names in:

    #fields<TAB>field1<TAB>field2...
    """

    fields = None
    data_lines = []

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:

            if line.startswith("#fields"):
                fields = (
                    line.rstrip("\n")
                    .split("\t")[1:]
                )

            elif line.startswith("#"):
                continue

            else:
                data_lines.append(
                    line.rstrip("\n")
                )

    if fields is None:
        raise ValueError(
            "Zeek TSV log does not contain a #fields header."
        )

    if not data_lines:
        return pl.DataFrame(
            schema={
                field: pl.String
                for field in fields
            }
        )

    rows = [
        line.split("\t")
        for line in data_lines
    ]

    return pl.DataFrame(
        rows,
        schema=fields,
        orient="row",
    )

def _normalize(
    self,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Normalize Zeek fields.

    Primarily maps conn.log fields:

    ts          -> timestamp
    id.orig_h   -> src_ip
    id.resp_h   -> dst_ip
    id.orig_p   -> src_port
    id.resp_p   -> dst_port
    proto       -> protocol
    orig_pkts   -> packet count
    orig_bytes  -> byte count
    duration    -> duration
    """

    expressions = [
        self._column_or_null(
            df,
            "ts",
            "timestamp",
        ),

        self._column_or_null(
            df,
            "id.orig_h",
            "src_ip",
        ),

        self._column_or_null(
            df,
            "id.resp_h",
            "dst_ip",
        ),

        self._column_or_null(
            df,
            "id.orig_p",
            "src_port",
        ),

        self._column_or_null(
            df,
            "id.resp_p",
            "dst_port",
        ),

        self._column_or_null(
            df,
            "proto",
            "protocol",
        ),

        self._sum_columns(
            df,
            ["orig_pkts", "resp_pkts"],
            "packet_count",
        ),

        self._sum_columns(
            df,
            ["orig_bytes", "resp_bytes"],
            "byte_count",
        ),

        self._column_or_null(
            df,
            "duration",
            "duration",
        ),

        pl.lit(None).alias("tcp_flags"),

        pl.lit(None).alias("ttl"),

        pl.lit(None).alias("tcp_window_size"),

        pl.lit("zeek").alias("source_type"),
    ]

    normalized = df.select(expressions)

    return normalized.with_columns(
        [
            self._parse_zeek_timestamp(),

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
        ]
    ).select(CANONICAL_COLUMNS)

@staticmethod
def _column_or_null(
    df: pl.DataFrame,
    source_column: str,
    target_column: str,
) -> pl.Expr:
    """
    Return a column if present, otherwise a null column.
    """

    if source_column in df.columns:
        return pl.col(source_column).alias(
            target_column
        )

    return pl.lit(None).alias(target_column)

@staticmethod
def _sum_columns(
    df: pl.DataFrame,
    columns: list[str],
    target_column: str,
) -> pl.Expr:
    """
    Sum available Zeek columns.

    Example:

    orig_pkts + resp_pkts
    """

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available:
        return pl.lit(None).alias(
            target_column
        )

    expression = (
        pl.col(available[0])
        .cast(pl.Float64, strict=False)
    )

    for column in available[1:]:

        expression = expression + (
            pl.col(column)
            .cast(pl.Float64, strict=False)
        )

    return expression.alias(target_column)

@staticmethod
def _parse_zeek_timestamp() -> pl.Expr:
    return (
        pl.col("timestamp")
        .cast(pl.Float64, strict=False)
        .mul(1_000_000)
        .cast(pl.Int64, strict=False)
        .cast(
            pl.Datetime(
                "us",
                time_zone="UTC",
            )
        )
        .alias("timestamp")
    )
