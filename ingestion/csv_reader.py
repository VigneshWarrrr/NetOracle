from __future__ import annotations

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

CIC_FEATURE_MAPPING = {
    "flow_duration": ["Flow Duration"],
    "total_fwd_packets": ["Total Fwd Packets", "Tot Fwd Pkts"],
    "total_backward_packets": ["Total Backward Packets", "Tot Bwd Pkts"],
    "flow_bytes_per_second": ["Flow Bytes/s", "Flow Byts/s"],
    "flow_packets_per_second": ["Flow Packets/s", "Flow Pkts/s"],
    "flow_iat_mean": ["Flow IAT Mean"],
    "flow_iat_std": ["Flow IAT Std"],
    "syn_flag_count": ["SYN Flag Count", "SYN Flag Cnt"],
    "ack_flag_count": ["ACK Flag Count", "ACK Flag Cnt"],
    "rst_flag_count": ["RST Flag Count", "RST Flag Cnt"],
    "average_packet_size": ["Average Packet Size", "Pkt Size Avg"],
    "destination_port": ["Destination Port", "Dst Port"],
    "label": ["Label"],
}


class CSVReaderService:
    DEFAULT_COLUMN_MAPPING = {
        "timestamp": ["timestamp", "Timestamp", "time", "Time", "ts"],
        "src_ip": ["src_ip", "Source IP", "SourceIP", "source_ip"],
        "dst_ip": ["dst_ip", "Destination IP", "DestinationIP", "destination_ip"],
        "src_port": ["src_port", "Source Port", "SourcePort", "source_port"],
        "dst_port": ["dst_port", "Destination Port", "DestinationPort", "destination_port"],
        "protocol": ["protocol", "Protocol", "proto"],
        "packet_count": ["packet_count", "Total Packets", "Packets"],
        "byte_count": ["byte_count", "Total Bytes", "Bytes"],
        "duration": ["duration", "Flow Duration", "Duration"],
        "tcp_flags": ["tcp_flags", "TCP Flags", "Flags"],
        "ttl": ["ttl", "TTL"],
        "tcp_window_size": ["tcp_window_size", "TCP Window Size", "Window Size"],
    }

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        separator: str = ",",
    ) -> None:
        self.column_mapping = column_mapping or {}
        self.separator = separator

    def read(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """Read a CSV and adapt it without discarding dataset-specific features."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        dataframe = pl.read_csv(
            file_path,
            separator=self.separator,
            ignore_errors=True,
            infer_schema_length=10_000,
            truncate_ragged_lines=True,
        )
        return self._normalize(dataframe)

    def scan(self, file_path: Union[str, Path]) -> pl.LazyFrame:
        """Lazily scan raw CSV data; call read for the normalized adapter output."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        return pl.scan_csv(
            file_path,
            separator=self.separator,
            infer_schema_length=10_000,
            ignore_errors=True,
        )

    def _normalize(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Add canonical and stable feature names while retaining raw columns."""
        expressions = []
        for canonical_column in CANONICAL_COLUMNS:
            if canonical_column == "source_type":
                expressions.append(pl.lit("csv").alias(canonical_column))
                continue
            if canonical_column in {"packet_count", "byte_count"}:
                continue

            source_column = self._resolve_column(canonical_column, dataframe.columns)
            if source_column is None:
                expressions.append(pl.lit(None).alias(canonical_column))
            else:
                expressions.append(pl.col(source_column).alias(canonical_column))

        fwd_packets = self._resolve_aliases(["Total Fwd Packets", "Tot Fwd Pkts"], dataframe.columns)
        bwd_packets = self._resolve_aliases(["Total Backward Packets", "Tot Bwd Pkts"], dataframe.columns)
        fwd_bytes = self._resolve_aliases(["Total Length of Fwd Packets", "TotLen Fwd Pkts"], dataframe.columns)
        bwd_bytes = self._resolve_aliases(["Total Length of Bwd Packets", "TotLen Bwd Pkts"], dataframe.columns)

        if fwd_packets and bwd_packets:
            expressions.append(
                (pl.col(fwd_packets).cast(pl.Float64, strict=False).fill_null(0.0)
                 + pl.col(bwd_packets).cast(pl.Float64, strict=False).fill_null(0.0))
                .alias("packet_count")
            )
        else:
            source_column = self._resolve_column("packet_count", dataframe.columns)
            expressions.append(
                (pl.col(source_column) if source_column else pl.lit(None)).alias("packet_count")
            )
        if fwd_bytes and bwd_bytes:
            expressions.append(
                (pl.col(fwd_bytes).cast(pl.Float64, strict=False).fill_null(0.0)
                 + pl.col(bwd_bytes).cast(pl.Float64, strict=False).fill_null(0.0))
                .alias("byte_count")
            )
        else:
            source_column = self._resolve_column("byte_count", dataframe.columns)
            expressions.append(
                (pl.col(source_column) if source_column else pl.lit(None)).alias("byte_count")
            )

        for feature_name, aliases in CIC_FEATURE_MAPPING.items():
            source_column = self._resolve_aliases(aliases, dataframe.columns)
            if source_column is not None:
                expressions.append(pl.col(source_column).alias(feature_name))

        normalized = dataframe.with_columns(expressions)
        return normalized.with_columns(
            [
                pl.col("timestamp").cast(pl.String).str.to_datetime(strict=False, time_zone="UTC").alias("timestamp"),
                pl.col("src_ip").cast(pl.String, strict=False),
                pl.col("dst_ip").cast(pl.String, strict=False),
                pl.col("src_port").cast(pl.Int64, strict=False),
                pl.col("dst_port").cast(pl.Int64, strict=False),
                pl.col("protocol").cast(pl.String, strict=False),
                pl.col("packet_count").cast(pl.Float64, strict=False),
                pl.col("byte_count").cast(pl.Float64, strict=False),
                pl.col("duration").cast(pl.Float64, strict=False),
                pl.col("tcp_flags").cast(pl.String, strict=False),
                pl.col("ttl").cast(pl.Int64, strict=False),
                pl.col("tcp_window_size").cast(pl.Int64, strict=False),
                pl.col("source_type").cast(pl.String),
            ]
        )

    def _resolve_column(self, canonical_column: str, available_columns: list[str]) -> Optional[str]:
        if canonical_column in self.column_mapping:
            mapped_column = self.column_mapping[canonical_column]
            if mapped_column in available_columns:
                return mapped_column
            raise ValueError(
                f"Mapped column '{mapped_column}' for '{canonical_column}' was not found."
            )
        return self._resolve_aliases(self.DEFAULT_COLUMN_MAPPING.get(canonical_column, []), available_columns)

    @staticmethod
    def _resolve_aliases(aliases: list[str], available_columns: list[str]) -> Optional[str]:
        normalized_columns = {
            column.strip().casefold().replace("_", " "): column
            for column in available_columns
        }
        for alias in aliases:
            resolved = normalized_columns.get(alias.strip().casefold().replace("_", " "))
            if resolved is not None:
                return resolved
        return None
