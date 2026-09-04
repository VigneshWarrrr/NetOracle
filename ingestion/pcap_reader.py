from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Union

import polars as pl
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

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

class PcapReaderService:

 def __init__(
    self,
    include_non_ip: bool = False,
    batch_size: int = 100_000,
) -> None:
    self.include_non_ip = include_non_ip
    self.batch_size = batch_size

def read(
    self,
    file_path: Union[str, Path],
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Read packets from a PCAP file.

    Parameters
    ----------
    file_path:
        Path to the PCAP/PCAPNG file.

    limit:
        Optional maximum number of packets to process.

    Returns
    -------
    pl.DataFrame
        Canonical network event DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    rows = []
    processed = 0

    with PcapReader(str(file_path)) as pcap:
        for packet in pcap:
            if limit is not None and processed >= limit:
                break

            event = self._packet_to_event(packet)

            if event is None:
                if self.include_non_ip:
                    event = self._empty_event(packet)

                else:
                    continue

            rows.append(event)
            processed += 1

    return self._to_dataframe(rows)

def iter_events(
    self,
    file_path: Union[str, Path],
) -> Iterator[dict]:
    """
    Stream normalized packet events.

    Useful for large PCAP files where loading everything
    into memory is undesirable.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    with PcapReader(str(file_path)) as pcap:
        for packet in pcap:
            event = self._packet_to_event(packet)

            if event is not None:
                yield event

def _packet_to_event(self, packet) -> Optional[dict]:
    """
    Convert one Scapy packet into the canonical event schema.
    """

    if IP not in packet:
        return None

    ip_layer = packet[IP]

    timestamp = datetime.fromtimestamp(
        float(packet.time),
        tz=timezone.utc,
    )

    protocol_number = int(ip_layer.proto)

    src_port = None
    dst_port = None
    tcp_flags = None
    tcp_window_size = None

    if TCP in packet:
        tcp_layer = packet[TCP]

        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        tcp_flags = str(tcp_layer.flags)
        tcp_window_size = int(tcp_layer.window)

        protocol = "TCP"

    elif UDP in packet:
        udp_layer = packet[UDP]

        src_port = int(udp_layer.sport)
        dst_port = int(udp_layer.dport)

        protocol = "UDP"

    else:
        protocol = str(protocol_number)

    return {
        "timestamp": timestamp,
        "src_ip": ip_layer.src,
        "dst_ip": ip_layer.dst,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "packet_count": 1,
        "byte_count": len(packet),
        "duration": 0.0,
        "tcp_flags": tcp_flags,
        "ttl": int(ip_layer.ttl),
        "tcp_window_size": tcp_window_size,
        "source_type": "pcap",
    }

def _empty_event(self, packet) -> dict:
    """
    Create a canonical event for non-IP packets.
    """

    timestamp = datetime.fromtimestamp(
        float(packet.time),
        tz=timezone.utc,
    )

    return {
        "timestamp": timestamp,
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "protocol": None,
        "packet_count": 1,
        "byte_count": len(packet),
        "duration": 0.0,
        "tcp_flags": None,
        "ttl": None,
        "tcp_window_size": None,
        "source_type": "pcap",
    }

@staticmethod
def _to_dataframe(rows: list[dict]) -> pl.DataFrame:
    """
    Convert rows into a strongly typed Polars DataFrame.
    """

    schema = {
        "timestamp": pl.Datetime("us", time_zone="UTC"),
        "src_ip": pl.String,
        "dst_ip": pl.String,
        "src_port": pl.Int64,
        "dst_port": pl.Int64,
        "protocol": pl.String,
        "packet_count": pl.Int64,
        "byte_count": pl.Int64,
        "duration": pl.Float64,
        "tcp_flags": pl.String,
        "ttl": pl.Int64,
        "tcp_window_size": pl.Int64,
        "source_type": pl.String,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    for column, dtype in schema.items():
        if column not in df.columns:
            df = df.with_columns(
                pl.lit(None).cast(dtype).alias(column)
            )

    return (
        df
        .select(CANONICAL_COLUMNS)
        .cast(schema, strict=False)
        .sort("timestamp")
    )
