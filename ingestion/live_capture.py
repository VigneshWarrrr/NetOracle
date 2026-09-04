"""Capture live IP packets from the host network interface with Scapy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scapy.layers.inet import IP, TCP, UDP


def packet_to_event(packet: Any) -> dict | None:
    if IP not in packet:
        return None

    ip_layer = packet[IP]
    event = {
        "timestamp": datetime.fromtimestamp(float(packet.time), tz=timezone.utc).isoformat(),
        "src_ip": ip_layer.src,
        "dst_ip": ip_layer.dst,
        "src_port": None,
        "dst_port": None,
        "protocol": str(ip_layer.proto),
        "packet_count": 1,
        "byte_count": len(packet),
        "duration": 0.0,
        "tcp_flags": None,
        "ttl": int(ip_layer.ttl),
        "tcp_window_size": None,
        "source_type": "live_interface",
    }
    if TCP in packet:
        event.update(
            protocol="TCP",
            src_port=int(packet[TCP].sport),
            dst_port=int(packet[TCP].dport),
            tcp_flags=str(packet[TCP].flags),
            tcp_window_size=int(packet[TCP].window),
        )
    elif UDP in packet:
        event.update(
            protocol="UDP",
            src_port=int(packet[UDP].sport),
            dst_port=int(packet[UDP].dport),
        )
    return event


def capture_forever(interface: str | None, batch_size: int, on_batch) -> None:
    from scapy.all import sniff

    batch: list[dict] = []

    def handle_packet(packet: Any) -> None:
        event = packet_to_event(packet)
        if event is None:
            return
        batch.append(event)
        if len(batch) >= batch_size:
            on_batch(batch.copy())
            batch.clear()

    sniff(iface=interface or None, prn=handle_packet, store=False)
