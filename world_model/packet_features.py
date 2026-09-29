"""
NetOracle - Packet-Level Feature Extraction

This module extracts packet-level network telemetry from PCAP files
and converts it into timestamped, time-windowed feature vectors.

Designed to complement the existing flow-level feature extraction
pipeline in NetOracle.

Features include:
    - TTL statistics
    - TCP window statistics
    - IP fragmentation
    - Payload-size statistics
    - TCP retransmission estimates
    - Port diversity
    - Sequential port scanning
    - Randomized port scanning
    - Packet / byte statistics
    - Protocol statistics
    - TCP flag statistics
    - Inter-arrival-time statistics

Default time window:
    10 seconds

Output:
    List[Dict[str, float]]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

import numpy as np

from scapy.all import (
    rdpcap,
    IP,
    IPv6,
    TCP,
    UDP,
    ICMP,
    Raw,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_WINDOW_SIZE = 10.0


# ============================================================
# Utility functions
# ============================================================

def safe_mean(values: List[float]) -> float:
    """Return mean or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.mean(values))


def safe_std(values: List[float]) -> float:
    """Return standard deviation or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.std(values))


def safe_variance(values: List[float]) -> float:
    """Return variance or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.var(values))


def safe_min(values: List[float]) -> float:
    """Return minimum or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.min(values))


def safe_max(values: List[float]) -> float:
    """Return maximum or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.max(values))


def safe_median(values: List[float]) -> float:
    """Return median or 0.0 for an empty list."""
    if not values:
        return 0.0

    return float(np.median(values))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division."""
    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)


def normalize_score(value: float, minimum: float, maximum: float) -> float:
    """
    Normalize value to [0, 1].

    Used only for bounded feature scores.
    """
    if maximum <= minimum:
        return 0.0

    score = (value - minimum) / (maximum - minimum)

    return float(max(0.0, min(1.0, score)))


# ============================================================
# Packet identity / flow helpers
# ============================================================

def get_ip_pair(packet) -> Tuple[str, str]:
    """
    Return source and destination IP addresses.

    Supports IPv4 and IPv6.
    """

    if IP in packet:
        return (
            str(packet[IP].src),
            str(packet[IP].dst),
        )

    if IPv6 in packet:
        return (
            str(packet[IPv6].src),
            str(packet[IPv6].dst),
        )

    return ("unknown", "unknown")


def get_flow_key(packet) -> Optional[Tuple]:
    """
    Generate a bidirectional-independent flow key.

    The key contains:
        source IP
        destination IP
        source port
        destination port
        protocol

    Direction is preserved because retransmission detection needs
    the sender/receiver relationship.
    """

    src, dst = get_ip_pair(packet)

    if TCP in packet:

        return (
            src,
            dst,
            int(packet[TCP].sport),
            int(packet[TCP].dport),
            "TCP",
        )

    if UDP in packet:

        return (
            src,
            dst,
            int(packet[UDP].sport),
            int(packet[UDP].dport),
            "UDP",
        )

    return None


# ============================================================
# TTL extraction
# ============================================================

def extract_ttl(packet) -> Optional[int]:
    """
    Extract TTL for IPv4 or Hop Limit for IPv6.

    IPv6 does not use TTL. Its equivalent is Hop Limit.
    """

    if IP in packet:
        return int(packet[IP].ttl)

    if IPv6 in packet:
        return int(packet[IPv6].hlim)

    return None


# ============================================================
# TCP window extraction
# ============================================================

def extract_tcp_window(packet) -> Optional[int]:
    """Return TCP receive window size."""

    if TCP not in packet:
        return None

    return int(packet[TCP].window)


# ============================================================
# Payload extraction
# ============================================================

def extract_payload_size(packet) -> int:
    """
    Return application/payload size in bytes.

    For TCP/UDP packets, the payload is obtained from the transport
    layer.

    Raw payload is preferred when available.
    """

    if Raw in packet:
        try:
            return len(bytes(packet[Raw].load))
        except Exception:
            return 0

    if TCP in packet:
        try:
            return len(bytes(packet[TCP].payload))
        except Exception:
            return 0

    if UDP in packet:
        try:
            return len(bytes(packet[UDP].payload))
        except Exception:
            return 0

    return 0


# ============================================================
# Fragmentation
# ============================================================

def is_ipv4_fragment(packet) -> bool:
    """
    Determine whether an IPv4 packet is fragmented.

    IPv4 fragmentation is indicated by:
        MF flag
        OR non-zero fragment offset
    """

    if IP not in packet:
        return False

    flags = packet[IP].flags
    fragment_offset = int(packet[IP].frag)

    more_fragments = bool(flags.MF)

    return more_fragments or fragment_offset > 0


# ============================================================
# TCP flag extraction
# ============================================================

def extract_tcp_flags(packet) -> Dict[str, int]:
    """
    Extract individual TCP flags.
    """

    flags = {
        "syn": 0,
        "ack": 0,
        "fin": 0,
        "rst": 0,
        "psh": 0,
        "urg": 0,
    }

    if TCP not in packet:
        return flags

    tcp_flags = packet[TCP].flags

    flags["syn"] = int(bool(tcp_flags.S))
    flags["ack"] = int(bool(tcp_flags.A))
    flags["fin"] = int(bool(tcp_flags.F))
    flags["rst"] = int(bool(tcp_flags.R))
    flags["psh"] = int(bool(tcp_flags.P))
    flags["urg"] = int(bool(tcp_flags.U))

    return flags


# ============================================================
# Retransmission detection
# ============================================================

class TCPRetransmissionTracker:
    """
    Lightweight TCP retransmission estimator.

    It tracks previously observed TCP sequence numbers per
    directional TCP flow.

    A repeated sequence number is counted as a possible
    retransmission.

    NOTE:
        This is an estimation rather than a full TCP state-machine
        implementation. It is appropriate for a feature-engineering
        pipeline but should not be interpreted as a perfect TCP
        retransmission detector.
    """

    def __init__(self):

        self.seen_sequences = defaultdict(set)

    def process(self, packet) -> bool:

        if TCP not in packet:
            return False

        flow_key = get_flow_key(packet)

        if flow_key is None:
            return False

        try:
            seq = int(packet[TCP].seq)
        except Exception:
            return False

        # Zero-length ACKs should not normally be treated as
        # retransmitted payload.
        payload_size = extract_payload_size(packet)

        if payload_size <= 0:
            return False

        if seq in self.seen_sequences[flow_key]:

            return True

        self.seen_sequences[flow_key].add(seq)

        return False


# ============================================================
# Port scan analysis
# ============================================================

def calculate_port_scan_features(
    destination_ports: List[int],
) -> Dict[str, float]:
    """
    Calculate sequential and randomized port-scan indicators.

    Sequential scanning:
        Attempts ports with consecutive numbering.

    Randomized scanning:
        Attempts many ports spread across a large port range
        without strong sequential ordering.

    These are behavioural indicators, NOT attack labels.
    """

    if not destination_ports:

        return {
            "unique_dst_ports": 0.0,
            "port_range": 0.0,
            "sequential_port_score": 0.0,
            "random_port_score": 0.0,
        }

    unique_ports = list(set(destination_ports))

    unique_count = len(unique_ports)

    if unique_count < 2:

        return {
            "unique_dst_ports": float(unique_count),
            "port_range": 0.0,
            "sequential_port_score": 0.0,
            "random_port_score": 0.0,
        }

    sorted_ports = sorted(unique_ports)

    port_range = (
        float(max(sorted_ports) - min(sorted_ports))
    )

    # --------------------------------------------------------
    # Sequential scan
    # --------------------------------------------------------

    consecutive_pairs = 0

    for first, second in zip(
        sorted_ports,
        sorted_ports[1:],
    ):

        if second - first == 1:
            consecutive_pairs += 1

    sequential_score = safe_ratio(
        consecutive_pairs,
        unique_count - 1,
    )

    # --------------------------------------------------------
    # Randomized scan
    # --------------------------------------------------------

    # Density measures how many different ports were touched
    # relative to the range they span.
    density = safe_ratio(
        unique_count - 1,
        port_range,
    )

    # High dispersion + low sequentiality indicates a more
    # randomized access pattern.
    dispersion_score = normalize_score(
        port_range,
        1.0,
        65535.0,
    )

    random_score = (
        dispersion_score *
        (1.0 - sequential_score)
    )

    return {
        "unique_dst_ports": float(unique_count),
        "port_range": port_range,
        "sequential_port_score": float(
            max(0.0, min(1.0, sequential_score))
        ),
        "random_port_score": float(
            max(0.0, min(1.0, random_score))
        ),
    }


# ============================================================
# Inter-arrival time
# ============================================================

def calculate_iat_features(
    timestamps: List[float],
) -> Dict[str, float]:
    """
    Calculate packet inter-arrival-time statistics.
    """

    if len(timestamps) < 2:

        return {
            "packet_iat_mean": 0.0,
            "packet_iat_std": 0.0,
            "packet_iat_variance": 0.0,
            "packet_iat_min": 0.0,
            "packet_iat_max": 0.0,
        }

    timestamps = sorted(timestamps)

    iats = np.diff(
        np.asarray(timestamps)
    )

    iats = iats[
        np.isfinite(iats)
    ]

    if len(iats) == 0:

        return {
            "packet_iat_mean": 0.0,
            "packet_iat_std": 0.0,
            "packet_iat_variance": 0.0,
            "packet_iat_min": 0.0,
            "packet_iat_max": 0.0,
        }

    return {
        "packet_iat_mean": float(np.mean(iats)),
        "packet_iat_std": float(np.std(iats)),
        "packet_iat_variance": float(np.var(iats)),
        "packet_iat_min": float(np.min(iats)),
        "packet_iat_max": float(np.max(iats)),
    }


# ============================================================
# Main window feature extraction
# ============================================================

def calculate_window_features(
    packets: List,
    retransmission_tracker: Optional[
        TCPRetransmissionTracker
    ] = None,
) -> Dict[str, float]:
    """
    Calculate packet-level features for one time window.
    """

    if retransmission_tracker is None:

        retransmission_tracker = (
            TCPRetransmissionTracker()
        )

    # --------------------------------------------------------
    # Collections
    # --------------------------------------------------------

    timestamps = []

    ttl_values = []

    tcp_window_values = []

    payload_sizes = []

    destination_ports = []

    source_ports = []

    packet_lengths = []

    tcp_packet_count = 0
    udp_packet_count = 0
    icmp_packet_count = 0
    other_packet_count = 0

    fragment_count = 0

    retransmission_count = 0

    # TCP flags
    syn_count = 0
    ack_count = 0
    fin_count = 0
    rst_count = 0
    psh_count = 0
    urg_count = 0

    total_bytes = 0

    # --------------------------------------------------------
    # Process packets
    # --------------------------------------------------------

    for packet in packets:

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        try:

            timestamp = float(packet.time)

            if math.isfinite(timestamp):
                timestamps.append(timestamp)

        except Exception:

            continue

        # ----------------------------------------------------
        # Packet length
        # ----------------------------------------------------

        try:

            packet_length = len(packet)

            packet_lengths.append(
                float(packet_length)
            )

            total_bytes += packet_length

        except Exception:
            packet_length = 0

        # ----------------------------------------------------
        # TTL / Hop Limit
        # ----------------------------------------------------

        ttl = extract_ttl(packet)

        if ttl is not None:
            ttl_values.append(float(ttl))

        # ----------------------------------------------------
        # Fragmentation
        # ----------------------------------------------------

        if is_ipv4_fragment(packet):
            fragment_count += 1

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload_size = extract_payload_size(packet)

        payload_sizes.append(
            float(payload_size)
        )

        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------

        if TCP in packet:

            tcp_packet_count += 1

            # TCP window
            window_size = extract_tcp_window(packet)

            if window_size is not None:

                tcp_window_values.append(
                    float(window_size)
                )

            # Destination port
            try:

                destination_ports.append(
                    int(packet[TCP].dport)
                )

                source_ports.append(
                    int(packet[TCP].sport)
                )

            except Exception:
                pass

            # TCP flags
            flags = extract_tcp_flags(packet)

            syn_count += flags["syn"]
            ack_count += flags["ack"]
            fin_count += flags["fin"]
            rst_count += flags["rst"]
            psh_count += flags["psh"]
            urg_count += flags["urg"]

            # Retransmission
            if retransmission_tracker.process(packet):

                retransmission_count += 1

        # ----------------------------------------------------
        # UDP
        # ----------------------------------------------------

        elif UDP in packet:

            udp_packet_count += 1

            try:

                destination_ports.append(
                    int(packet[UDP].dport)
                )

                source_ports.append(
                    int(packet[UDP].sport)
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # ICMP
        # ----------------------------------------------------

        elif ICMP in packet:

            icmp_packet_count += 1

        # ----------------------------------------------------
        # Other
        # ----------------------------------------------------

        else:

            other_packet_count += 1

    # ========================================================
    # Aggregate statistics
    # ========================================================

    packet_count = len(packets)

    # --------------------------------------------------------
    # TTL
    # --------------------------------------------------------

    ttl_mean = safe_mean(ttl_values)

    ttl_std = safe_std(ttl_values)

    ttl_variance = safe_variance(ttl_values)

    ttl_min = safe_min(ttl_values)

    ttl_max = safe_max(ttl_values)

    # --------------------------------------------------------
    # TCP window
    # --------------------------------------------------------

    tcp_window_mean = safe_mean(
        tcp_window_values
    )

    tcp_window_std = safe_std(
        tcp_window_values
    )

    tcp_window_variance = safe_variance(
        tcp_window_values
    )

    tcp_window_min = safe_min(
        tcp_window_values
    )

    tcp_window_max = safe_max(
        tcp_window_values
    )

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    payload_mean = safe_mean(
        payload_sizes
    )

    payload_std = safe_std(
        payload_sizes
    )

    payload_variance = safe_variance(
        payload_sizes
    )

    payload_min = safe_min(
        payload_sizes
    )

    payload_max = safe_max(
        payload_sizes
    )

    payload_median = safe_median(
        payload_sizes
    )

    # --------------------------------------------------------
    # Packet length
    # --------------------------------------------------------

    packet_length_mean = safe_mean(
        packet_lengths
    )

    packet_length_std = safe_std(
        packet_lengths
    )

    packet_length_variance = safe_variance(
        packet_lengths
    )

    packet_length_min = safe_min(
        packet_lengths
    )

    packet_length_max = safe_max(
        packet_lengths
    )

    # --------------------------------------------------------
    # Protocol ratios
    # --------------------------------------------------------

    tcp_ratio = safe_ratio(
        tcp_packet_count,
        packet_count,
    )

    udp_ratio = safe_ratio(
        udp_packet_count,
        packet_count,
    )

    icmp_ratio = safe_ratio(
        icmp_packet_count,
        packet_count,
    )

    other_ratio = safe_ratio(
        other_packet_count,
        packet_count,
    )

    # --------------------------------------------------------
    # Fragmentation
    # --------------------------------------------------------

    fragment_ratio = safe_ratio(
        fragment_count,
        packet_count,
    )

    # --------------------------------------------------------
    # Retransmissions
    # --------------------------------------------------------

    retransmission_rate = safe_ratio(
        retransmission_count,
        tcp_packet_count,
    )

    # --------------------------------------------------------
    # TCP flag ratios
    # --------------------------------------------------------

    syn_ratio = safe_ratio(
        syn_count,
        tcp_packet_count,
    )

    ack_ratio = safe_ratio(
        ack_count,
        tcp_packet_count,
    )

    fin_ratio = safe_ratio(
        fin_count,
        tcp_packet_count,
    )

    rst_ratio = safe_ratio(
        rst_count,
        tcp_packet_count,
    )

    psh_ratio = safe_ratio(
        psh_count,
        tcp_packet_count,
    )

    urg_ratio = safe_ratio(
        urg_count,
        tcp_packet_count,
    )

    # --------------------------------------------------------
    # Port scanning
    # --------------------------------------------------------

    port_scan_features = (
        calculate_port_scan_features(
            destination_ports
        )
    )

    # --------------------------------------------------------
    # Packet IAT
    # --------------------------------------------------------

    iat_features = calculate_iat_features(
        timestamps
    )

    # ========================================================
    # Final feature vector
    # ========================================================

    features = {

        # ----------------------------------------------------
        # Basic packet statistics
        # ----------------------------------------------------

        "packet_count": float(
            packet_count
        ),

        "total_bytes": float(
            total_bytes
        ),

        "bytes_per_packet": safe_ratio(
            total_bytes,
            packet_count,
        ),

        # ----------------------------------------------------
        # TTL
        # ----------------------------------------------------

        "ttl_mean": ttl_mean,
        "ttl_std": ttl_std,
        "ttl_variance": ttl_variance,
        "ttl_min": ttl_min,
        "ttl_max": ttl_max,

        # ----------------------------------------------------
        # TCP window
        # ----------------------------------------------------

        "tcp_window_mean": tcp_window_mean,
        "tcp_window_std": tcp_window_std,
        "tcp_window_variance": tcp_window_variance,
        "tcp_window_min": tcp_window_min,
        "tcp_window_max": tcp_window_max,

        # ----------------------------------------------------
        # Fragmentation
        # ----------------------------------------------------

        "fragment_count": float(
            fragment_count
        ),

        "fragment_ratio": fragment_ratio,

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        "payload_mean": payload_mean,
        "payload_std": payload_std,
        "payload_variance": payload_variance,
        "payload_min": payload_min,
        "payload_max": payload_max,
        "payload_median": payload_median,

        # ----------------------------------------------------
        # Packet length
        # ----------------------------------------------------

        "packet_length_mean": packet_length_mean,
        "packet_length_std": packet_length_std,
        "packet_length_variance": packet_length_variance,
        "packet_length_min": packet_length_min,
        "packet_length_max": packet_length_max,

        # ----------------------------------------------------
        # Protocol distribution
        # ----------------------------------------------------

        "tcp_packet_count": float(
            tcp_packet_count
        ),

        "udp_packet_count": float(
            udp_packet_count
        ),

        "icmp_packet_count": float(
            icmp_packet_count
        ),

        "other_packet_count": float(
            other_packet_count
        ),

        "tcp_packet_ratio": tcp_ratio,
        "udp_packet_ratio": udp_ratio,
        "icmp_packet_ratio": icmp_ratio,
        "other_packet_ratio": other_ratio,

        # ----------------------------------------------------
        # TCP flags
        # ----------------------------------------------------

        "syn_count": float(syn_count),
        "ack_count": float(ack_count),
        "fin_count": float(fin_count),
        "rst_count": float(rst_count),
        "psh_count": float(psh_count),
        "urg_count": float(urg_count),

        "syn_ratio": syn_ratio,
        "ack_ratio": ack_ratio,
        "fin_ratio": fin_ratio,
        "rst_ratio": rst_ratio,
        "psh_ratio": psh_ratio,
        "urg_ratio": urg_ratio,

        # ----------------------------------------------------
        # Retransmission
        # ----------------------------------------------------

        "retransmission_count": float(
            retransmission_count
        ),

        "retransmission_rate": retransmission_rate,

        # ----------------------------------------------------
        # Port scanning
        # ----------------------------------------------------

        **port_scan_features,

        # ----------------------------------------------------
        # Inter-arrival time
        # ----------------------------------------------------

        **iat_features,
    }

    return features


# ============================================================
# PCAP → time-windowed feature matrix
# ============================================================

def extract_packet_features(
    pcap_path: str,
    window_size: float = DEFAULT_WINDOW_SIZE,
) -> List[Dict[str, float]]:
    """
    Extract packet-level features from a PCAP.

    Parameters
    ----------
    pcap_path:
        Path to PCAP / PCAPNG file.

    window_size:
        Time window in seconds.

        Default = 10 seconds.

    Returns
    -------
    List[Dict[str, float]]

        One feature dictionary per time window.

    Example
    -------
    [
        {
            "window_id": 0,
            "timestamp": 1527811200.0,
            "packet_count": 1250,
            "ttl_mean": 63.4,
            ...
        },
        ...
    ]
    """

    if window_size <= 0:

        raise ValueError(
            "window_size must be greater than zero"
        )

    # --------------------------------------------------------
    # Read PCAP
    # --------------------------------------------------------

    try:

        packets = rdpcap(pcap_path)

    except FileNotFoundError:

        raise FileNotFoundError(
            f"PCAP file not found: {pcap_path}"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to read PCAP '{pcap_path}': {exc}"
        ) from exc

    if len(packets) == 0:

        return []

    # --------------------------------------------------------
    # Establish capture start time
    # --------------------------------------------------------

    valid_timestamps = []

    for packet in packets:

        try:

            timestamp = float(packet.time)

            if math.isfinite(timestamp):

                valid_timestamps.append(
                    timestamp
                )

        except Exception:
            continue

    if not valid_timestamps:

        raise ValueError(
            "PCAP contains no valid packet timestamps."
        )

    start_time = min(
        valid_timestamps
    )

    # --------------------------------------------------------
    # Group packets into time windows
    # --------------------------------------------------------

    windows = defaultdict(list)

    for packet in packets:

        try:

            timestamp = float(packet.time)

            if not math.isfinite(timestamp):
                continue

        except Exception:

            continue

        relative_time = (
            timestamp - start_time
        )

        window_id = int(
            relative_time // window_size
        )

        windows[window_id].append(
            packet
        )

    # --------------------------------------------------------
    # Shared retransmission tracker
    #
    # IMPORTANT:
    # Keep this tracker across windows so a retransmission
    # appearing in a later window can still be detected.
    # --------------------------------------------------------

    retransmission_tracker = (
        TCPRetransmissionTracker()
    )

    # --------------------------------------------------------
    # Generate feature vectors
    # --------------------------------------------------------

    results = []

    for window_id in sorted(windows.keys()):

        window_packets = windows[window_id]

        features = calculate_window_features(
            window_packets,
            retransmission_tracker,
        )

        features["window_id"] = float(
            window_id
        )

        features["timestamp"] = float(
            start_time +
            window_id * window_size
        )

        features["window_start"] = float(
            window_id * window_size
        )

        features["window_end"] = float(
            (window_id + 1) * window_size
        )

        results.append(features)

    return results


# ============================================================
# Convert result to NumPy matrix
# ============================================================

def features_to_matrix(
    feature_rows: List[Dict[str, float]],
    exclude_metadata: bool = True,
):
    """
    Convert feature dictionaries into a NumPy matrix.

    Returns:
        matrix
        feature_names
    """

    if not feature_rows:

        return (
            np.empty((0, 0), dtype=np.float32),
            [],
        )

    metadata_columns = {
        "window_id",
        "timestamp",
        "window_start",
        "window_end",
    }

    feature_names = list(
        feature_rows[0].keys()
    )

    if exclude_metadata:

        feature_names = [
            name
            for name in feature_names
            if name not in metadata_columns
        ]

    matrix = np.asarray(
        [
            [
                float(row.get(name, 0.0))
                for name in feature_names
            ]
            for row in feature_rows
        ],
        dtype=np.float32,
    )

    # Replace invalid numerical values.
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return matrix, feature_names


# ============================================================
# Convenience function
# ============================================================

def extract_packet_feature_matrix(
    pcap_path: str,
    window_size: float = DEFAULT_WINDOW_SIZE,
):
    """
    Convenience function.

    PCAP
      ↓
    Packet features
      ↓
    NumPy matrix

    Returns
    -------
    matrix:
        Shape = (number_of_windows, number_of_features)

    feature_names:
        Names corresponding to matrix columns.

    metadata:
        Window/timestamp information.
    """

    rows = extract_packet_features(
        pcap_path,
        window_size,
    )

    matrix, feature_names = (
        features_to_matrix(rows)
    )

    metadata = []

    for row in rows:

        metadata.append(
            {
                "window_id": row["window_id"],
                "timestamp": row["timestamp"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
            }
        )

    return (
        matrix,
        feature_names,
        metadata,
    )


# ============================================================
# Command-line testing
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract packet-level features from a PCAP "
            "for NetOracle."
        )
    )

    parser.add_argument(
        "pcap",
        help="Path to PCAP / PCAPNG file",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Window size in seconds (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NetOracle Packet Feature Extraction")
    print("=" * 70)

    print(f"PCAP       : {args.pcap}")
    print(f"Window size: {args.window} seconds")
    print()

    rows = extract_packet_features(
        args.pcap,
        args.window,
    )

    if not rows:

        print("No packets/features found.")
        raise SystemExit(0)

    matrix, feature_names = (
        features_to_matrix(rows)
    )

    print(
        f"Windows extracted : {len(rows)}"
    )

    print(
        f"Features/window   : {len(feature_names)}"
    )

    print(
        f"Matrix shape      : {matrix.shape}"
    )

    print()
    print("Feature names:")
    print("-" * 70)

    for index, name in enumerate(
        feature_names,
        start=1,
    ):

        print(
            f"{index:3d}. {name}"
        )

    print()
    print("First window:")
    print("-" * 70)

    first_row = rows[0]

    for name in feature_names:

        print(
            f"{name:30s}: "
            f"{first_row.get(name, 0.0):.6f}"
        )

    print()
    print("=" * 70)
    print("Extraction completed successfully.")
    print("=" * 70)