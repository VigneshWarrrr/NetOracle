"""Build Phase 6A graph snapshots against the canonical Phase 3.5 index."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from build_temporal_dataset import FILES, KNOWN_1970_ROWS, parse_timestamp, window_start


WINDOW_SECONDS = 10
HISTORY_WINDOWS = 6
FORECAST_HORIZON_WINDOWS = 6
META_COLUMNS = {
    "window_start", "source_file", "split", "current_attack", "current_attack_types",
    "future_attack_within_horizon", "future_attack_types", "history_available",
    "history_window_count", "forecast_sample_eligible",
}
IDENTIFIER_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}


def key_for_column(column: str) -> str:
    return column.lower().replace(" ", "_").replace("/", "_per_")


def finite_number(raw: str) -> float | None:
    try:
        value = float(raw.strip())
    except (AttributeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_canonical(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def empty_stat() -> dict[str, object]:
    return {"count": 0, "sum": 0.0, "minimum": None, "maximum": None}


def add_stat(stat: dict[str, object], value: float | None) -> None:
    if value is None:
        return
    stat["count"] = int(stat["count"]) + 1
    stat["sum"] = float(stat["sum"]) + value
    stat["minimum"] = value if stat["minimum"] is None else min(float(stat["minimum"]), value)
    stat["maximum"] = value if stat["maximum"] is None else max(float(stat["maximum"]), value)


def finalize_stats(prefix: str, stats: dict[str, object], output: dict[str, object]) -> None:
    count = int(stats["count"])
    output[f"{prefix}__sum"] = float(stats["sum"])
    output[f"{prefix}__mean"] = float(stats["sum"]) / count if count else 0.0
    output[f"{prefix}__min"] = float(stats["minimum"]) if count else 0.0
    output[f"{prefix}__max"] = float(stats["maximum"]) if count else 0.0
    output[f"{prefix}__finite_count"] = count


def build_partition(raw_path: Path, canonical_path: Path, output_dir: Path) -> dict[str, object]:
    canonical_columns, canonical_rows = read_canonical(canonical_path)
    canonical_by_window = {row["window_start"]: (index, row) for index, row in enumerate(canonical_rows)}
    if len(canonical_by_window) != len(canonical_rows):
        raise ValueError(f"Duplicate canonical window keys in {canonical_path.name}")
    source_columns = [column for column in canonical_columns if column not in META_COLUMNS and column != "flow_count"]
    source_bases = sorted({column.rsplit("__", 1)[0] for column in source_columns if column.endswith(("__sum", "__mean"))})
    raw_by_key = {key_for_column(column): column for column in source_bases}
    edge_data: dict[str, dict[tuple[str, str], dict[str, object]]] = defaultdict(dict)
    entity_order: dict[str, int] = {}
    valid_flow_counts: defaultdict[str, int] = defaultdict(int)
    excluded = 0
    with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        header = [value.strip() for value in next(reader)]
        positions = {column: index for index, column in enumerate(header)}
        required = {"Timestamp", "Label", "Src IP", "Dst IP"}
        if not required.issubset(positions):
            raise ValueError(f"{raw_path.name} lacks required graph columns")
        numeric_columns = [column for column in header if column not in IDENTIFIER_COLUMNS and column != "Src Port"]
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header) or row[positions["Label"]].strip() == "Label":
                excluded += 1
                continue
            timestamp = parse_timestamp(row[positions["Timestamp"]])
            if timestamp is None or (raw_path.name, row_number) in KNOWN_1970_ROWS:
                excluded += 1
                continue
            window = window_start(timestamp).isoformat(sep=" ")
            if window not in canonical_by_window:
                raise ValueError(f"Raw window {window} is not in canonical Phase 3.5 index for {raw_path.name}")
            source = row[positions["Src IP"]].strip()
            target = row[positions["Dst IP"]].strip()
            if not source or not target:
                excluded += 1
                continue
            for entity in (source, target):
                if entity not in entity_order:
                    entity_order[entity] = len(entity_order)
            valid_flow_counts[window] += 1
            edge = edge_data[window].setdefault((source, target), {
                "flow_count": 0, "src_ports": set(), "dst_ports": set(), "protocols": set(),
                "stats": {key_for_column(column): empty_stat() for column in numeric_columns},
            })
            edge["flow_count"] = int(edge["flow_count"]) + 1
            if "Src Port" in positions:
                edge["src_ports"].add(row[positions["Src Port"]].strip())
            if "Dst Port" in positions:
                edge["dst_ports"].add(row[positions["Dst Port"]].strip())
            if "Protocol" in positions:
                edge["protocols"].add(row[positions["Protocol"]].strip())
            for column in numeric_columns:
                add_stat(edge["stats"][key_for_column(column)], finite_number(row[positions[column]]))

    mapping_rows = [{"node_id": node_id, "entity_key": entity, "node_type": "host", "source_file": raw_path.name} for entity, node_id in sorted(entity_order.items(), key=lambda item: item[1])]
    write_gzip_csv(output_dir / "entity_mapping" / f"{raw_path.stem}.csv.gz", ["node_id", "entity_key", "node_type", "source_file"], mapping_rows)

    snapshot_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for window_index, canonical in enumerate(canonical_rows):
        window = canonical["window_start"]
        edges = edge_data.get(window, {})
        node_stats: dict[str, dict[str, object]] = defaultdict(lambda: {"in_flow": 0, "out_flow": 0, "in_bytes": 0.0, "out_bytes": 0.0, "in_packets": 0.0, "out_packets": 0.0, "in_peers": set(), "out_peers": set(), "ports": set(), "protocols": set()})
        for (source, target), edge in edges.items():
            edge_output: dict[str, object] = {
                "source_file": raw_path.name, "window_index": window_index, "window_start": window,
                "source_node_id": entity_order[source], "target_node_id": entity_order[target],
                "flow_count": int(edge["flow_count"]), "unique_source_ports": len(edge["src_ports"]),
                "unique_destination_ports": len(edge["dst_ports"]), "unique_protocols": len(edge["protocols"]),
            }
            for key, stat in edge["stats"].items():
                finalize_stats(key, stat, edge_output)
            edge_rows.append(edge_output)
            source_stats = node_stats[source]
            target_stats = node_stats[target]
            flow_count = int(edge["flow_count"])
            source_stats["out_flow"] += flow_count
            target_stats["in_flow"] += flow_count
            source_stats["out_peers"].add(target)
            target_stats["in_peers"].add(source)
            source_stats["ports"].update(edge["src_ports"])
            target_stats["ports"].update(edge["dst_ports"])
            source_stats["protocols"].update(edge["protocols"])
            target_stats["protocols"].update(edge["protocols"])
            byte_stat = edge["stats"].get("totlen_fwd_pkts") or edge["stats"].get("flow_byts_per_s")
            packet_stat = edge["stats"].get("tot_fwd_pkts")
            if byte_stat:
                source_stats["out_bytes"] += float(byte_stat["sum"])
                target_stats["in_bytes"] += float(byte_stat["sum"])
            if packet_stat:
                source_stats["out_packets"] += float(packet_stat["sum"])
                target_stats["in_packets"] += float(packet_stat["sum"])
        active_nodes = sorted(node_stats, key=lambda entity: entity_order[entity])
        for entity in active_nodes:
            stats = node_stats[entity]
            node_rows.append({
                "source_file": raw_path.name, "window_index": window_index, "window_start": window,
                "node_id": entity_order[entity], "entity_key": entity, "node_type": "host",
                "in_flow_count": stats["in_flow"], "out_flow_count": stats["out_flow"], "total_flow_count": stats["in_flow"] + stats["out_flow"],
                "in_packet_count": stats["in_packets"], "out_packet_count": stats["out_packets"], "in_byte_count": stats["in_bytes"], "out_byte_count": stats["out_bytes"],
                "unique_in_peers": len(stats["in_peers"]), "unique_out_peers": len(stats["out_peers"]), "unique_ports": len(stats["ports"]), "unique_protocols": len(stats["protocols"]),
            })
        snapshot_rows.append({"source_file": raw_path.name, "window_index": window_index, "window_start": window, "split": canonical["split"], "node_count": len(active_nodes), "edge_count": len(edges), "flow_count": valid_flow_counts.get(window, 0), "forecast_sample_eligible": canonical["forecast_sample_eligible"]})

    stem_dir = output_dir / "snapshots" / raw_path.stem
    write_jsonl_gz(stem_dir / "snapshots.jsonl.gz", snapshot_rows)
    write_gzip_csv(stem_dir / "nodes.csv.gz", list(node_rows[0]) if node_rows else ["source_file", "window_index", "window_start", "node_id", "entity_key", "node_type"], node_rows)
    write_gzip_csv(stem_dir / "edges.csv.gz", list(edge_rows[0]) if edge_rows else ["source_file", "window_index", "window_start", "source_node_id", "target_node_id", "flow_count"], edge_rows)
    return {"source_file": raw_path.name, "window_count": len(snapshot_rows), "node_count": len(node_rows), "edge_count": len(edge_rows), "flow_count": sum(valid_flow_counts.values()), "excluded_rows": excluded, "empty_windows": sum(row["edge_count"] == 0 for row in snapshot_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/phase6a_graphs")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 6A output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    all_samples: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    partition_summaries = []
    canonical_schema: list[str] | None = None
    for filename in FILES:
        canonical_path = args.windows_dir / filename
        raw_path = args.raw_dir / filename
        if not canonical_path.exists() or not raw_path.exists():
            raise FileNotFoundError(f"Missing canonical/raw pair for {filename}")
        columns, rows = read_canonical(canonical_path)
        canonical_schema = canonical_schema or columns
        if columns != canonical_schema:
            raise ValueError("Phase 3.5 partition schemas differ")
        for index, row in enumerate(rows):
            if row["forecast_sample_eligible"] == "1":
                history_indices = list(range(index - HISTORY_WINDOWS + 1, index + 1))
                all_samples[row["split"]].append({"sample_id": f"{filename}:{index}", "source_file": filename, "split": row["split"], "current_window_index": index, "history_window_indices": ";".join(map(str, history_indices)), "history_window_starts": ";".join(rows[position]["window_start"] for position in history_indices), "target": int(row["future_attack_within_horizon"])})
        partition_summaries.append(build_partition(raw_path, canonical_path, args.output_dir))
    for split, rows in all_samples.items():
        write_gzip_csv(args.output_dir / "samples" / f"{split}.csv.gz", list(rows[0]) if rows else ["sample_id", "source_file", "split", "current_window_index", "history_window_indices", "history_window_starts", "target"], rows)
    metadata = {"window_seconds": WINDOW_SECONDS, "history_windows": HISTORY_WINDOWS, "forecast_horizon_windows": FORECAST_HORIZON_WINDOWS, "canonical_index": "data/windows Phase 3.5 partition rows", "partitions": partition_summaries, "sample_counts": {split: len(rows) for split, rows in all_samples.items()}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "schema.json").write_text(json.dumps({"node_identity": "source-day-scoped IP", "edge_identity": "source-day/window/src_ip/dst_ip", "node_features": "snapshot-local traffic and degree features", "edge_features": "finite source numeric sums/means/min/max plus counts", "labels_excluded": True}, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()