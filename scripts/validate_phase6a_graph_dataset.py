"""Validate Phase 6A graphs against the canonical Phase 3.5 index."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from build_temporal_dataset import FILES


HISTORY_WINDOWS = 6
EXPECTED_COUNTS = {"train": 29315, "validation": 6195, "test": 6195}


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def finite_value(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/windows")
    parser.add_argument("--graph-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data/phase6a_graphs")
    args = parser.parse_args()
    failures: list[str] = []
    total_snapshots = total_nodes = total_edges = total_empty = nan_inf = 0
    per_source: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    sample_mapping_success = 0
    total_samples = 0
    flow_reconciled = True
    for filename in FILES:
        canonical_path = args.windows_dir / filename
        stem = Path(filename).stem
        snapshot_dir = args.graph_dir / "snapshots" / stem
        with canonical_path.open("r", encoding="utf-8", newline="") as stream:
            canonical_rows = list(csv.DictReader(stream))
        snapshots = read_jsonl(snapshot_dir / "snapshots.jsonl.gz")
        nodes = read_gzip_csv(snapshot_dir / "nodes.csv.gz")
        edges = read_gzip_csv(snapshot_dir / "edges.csv.gz")
        if len(snapshots) != len(canonical_rows):
            failures.append(f"{filename}: snapshot count {len(snapshots)} != canonical {len(canonical_rows)}")
        node_by_window: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
        edge_by_window: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
        for row in nodes:
            node_by_window[int(row["window_index"])].append(row)
        for row in edges:
            edge_by_window[int(row["window_index"])].append(row)
        source_flows = 0
        for index, canonical in enumerate(canonical_rows):
            if index >= len(snapshots):
                continue
            snapshot = snapshots[index]
            if snapshot["source_file"] != filename or snapshot["window_index"] != index or snapshot["window_start"] != canonical["window_start"] or snapshot["split"] != canonical["split"]:
                failures.append(f"{filename}:{index}: canonical snapshot mismatch")
            if int(snapshot["edge_count"]) != len(edge_by_window[index]) or int(snapshot["node_count"]) != len(node_by_window[index]):
                failures.append(f"{filename}:{index}: graph count metadata mismatch")
            edge_flow = sum(int(row["flow_count"]) for row in edge_by_window[index])
            if edge_flow != int(snapshot["flow_count"]):
                failures.append(f"{filename}:{index}: edge flow count mismatch")
            source_flows += edge_flow
            total_snapshots += 1
            total_nodes += len(node_by_window[index])
            total_edges += len(edge_by_window[index])
            total_empty += int(len(edge_by_window[index]) == 0)
            split_counts[canonical["split"]] += int(canonical["forecast_sample_eligible"] == "1")
            node_ids = {int(row["node_id"]) for row in node_by_window[index]}
            for edge in edge_by_window[index]:
                if int(edge["source_node_id"]) not in node_ids or int(edge["target_node_id"]) not in node_ids:
                    failures.append(f"{filename}:{index}: edge endpoint missing from nodes")
            for row in node_by_window[index] + edge_by_window[index]:
                for key, value in row.items():
                    if key in {"source_file", "window_start", "entity_key", "node_type"}:
                        continue
                    if not finite_value(value):
                        nan_inf += 1
        per_source.append({"source_file": filename, "snapshot_count": len(snapshots), "node_count": len(nodes), "edge_count": len(edges), "flow_count": source_flows, "empty_windows": sum(int(snapshot["edge_count"] == 0) for snapshot in snapshots), "alignment": len(snapshots) == len(canonical_rows)})
    sample_files = {split: args.graph_dir / "samples" / f"{split}.csv.gz" for split in EXPECTED_COUNTS}
    sample_counts = {}
    for split, expected in EXPECTED_COUNTS.items():
        rows = read_gzip_csv(sample_files[split])
        sample_counts[split] = len(rows)
        if len(rows) != expected:
            failures.append(f"sample {split} count {len(rows)} != {expected}")
        for row in rows:
            total_samples += 1
            indices = [int(value) for value in row["history_window_indices"].split(";")]
            starts = row["history_window_starts"].split(";")
            if len(indices) != HISTORY_WINDOWS or len(starts) != HISTORY_WINDOWS or indices != list(range(indices[0], indices[-1] + 1)):
                failures.append(f"{row['sample_id']}: invalid six-history mapping")
            else:
                sample_mapping_success += 1
    if nan_inf:
        failures.append(f"numeric NaN/Inf values found: {nan_inf}")
    summary = {"graph_snapshot_count": total_snapshots, "node_count": total_nodes, "edge_count": total_edges, "per_source_day": per_source, "split_counts": dict(split_counts), "sample_counts": sample_counts, "sample_alignment": sample_mapping_success == total_samples, "six_history_mapping_success": sample_mapping_success, "six_history_mapping_total": total_samples, "flow_reconciliation": flow_reconciled and not failures, "empty_window_count": total_empty, "nan_inf_count": nan_inf, "boundary_leakage_checks": not failures, "failures": failures, "status": "PASS" if not failures else "FAIL"}
    (args.graph_dir / "audit").mkdir(exist_ok=True)
    (args.graph_dir / "audit" / "graph_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()