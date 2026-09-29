# Phase 6A Dynamic Graph Design

Phase 6A represents the existing CIC-IDS2018 flow data as source-day-scoped, directed communication graphs while preserving Phase 3.5 as the canonical temporal index.

## Canonical constraints

- Phase 3.5 `data/windows/*.csv` defines every valid window key, ordering, split, eligibility flag, and sample history.
- Graph construction must not independently create, remove, or reindex windows.
- Raw CIC files provide entity and relationship fields absent from Phase 3.5 state CSVs.
- Window duration remains 10 seconds; each eligible sample maps to six graph snapshots `(t-5 ... t)`.
- The target remains `future_attack_within_horizon`, based only on t+1 through t+6.
- Phase 3.5, Phase 4, Phase 5, raw data, and production artifacts remain read-only.

## Graph model

Nodes are host entities identified by canonical `Src IP` or `Dst IP`, with IDs scoped to one source-day file. Edges are directed `Src IP -> Dst IP` relationships aggregated by source-day, canonical window key, and endpoint pair. Ports and protocols are edge attributes, not node identities.

Node features include inbound/outbound/total flow counts, packet and byte totals, unique peers, ports and protocols, and snapshot-local activity statistics. Edge features include flow count, unique ports/protocols, and finite sums, means, minima, maxima, and finite counts for available numeric CICFlowMeter fields. These preserve flow-level and packet-derived information such as packet counts, packet lengths, rates, TCP flags, header lengths, inter-arrival times, subflow counts, and active/idle statistics.

Labels, current attack metadata, future labels, timestamps as model features, raw flow identifiers, and target-derived fields are excluded from graph features.

## Cleaning and missing values

The builder follows Phase 3.5 row handling: malformed rows, repeated headers, invalid timestamps, and known malformed year-1970 rows are excluded. Invalid numeric values become missing; finite observations alone contribute to aggregates; empty aggregates become finite zero values. Every numeric graph output must be finite.

## Window and sample construction

The builder first reads each Phase 3.5 partition and records its exact ordered `window_start` keys. Raw flows are assigned to a key only if that key exists in the canonical partition. Missing or extra raw keys are fatal errors. Empty canonical windows produce empty graph snapshots, preserving the exact index.

For each eligible Phase 3.5 row at index t, the sample index stores graph windows t-5 through t, source file, split, current window, and the existing target. Histories cannot cross source-day or split boundaries. No future graph snapshot is included.

## Unseen entities

Entity IDs are deterministic within each source day and assigned on first valid chronological occurrence. Nodes absent from a window are not emitted. A later online consumer may use an explicit `UNSEEN_HOST` representation, but Phase 6A does not learn a cross-partition entity vocabulary.

## Storage

The output is written to `data/phase6a_graphs/` as compressed CSV/JSONL files:

```text
metadata.json
schema.json
entity_mapping/<source_day>.csv.gz
snapshots/<source_day>/snapshots.jsonl.gz
snapshots/<source_day>/nodes.csv.gz
snapshots/<source_day>/edges.csv.gz
samples/train.csv.gz
samples/validation.csv.gz
samples/test.csv.gz
audit/graph_validation_summary.json
```

Raw identifiers are retained only in source-day-scoped entity mappings and snapshot node records, not duplicated into model sample records.

## Validation contract

The validator checks exact snapshot/window/source/split alignment, graph counts, edge endpoint integrity, finite numeric values, flow reconciliation, empty-window representation, split counts, six-history mapping, source-day boundaries, and leakage conditions. Any failure exits nonzero and reports the failure; it never modifies Phase 3.5 data.

## Implementation files

- `NetOracle/scripts/build_phase6a_graph_dataset.py`
- `NetOracle/scripts/validate_phase6a_graph_dataset.py`
- `NetOracle/docs/PHASE6A_DYNAMIC_GRAPH_DESIGN.md`