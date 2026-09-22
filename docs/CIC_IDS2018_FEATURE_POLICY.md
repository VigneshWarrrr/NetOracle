# CIC-IDS2018 Feature Policy

This policy is an audit artifact only. It does not delete, transform, impute, or rename any raw column.

## Classification

### SAFE_TRAFFIC_FEATURE

The CICFlowMeter-derived traffic measurements are candidate traffic features: protocol, destination port, packet counts, byte totals, flow duration, packet lengths, rates, inter-arrival statistics, TCP flags, headers, window values, subflow values, active/idle statistics, and related counters. They remain subject to numeric-quality checks and domain validation.

### POTENTIAL_IDENTIFIER

- `Flow ID`: flow-level identifier; may allow memorization and should not be used as an ordinary numeric feature.
- `Src IP`: host or network identifier; may encode environment, role, or split membership.
- `Dst IP`: host or network identifier; may encode environment, role, or split membership.

These fields are present only in `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`. The other nine files do not contain them. Any later use requires an explicit generalization, grouping, and train/test split policy.

### POTENTIAL_LEAKAGE

- `Label`: supervised target. It must not be supplied as an input feature.
- `Timestamp`: can leak collection order, attack-period boundaries, or file identity if used directly. It requires a time-aware validation policy and explicit justification for any derived feature.

### NEEDS_INVESTIGATION

- `Src Port` and `Dst Port`: ports are traffic attributes but can act as strong attack or service proxies. Review their semantics, cardinality, and split behavior before use.
- `Timestamp`: determine whether it is used only for ordering/splitting or also transformed into model inputs.
- `Flow ID`, `Src IP`, and `Dst IP`: assess identifier memorization and cross-file generalization.
- All numeric columns: inspect the accompanying `data/audit/numeric_feature_quality.csv` before deciding how invalid, infinite, or missing values will be handled.

## Packet-level gap

The current processed CSVs contain flow-level, CICFlowMeter-derived features. SIH 26153 also requires packet-level features. No packet-level integration is included in this phase.

## Policy before modeling

Keep raw inputs immutable, preserve file provenance, resolve the Tuesday-only schema explicitly, exclude `Label` from inputs, define identifier handling, and use file-aware and time-aware validation. Do not impute or repair values until those policies are approved.