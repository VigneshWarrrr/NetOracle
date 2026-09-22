# Temporal Dataset Design

This artifact describes the read-only Phase 3 dataset build. Raw CSV files remain unchanged, source days remain separate, and no model training or model-weight generation occurs here. Each source-day partition is written as a compact CSV because the selected environment does not provide a Parquet writer; no new dependency is added.

## 1. State representation

Each source file is reconstructed independently in timestamp order. Every 10-second state contains `flow_count`, current-window attack metadata, and sum/mean aggregates for the common numeric CICFlowMeter columns. The resulting state vector has 157 numeric features. The final feature policy is `<source-column>__sum` and `<source-column>__mean` for every common numeric CICFlowMeter column, excluding `Label`, raw `Timestamp`, `Flow ID`, `Src IP`, `Dst IP`, and Tuesday-only `Src Port`.

## 2. Cleaning policy

Invalid numeric strings and `+inf`/`-inf` are represented as missing during aggregation. Invalid values do not remove an otherwise valid flow. Aggregates use only finite observations; an empty aggregate is represented as 0.0 as a structural no-observation value, not as learned imputation. Affected-value counts are recorded in the summary. No training statistics are learned in this phase.

## 3. Timestamp policy

Repeated header rows, the 14 known year-1970 rows, other unparseable timestamps, and malformed-width rows are excluded from modeling and logged in `data/audit/temporal_preprocessing_log.csv`. Valid flows are externally sorted within each source file. Raw timestamps are not model inputs.

## 4. Label policy

`Infilteration` is normalized to `Infiltration`; known capitalization variants of DDoS labels are normalized. Raw labels remain available in the preprocessing audit. `Benign` maps to current attack indicator 0. Any other known non-header label maps to 1. Repeated `Label` rows are excluded and never become attack classes.

## 5. Window size

`WINDOW_SECONDS = 10`. Windows are anchored to ten-second boundaries and are created independently per source day, including empty traffic windows between the valid minimum and maximum timestamps.

## 6. History length

`HISTORY_WINDOWS = 6`. `history_available` marks rows with enough preceding states for a later sequence consumer; no sequences are trained or materialized here.

## 7. Forecast horizon

`FORECAST_HORIZON_WINDOWS = 6`, representing the next 60 seconds.

## 8. Future-label definition

For state S(t), `future_attack_within_horizon` is 1 when any current attack indicator in S(t+1) through S(t+6) is 1. S(t) is explicitly excluded. Future attack types are retained where available. The final 60 terminal windows with fewer than six future states are retained for state continuity but marked `forecast_sample_eligible = 0` and excluded from forecasting samples.

## 9. Split strategy

Each source-day partition is split chronologically by window position: first 70% train, next 15% validation, final 15% test. A forecasting sample is eligible only when its six history states, current state, and six future states all belong to the same split. This removes split guard windows and prevents histories or targets from crossing boundaries. No individual flows are randomly split and no source-day sequence crosses a partition boundary. Totals: 41705 valid samples; train 29315, validation 6195, test 6195.

## 10. Leakage prevention

`Label`, raw `Timestamp`, `Flow ID`, `Src IP`, and `Dst IP` are excluded from numeric state features. Current attack metadata is retained only for audit/evaluation and is not part of the feature vector. Future labels use only strictly later windows. Tuesday-only identifiers are not required by the common feature schema.

## 11. Final feature list

The exact generated feature columns are the common numeric source columns with `__sum` and `__mean` suffixes, plus `flow_count`. The state metadata columns are `current_attack`, `current_attack_types`, `future_attack_within_horizon`, `future_attack_types`, `history_available`, `history_window_count`, `forecast_sample_eligible`, `source_file`, `window_start`, and `split`; metadata is not a model input.

## 12. Known limitations

- The source CSVs are flow-level CICFlowMeter data, not packet-level SIH 26153 data.
- The source files are heavily out of order and are reconstructed with external chunk sorting.
- Empty-window zero values are structural aggregation values, not imputation.
- The final six windows of each source partition have fewer than six future observations and should be filtered or treated explicitly by a later training consumer.
- Samples near chronological split boundaries are intentionally excluded when their history or future span would cross a split.
- Imputation/scaling parameters must be fit on eligible training samples only, then applied unchanged to validation and test.
- No learned imputation, scaling, threshold tuning, model training, or temporal-window model integration is performed.

## Output totals

- Total temporal windows: 42035
- Valid forecasting samples: 41705
- Incomplete future-horizon windows removed from forecasting samples: 60
- Future-positive eligible samples: 7926
- Future-negative eligible samples: 33779

## Phase 3.5 validation

- Valid forecasting samples: 41,705.
- State windows retained for partition continuity: 42,035.
- Incomplete six-step future horizons excluded: 60 terminal windows.
- Additional split-boundary guard samples excluded: 270; their histories or futures would cross train/validation/test boundaries.
- Final eligible future labels: 7,926 positive and 33,779 negative.
- Eligible split counts: train 29,315, validation 6,195, test 6,195.
- Leakage status: PASS. Every eligible sample has six history states, six future states, one source-day partition, and one split; current state t is excluded from its target.
- Duplicate status: 13,287 duplicate eligible states and 12,211 duplicate eligible six-state sequences were found by global hash. They are retained for review and were not automatically deleted.
- Feature-quality status: 16 constant features were found; no feature exceeded the extreme-value threshold of absolute value 1e12. Constant features are retained pending modeling review.
- Future preprocessing policy: fit imputation and scaling parameters using eligible training samples only, then apply those parameters unchanged to validation and test. Do not learn statistics from validation or test.

READY FOR BASELINE TRAINING
