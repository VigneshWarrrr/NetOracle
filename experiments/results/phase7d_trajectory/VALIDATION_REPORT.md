# Phase 7D Trajectory Generation -- Validation Report

Samples generated: 5
Duration: 0.759s on cuda
All samples passed cross-check against Phase 7B persisted predictions: True
Checkpoint state unchanged during run: True
Tracked frozen files unchanged: True
Deterministic regeneration of sample 1 matches: True

## Per-sample results

| # | source_file | window_start | label | prob diff | stages match | overall match | disagreement |
|---|---|---|---:|---:|---|---|---|
| 1 | Friday-02-03-2018_TrafficForML | 2018-03-02 11:12:50 | 1 | 3.58e-07 | True | True | False |
| 2 | Friday-02-03-2018_TrafficForML | 2018-03-02 11:13:00 | 1 | 4.77e-07 | True | True | False |
| 3 | Friday-16-02-2018_TrafficForML | 2018-02-16 11:11:40 | 0 | 6.19e-07 | True | True | False |
| 4 | Friday-16-02-2018_TrafficForML | 2018-02-16 11:11:50 | 0 | 6.19e-07 | True | True | False |
| 5 | Friday-16-02-2018_TrafficForML | 2018-02-16 11:12:00 | 0 | 6.19e-07 | True | True | False |

## Example narrative (sample 1)

Native horizon attack probability (the model's own single risk estimate for the complete 6-step / 60-second forecast horizon): 0.9999. Overall predicted MITRE stage, derived by combining the six independent per-step assessments: COMMAND_AND_CONTROL (confidence 0.9994). All six independent future-stage assessments (t+1 through t+6) predict COMMAND_AND_CONTROL. The MITRE-stage-derived non-benign signal (NOT attack probability, NOT attack risk, NOT infiltration probability) is roughly stable across the horizon (step 1: 0.9998, step 6: 1.0000). The feature contributing most (per gradient-based attribution on the frozen model) to step 6's predicted stage (COMMAND_AND_CONTROL) was 'dst_port__mean'. This describes model attribution only, not a real-world or causal explanation. These six future-stage assessments are independent per-step model outputs, not a guaranteed or causally-linked kill-chain progression.

