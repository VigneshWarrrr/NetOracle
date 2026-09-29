# Phase 9K: World Model Track Reconciliation

## Decision

**AUTHORITATIVE**: `experiments/phase6b_vector_world_model.py` (VectorWorldModel backbone)
+ `experiments/phase7b_mitre_stage_head.py` (VectorWorldModelWithStageHead)
+ checkpoint `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt`
(the frozen Phase 6B "Run1" backbone -- `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling`
-- plus the trained MitreStageHead, byte-verified unchanged from Run1 by Phase 7B's own training protocol)
+ scaler `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib`.

This is now exposed through exactly ONE module: `experiments/inference_engine.py`
(`NetOracleInferenceEngine`).

**NON-AUTHORITATIVE / LEGACY**: `world_model/world_model.py` (`NetworkWorldModel`),
its feature pipeline `world_model/feature_extraction.py`, its Streamlit dashboard
`world_model/src/streamlit.py`, and its evaluation artifacts
`world_model/evaluation_results.json` / `world_model/zero_shot_results.json`.

## Why

Phase 9J traced every reference to Track B and found:

- No validated checkpoint exists anywhere in the repository for `NetworkWorldModel`
  (`models/best_world_model.pth` absent, re-confirmed by this phase's safety manifest).
- `world_model/evaluation_results.json` (the source of the Streamlit "Benchmark" tab's
  hardcoded numbers) used a non-standard, undocumented split
  (`Wednesday-28-INFIL-TRAIN.csv` / `-INFIL-TEST.csv`, absent from `data/windows/`
  and `data/raw/`) and shows a validation/test metric inversion (zero validation
  recall at every horizon, 79-85% test recall) consistent with leakage or an
  invalid evaluation methodology.
- The Django dashboard's own model loader (`world_model/inference_service.py`)
  looks for a THIRD, also-absent checkpoint (`models/attack_forecaster.pt`) and
  falls back to a 2-layer heuristic -- it does not currently call either Track A
  or Track B.

## Disposition

Per Phase 9K's explicit instruction ("do not spend time trying to improve Track
B", "prefer a compatibility/deprecation boundary if existing UI code depends on
it", "do not delete files blindly"):

- Track B's files are **left in place, unmodified** -- no deletion.
- Track B is **quarantined by documentation only** in this phase: this file, plus
  a NON-AUTHORITATIVE marker referenced from `experiments/inference_engine.py`'s
  own module docstring, are the deprecation boundary.
- No UI code in this repository was found to import a Track B module through any
  Django-reachable path (Phase 9J's repo-wide trace: Track B is only imported by
  `world_model/src/streamlit.py`, `world_model/build_project.py`, and
  `experiments/tests.py`'s own guard tests -- none of which are Django URL-routed).
  Therefore no compatibility shim was required; Track B was already effectively
  isolated from the production Django path.
- `world_model/evaluation_results.json` and `world_model/zero_shot_results.json`
  remain on disk (frozen, Phase 9J already analyzed them) but this phase's
  authoritative inference engine and reports **never read, import, or cite either
  file** as evidence, metrics, or predictions.

## What would fully retire Track B (not done in this phase)

Deleting `world_model/world_model.py`, `world_model/feature_extraction.py`,
`world_model/src/streamlit.py`, `world_model/evaluation_results.json`, and
`world_model/zero_shot_results.json` is a reasonable FUTURE cleanup once a human
maintainer confirms no other undiscovered dependency exists -- explicitly out of
scope for Phase 9K's audit-then-integrate discipline (no blind deletion).
