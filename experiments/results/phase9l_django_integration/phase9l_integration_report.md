# Phase 9L: Django Dashboard Integration of the Authoritative Inference Engine

Status: **GREEN**

## 1. Executive Summary

Phase 9L replaced the Django dashboard's fallback-only forecasting path with a new, additive integration of the Phase 9K authoritative inference engine. A new thin service class (AuthoritativeForecastService) wraps NetOracleInferenceEngine as a process-level singleton; a new view, template, and DRF API endpoint expose current state, whole-horizon attack probability, 6-step future rollout, 6-step MITRE stage trajectory, gradient-based feature attribution, and full provenance -- with no fabricated per-step probabilities and no SHAP claim. The pre-existing heuristic path was left unchanged and is now clearly marked non-authoritative in the UI (Option B). Django-vs-direct-engine predictions matched exactly on all checked samples; the Django smoke test (system check, URL/view resolution, template rendering, no forbidden UI claims) passed; performance overhead from the Django service layer is negligible relative to model inference.

## 2. Prior Django Path (traced, Step 1)

`dashboard/views.py:get_forecast_context()` called `world_model.inference_service.AttackForecastService`, which looked for `models/attack_forecaster.pt` (absent) and always fell back to a 2-layer sigmoid-of-mean heuristic (`model_source: "heuristic_fallback"`), fed by `run_pipeline.py`'s live-capture `FlowFeatureEngine`/`TemporalFeatureEngine` pipeline -- a schema incompatible with the authoritative model's 157-feature CICFlowMeter-window schema.

## 3. Integration Boundary (Step 2)

`Django views/API -> world_model/inference_service.py: AuthoritativeForecastService -> experiments/inference_engine.py: NetOracleInferenceEngine -> checkpoint`. `AuthoritativeForecastService` contains no model logic: it locates, lazily loads (process-level singleton, Step 3), and calls the real engine, and never mutates the checkpoint or refits the scaler. The pre-existing `AttackForecastService` / `AttackRiskForecaster` classes are UNCHANGED (Option B: left as an undocumented-as-integrated legacy path, see Section 9).

## 4. Model Lifecycle / Singleton (Step 3)

`AuthoritativeForecastService._engine` is a process-level singleton, populated on first use. Cold load (checkpoint + scaler + Phase 3.5 windows dir): ~19s (one-time per process). Warm calls after that: ~3.1 ms (direct engine) / ~3.1 ms (via Django service).

## 5. Input Contract (Step 4)

Strict: `numpy.ndarray` shape `(6, 157)`, real (unscaled) feature units, exact `world_model_dataset.py` column order; rejects wrong shape/type and non-finite values via `NetOracleInferenceEngine.validate_feature_vector()` -- never silently padded, reordered, or dropped. Because the live-capture pipeline cannot currently produce this schema, the dashboard demonstrates the engine on real, already-validated Phase 3.5 TEST-split samples via `AuthoritativeForecastService.predict_demo_sample()`.

## 6. Dashboard UI Fields (Step 5)

`templates/dashboard/authoritative_forecast.html` (new page, `/dashboard/forecast/authoritative/`) shows: current state (window t, feature count), 6-step future state rollout (raw units, linked to the API for the full 157-dim vectors), whole-horizon attack probability with its semantics string, 6-step MITRE stage trajectory labeled 'Predicted attack-stage trajectory' with Phase 7B limitations text, top gradient-attribution features labeled 'Gradient-based feature attribution' (never SHAP), and full provenance (model id, checkpoint/scaler/feature-schema hashes, code commit, device, inference duration). A dedicated Limitations panel lists every explicitly-unsupported capability. The legacy `forecast.html` page and `index.html` were both updated with a banner/link distinguishing them from this authoritative page, and the misleading `best_world_model.pth` (Track B) reference in `index.html` was replaced.

## 7. API Contract

`GET /api/authoritative-predict/?index=<n>` (DRF `APIView`, `IsAuthenticated`) returns the engine's own result dict verbatim (already JSON-serializable: plain floats/lists/strings, no tensors) -- confirmed by `json.dumps()` round-trip in both the test suite and this script. Errors (missing checkpoint, invalid `index`) return explicit 4xx/5xx JSON, never a silently substituted heuristic.

## 8. Explainability Labeling

UI and API text says 'Gradient-based feature attribution' (matching `explanations.future_attack_risk_prediction.explanation_method` / `explanations.mitre_stage_prediction.explanation_method`, reused unmodified from Phase 7C's `explain_sample()`). 'SHAP' never appears as a claim of what method was used.

## 9. Legacy Path Disposition (Alerts + Heuristic Fallback)

Per this phase's explicit Option B allowance: `AttackForecastService`, `AttackRiskForecaster`, `run_pipeline.py`, and `feature_engine/services.py` are left completely UNCHANGED and are NOT documented as integrated with the authoritative engine -- they remain a separate, clearly legacy, non-authoritative path (`templates/dashboard/forecast.html` now says so explicitly). `alerts/services.py` (`IDSEngine`, `check_alert_rules`) was inspected and found to be a rule-based engine over ingested log text, entirely independent of any world-model risk score -- it never consumed `AttackForecastService`'s heuristic output and therefore required no migration or legacy marking of its own. No new threshold was invented or tuned anywhere.

## 10. Track B Non-Use

`world_model/world_model.py` (`NetworkWorldModel`) is not imported by `world_model/inference_service.py`, `dashboard/views.py`, or `api/views.py` -- verified both by direct source inspection and by an AST-based test (see test suite, area 3). It remains on disk, unmodified, unused as a prediction source, per Phase 9K's reconciliation.

## 11. Step 8: Deterministic Django-vs-Direct Equivalence

- Samples checked: 5
- All within tolerance (0.001): **True**
- All stage trajectories exact match: **True**
- All future rollouts close match: **True**
- All attributions close match: **True**
- All Django results JSON-serializable: **True**
- **Overall: True**

| # | source_file | window_start | direct_prob | django_prob | diff | stages_match |
|---|---|---|---:|---:|---:|---|
| 0 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:12:50 | 0.999852 | 0.999852 | 0.00e+00 | True |
| 1 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:00 | 0.999850 | 0.999850 | 0.00e+00 | True |
| 2 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:10 | 0.999847 | 0.999847 | 0.00e+00 | True |
| 3 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:20 | 0.999869 | 0.999869 | 0.00e+00 | True |
| 4 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:30 | 0.999870 | 0.999870 | 0.00e+00 | True |

## 12. Step 9: Django Smoke Test

All steps passed: **True**

| step | ok | detail |
|---|---|---|
| django_system_check | True |  |
| url_resolution | True | {'dashboard': '/dashboard/forecast/authoritative/', 'api': '/api/authoritative-predict/'} |
| dashboard_view_renders | True | {'status_code': 200, 'template_names': ['dashboard/authoritative_forecast.html']} |
| dashboard_view_no_forbidden_claims | True | {'offenders': []} |
| api_view_responds_200 | True | {'status_code': 200} |
| api_view_json_serializable | True |  |
| index_view_renders | True | {'status_code': 200} |
| legacy_forecast_view_renders | True | {'status_code': 200} |

## 13. Step 10: Performance

- Device: cuda:0
- Direct engine inference latency: 3.05 ms
- Django service latency: 3.06 ms
- Django service overhead: 0.005 ms
- Full HTTP request latency (dashboard page): 47.86 ms
- Full HTTP request latency (API endpoint): 52.50 ms

## 14. Error Handling (Step 6)

`get_authoritative_forecast_context()` and `AuthoritativeForecastAPIView.get()` catch `FileNotFoundError` (checkpoint missing), `ValueError`/`TypeError` (invalid input), and any other exception, surfacing a clear error message and `authoritative_available: False` / an explicit HTTP error status -- never silently substituting the legacy heuristic.

## 15. Tests (Step 7)

See the final response for exact pass counts across tests/test_phase9l_django_integration.py and the required prior-phase regressions.

## 16. Frozen-Artifact Integrity

Prior result directories unchanged: **True**

## 17. Unsupported-Claims Audit

- SHAP (actual method is gradient-based feature attribution with SHAP-compatible infrastructure)
- Native per-step attack probability (only whole-horizon P(attack in t+1..t+6))
- Calibrated probabilities (Phase 8B finding stands unchanged)
- Causal attacker kill-chain inference (MITRE stages are a reasoned label mapping, Phase 7B limitations preserved)
- Unseen-attack generalization claims
- PCAP-derived model input (Phase 9H/9I RED verdicts stand unchanged)
- Uncertainty quantification of any kind
- Counterfactual simulation of any kind

## 18. Remaining Limitations / Recommended Phase 9M

- The dashboard demonstrates the authoritative engine on a fixed, already-validated Phase 3.5 TEST-split sample, not live-captured network data -- the live-capture pipeline's feature schema remains incompatible with the model (unresolved since Phase 9H/9I; reconciling it is a separate, non-trivial future phase, not attempted here).
- The legacy heuristic path (AttackForecastService / run_pipeline.py) is unchanged and still reachable at /dashboard/forecast/ -- it is now clearly labeled non-authoritative in the UI but was not removed or migrated (Option B, explicitly permitted).
- Track B (world_model/world_model.py) remains on disk, unmodified, unused -- full retirement is still out of scope.
- Cold-start latency (~19s, checkpoint+scaler+windows-dir load) occurs on first use per Django process; not addressed here since Step 10 explicitly asked for measurement, not optimization.

A future phase could reconcile the live-capture feature schema with the authoritative model's 157-feature CICFlowMeter-window schema (a non-trivial feature-engineering project, not integration work), or retire Track B / the legacy heuristic path entirely once a human maintainer confirms no other dependency exists. Neither is in scope for Phase 9L.

STOP AFTER PHASE 9L. Do not begin 9M.
