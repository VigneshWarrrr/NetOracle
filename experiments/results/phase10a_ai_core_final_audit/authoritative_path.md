# Phase 10A: Authoritative AI-core execution path

Audit date 2026-09-20. Every step below was checked against the code and, where marked **[verified]**, exercised by the Phase 10A evidence script. Nothing here was modified.

## 0. Artifacts (frozen)

| Role | Path | SHA-256 (first 16) |
|---|---|---|
| Authoritative checkpoint (Run-1 backbone + trained stage head) | `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt` | `f9d16f1943aeeed4` |
| Authoritative scaler (train-only StandardScaler, 157 features) | `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib` | `b3a0cea7f9d2d0fc` |
| Backbone provenance (Phase 6B "Run-1", NOT the plain 6B directory) | `.../phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt` | `6374a9c47d722215` |
| Feature schema | derived at engine start from `../data/windows/*.csv` headers (157 names) | schema sha `f5117fda8518c9e3` |

## 1. Request surface (Django)

1. `GET /dashboard/forecast/authoritative/` -> `dashboard/views.py:AuthoritativeForecastView` (l.115; extends the staff-only `AdminDashboardView`) -> `get_authoritative_forecast_context()` (l.17) -> `AuthoritativeForecastService.predict_demo_sample(index=0)`.
2. `GET /api/authoritative-predict/?index=<n>` -> `api/views.py:AuthoritativeForecastAPIView.get` (l.38; `IsAuthenticated`) -> `AuthoritativeForecastService.predict_demo_sample(index=n)`. Missing checkpoint -> HTTP 503, bad/out-of-range index -> HTTP 400; there is **no heuristic fallback** on either surface **[verified]**.
3. Routes: `dashboard/urls.py:9`, `api/urls.py:13`.

## 2. Service layer

`world_model/inference_service.py:AuthoritativeForecastService` (l.119-200)
- `get_engine()` (l.145): class-level singleton; inserts `experiments/` on `sys.path`, imports `inference_engine`, constructs `NetOracleInferenceEngine()` once per process (no lock; first call ~19 s **[verified]**).
- `predict_demo_sample(index)` (l.186): takes a real sample from the frozen Phase 3.5 **test** split (`engine.get_test_sample`) and calls `predict()`.
- `predict(x_raw, ...)` (l.159): `np.asarray(float32)` -> `engine.predict(...)`. No model logic lives here.

## 3. Engine construction (`experiments/inference_engine.py`, `NetOracleInferenceEngine.__init__`, l.157-204)

1. l.158-161: hard-fails (`FileNotFoundError`) if the Run-1 scaler or the Phase 7B checkpoint is missing.
2. l.163: device = cuda:0 if available else CPU.
3. l.167-172: `world_model_dataset.read_world_model_samples(windows_dir)` reads all ten CSVs -> `feature_columns` (must be exactly 157) and the test samples used for the demo. **This is the only source of feature names/order.**
4. l.178: `joblib.load(scaler)` -- never refit **[verified: equals a fresh train-only refit, rel diff 0e+00]**.
5. l.183-186: `phase7c_explainability.load_trained_phase7b_model()` loads the full checkpoint `strict=True` into `VectorWorldModelWithStageHead`; `.eval()`; `requires_grad_(False)` on all parameters (356452 params = 347806 backbone + 8646 stage head).
6. l.189-204: `EngineProvenance` (model id, checkpoint/scaler/schema sha256, `git rev-parse HEAD`, semantics strings).

## 4. `predict()` (l.226-305)

| Step | Code | What happens |
|---|---|---|
| 1 | l.238 -> `validate_feature_vector` (l.210) | must be `numpy.ndarray`, shape `(6,157)`, all finite; otherwise `TypeError`/`ValueError` (never padded/reordered/dropped) |
| 2 | l.241 `scale_array` (phase6b l.202) | frozen scaler `transform` on `reshape(-1,157)` |
| 3 | l.245-249 `self.model(x)` under `no_grad` | `VectorWorldModelWithStageHead.forward` (phase7b l.104): `state_encoder` -> `temporal_context` (2-layer self-attention over t-5..t) -> last context vector -> `transition` applied 6x (residual) = `z_future [B,6,128]` -> `state_decoder` = `y_hat [B,6,157]`; `attack_head(mean(z_future))` = ONE attack logit; `stage_head(z_future)` = `stage_logits [B,6,6]` |
| 4 | l.247 | `attack_probability = sigmoid(attack_logit)` -- **single whole-horizon scalar** |
| 5 | l.248, 253-255 | `softmax(stage_logits)` -> per-step argmax stage name + softmax confidence (`CLASS_INDEX_TO_STAGE`: BENIGN, INITIAL_ACCESS, CREDENTIAL_ACCESS, LATERAL_MOVEMENT, COMMAND_AND_CONTROL, IMPACT) |
| 6 | l.251 | rollout in raw units = `y_hat * scaler.scale_ + scaler.mean_` |
| 7 | l.262-265 -> `phase7c_explainability.explain_sample` (l.185) | separate forward/backward passes; `TemporalAwareExplainer.explain_attack_risk` (`explainability/shap_explainer.py` l.110) -> tries SHAP, falls back to **Gradient x Input** (`_explain_with_gradients` l.372; gradient of the sigmoid probability); stage attribution via `StageLogitView`. Reports `explanation_method` per call **[verified: "Gradient x Input" in 300/300]** |
| 8 | l.269-304 | JSON-serialisable dict: `input_metadata`, `current_state`, `future_state_rollout`, `whole_horizon_attack_probability {value, semantics}`, `mitre_stage_trajectory`, `explanations`, `provenance` |

## 5. Output contract (what each field IS)

- `whole_horizon_attack_probability.value` -- uncalibrated sigmoid score for "attack somewhere in t+1..t+6". Not per-step. (Phase 8B: raw test ECE 0.112.)
- `future_state_rollout` -- six predicted 157-dim STATE vectors. Not a risk series.
- `mitre_stage_trajectory` -- six per-step stage labels + softmax confidence from the Phase 7B stage head. Not attack risk.
- `explanations` -- Gradient x Input attributions ([6,157]) and per-window top features.
- `provenance` -- hashes/commit/device/timing.

## 6. What is NOT on the path

- Phase 9N per-step risk head: never imported/loaded (0 references) **[verified]**.
- Track B `world_model/world_model.py`, `inference.py`, Streamlit: never imported **[verified]**.
- Legacy `AttackForecastService` heuristic + `run_pipeline.py`: reachable only via `/dashboard/forecast/` (live-capture events) and CLI; not used by the authoritative page/API.
- Live capture, PCAP, Zeek: no route into the engine (schema incompatible).

## 7. Verified properties of this path

- Engine output = frozen checkpoint: reproduces all 6195 frozen Phase 7B test predictions to 5e-13 under the frozen (mixed-precision) protocol; the shipped fp32 path differs by <= 2.4e-03 (6/37170 stage cells flip).
- Deterministic (bit-exact repeats and fresh instances); CPU vs GPU <= 2.2e-04, identical stage trajectories.
- Concurrent calls (4 threads) safe once loaded.
