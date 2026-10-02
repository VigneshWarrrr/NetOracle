# NetOracle: attack-risk forecasting from network-flow windows

NetOracle is a Django application around a PyTorch world model trained on CSE-CIC-IDS2018 flow data. Given the six most recent 10-second network-state windows (157 features each), it predicts the next six windows, one attack-risk score for that whole horizon, and a six-step MITRE-stage trajectory, with Gradient x Input feature attribution.

## SIH problem statement

- Smart India Hackathon (SIH) 2026
- Problem statement ID: SIH26153, *AI based Network Attack Forecasting from Network Traffic Data*

## Demo status and scope

The demo uses validated replay windows from the held-out test split and does not perform live network detection. Each request runs the model on one stored replay window (six consecutive 10-second windows) from the 6,195-sample test split of the derived window dataset.

Not part of the authoritative demo:

- No live network detection and no live capture.
- No PCAP input and no packet+flow fusion; the input is flow-derived window features only.
- No graph neural network in the authoritative model.
- No calibrated probability and no attack probability per future step.
- No established generalization to unseen attack families.
- No causal kill-chain inference.
- No blockchain functionality.
- No Docker configuration is provided.

## Authoritative inference path

```text
Django page / API
   -> world_model/inference_service.py::AuthoritativeForecastService   (loads the engine once per process)
   -> experiments/inference_engine.py::NetOracleInferenceEngine.predict()
   -> validation of (6, 157) finite input
   -> frozen scaler
   -> VectorWorldModelWithStageHead
   -> whole-horizon attack-risk score
   -> six-step MITRE-stage trajectory
   -> six predicted future states
   -> Gradient x Input attribution
```

The API never falls back to a heuristic: a missing artifact returns HTTP 503 and an invalid request returns HTTP 400.

## Authoritative and legacy paths

| | Authoritative (the SIH demo) | Legacy, NON-AUTHORITATIVE |
|---|---|---|
| Inference | `experiments/inference_engine.py` (`NetOracleInferenceEngine`) | `world_model/inference_service.py::AttackForecastService` (a sigmoid-of-mean heuristic) |
| Django wiring | `dashboard/views.py`, `api/views.py` -> `world_model/inference_service.py::AuthoritativeForecastService` -> `NetOracleInferenceEngine` | `/dashboard/forecast/`, `run_pipeline.py`, `train.py`, `capture_traffic` |
| Checkpoint | `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt` | `models/attack_forecaster.pt` is **not shipped**; without it the legacy path only returns the heuristic |
| Status | frozen, audited (Phases 9K, 9L, 10A) | kept for compatibility only; do not treat its output as a NetOracle model prediction |

Also not part of the authoritative path: the "Track B" model in `world_model/` (`world_model.py`, `inference.py`, `build_project.py`, `src/streamlit.py`), and the Phase 9N per-step risk experiment (its verdict is RED and it is excluded).

## Model contract

**Input:** one `numpy.ndarray` of shape `(6, 157)` in raw (unscaled) units: six consecutive 10-second windows `t-5 ... t`, 157 features each, in the exact column order of `data/windows/*.csv` (schema SHA-256 `f5117fda8518c9e373a4260ff65281800a4760d8af16bb97411ca8371b85bb73`). The engine rejects a wrong type, a wrong shape or non-finite values; nothing is padded or reordered.

**Output** (JSON, from `NetOracleInferenceEngine.predict`):

- `whole_horizon_attack_probability.value` is the **attack-risk score** for the *whole* horizon `t+1 ... t+6`. The whole-horizon attack field is an attack-risk score, not a calibrated probability (the key name is historical). There is one whole-horizon attack-risk score, not one attack-risk probability per future step. Read it as a ranking score.
- `mitre_stage_trajectory` holds the six-step stage prediction (`per_step_stage`, `per_step_confidence`). It is separate from the attack-risk score.
- `future_state_rollout` holds six predicted 157-feature network states (`predicted_state_raw_units`, `predicted_state_scaled`).
- `explanations` holds post-hoc **Gradient x Input** feature attribution; `explanations.future_attack_risk_prediction.explanation_method` reports `Gradient × Input` on every call. Attribution magnitudes are small when the score is saturated near 0 or 1 (for example 1e-6): a small number does not mean zero importance, so compare relative values.
- `current_state` and `input_metadata` describe the observed input window.
- `provenance` records the checkpoint, scaler and feature-schema hashes, the Git commit, the device and timing.

Explainability uses Gradient x Input, not SHAP. SHAP is **not** used by the authoritative path; the `shap` package listed in `requirements.txt` is not required for it.

## MITRE-stage output

The stage head predicts one of six classes for each of the six future steps, with a stage-class softmax confidence: `BENIGN`, `INITIAL_ACCESS`, `CREDENTIAL_ACCESS`, `LATERAL_MOVEMENT`, `COMMAND_AND_CONTROL`, `IMPACT`.

The stage labels come from a reasoned mapping of CSE-CIC-IDS2018 attack labels. They are not MITRE ATT&CK ground truth, not kill-chain inference and not proof of attacker behavior.

## Authoritative artifacts and SHA-256

| Role | Path | SHA-256 |
|---|---|---|
| Checkpoint (Phase 6B Run 1 backbone + trained Phase 7B stage head, 356,452 parameters) | `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt` | `f9d16f1943aeeed4f355659b85e90cb6cf8cb94ae84d2fcbd473c59fa3342fe4` |
| Scaler (train-only `StandardScaler`, 157 features) | `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib` | `b3a0cea7f9d2d0fc59de163c8768116cbd95abbe3b30c127c44a2322f0c24881` |
| Backbone reference (its 39 tensors are bit-identical to those inside the checkpoint) | `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt` | `6374a9c47d722215e64a3cae1b1e24c425a89dad26407b147b153af03404fb79` |

These three files are committed to Git.

## Dataset and replay artifacts

**The window dataset is external to Git.** The engine reads `../data/windows/*.csv` (ten derived Phase 3.5 files, about 71 MB) when it is constructed, to obtain the 157 feature names and the replay samples. The files are therefore required for the model to load at all, not only for replay. Keep this layout:

```text
<workspace>/
  NetOracle/        <- this repository
  data/windows/     <- the ten *_TrafficForML_CICFlowMeter.csv window files (not in Git)
```

Each file's SHA-256 is in `experiments/results/phase10b_submission_remediation/authoritative_artifacts.json`. There is no public download for these derived files: obtain them from the submission bundle, or rebuild them with `scripts/build_temporal_dataset.py` from the public CSE-CIC-IDS2018 CSVs (a rebuild is **not** verified to be byte-identical; compare against the recorded hashes).

Verify the checkpoint, scaler, backbone reference and all ten window files against the manifest:

```powershell
python scripts/verify_authoritative_artifacts.py
```

The script prints one `OK` or `FAIL` line per file and exits non-zero on any mismatch.

## Installation

Requires Python 3.10+. Tested environment: Python 3.14, Django 6.0.2, djangorestframework 3.16.1, torch 2.10.0+cu128, numpy 2.3.5, scikit-learn 1.8.0, reportlab 5.0.1. A CUDA GPU is optional; on CPU the stage trajectories were the same and the score differed by at most 2.2e-4. Loading the scaler under scikit-learn 1.8.0 prints an `InconsistentVersionWarning` because it was pickled with 1.9.1.

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

`reportlab` is required: `server/views.py` imports it at start-up.

## Database setup

After cloning the repository, the local database may be empty or not exist yet. Run migrations before starting the server; this creates Django's built-in tables, including `django_session`, which is required for login and sessions.

```powershell
python manage.py migrate
python manage.py createsuperuser
python manage.py check
```

The dashboard pages are restricted to staff accounts, so a staff or superuser account is needed.

## Running the dashboard

```powershell
python manage.py runserver --noreload
```

Open `http://127.0.0.1:8000/`, sign in, then:

- `/dashboard/` describes the authoritative system and checks that its artifacts are present (it does not load the model).
- `/dashboard/forecast/authoritative/` runs the model on a replay window. **The first request loads the model and takes about 20 seconds**; later requests take milliseconds.
- `/dashboard/forecast/` is the legacy, non-authoritative page.

## Authoritative API

```text
GET /api/authoritative-predict/?index=<n>
```

Returns the full engine JSON for replay window `n` of the test split. Valid indices are `0` to `6194` (`0` is a high-score attack window, `1000` a low-score benign one); `index` defaults to `0`. Login is required. An index past the end or a non-integer returns HTTP 400.

## Reproduction workflow

1. Place the ten window files in `../data/windows/`.
2. `python -m pip install -r requirements.txt`
3. `python scripts/verify_authoritative_artifacts.py` and confirm every line is `OK`.
4. `python manage.py migrate`, then `python manage.py createsuperuser`.
5. `python manage.py runserver --noreload`
6. Sign in and open `/dashboard/forecast/authoritative/`, or request `/api/authoritative-predict/?index=0`.
7. Compare `provenance.checkpoint_sha256`, `provenance.scaler_sha256` and `provenance.feature_schema_sha256` in the response with the values above.

## Tests

```powershell
python -m unittest tests.test_phase9k_integration tests.test_phase9l_django_integration
python -m unittest tests.test_phase10b_remediation
```

The first command covers the engine and the Django integration. The second covers the submission remediation checks (pages, README claims, artifact provenance). Both load the model and need the window files.

## Reproducibility and provenance

The authoritative source (the engine and every module it imports), the checkpoint, the scaler, the Django wiring, the tests and the audit reports are committed to Git; `git ls-files` shows what is tracked. Every prediction carries a `provenance` block with the artifact hashes and `git rev-parse HEAD` at run time, and the dashboard shows it. The artifact manifest is `experiments/results/phase10b_submission_remediation/authoritative_artifacts.json`.

## Evaluation evidence

From the Phase 10A audit (`experiments/results/phase10a_ai_core_final_audit/`, see `phase10a_report.md`):

- The checkpoint and scaler hashes match the recorded values.
- The engine reproduces the frozen Phase 7B predictions on all 6,195 test samples: to about 5e-13 under the frozen mixed-precision protocol, and to at most 2.4e-3 in the shipped fp32 path, where 6 of 37,170 stage cells flip.
- Inference is deterministic, and the stage trajectory comes from the stage head.

## Known limitations

- The **Temporal Transformer is the stronger binary detector** at the audited controlled-FPR operating point (PR-AUC 0.8638 vs 0.8487, test F1 0.8279 vs 0.7471). The World Model is kept for its future-state rollout, stage trajectory and explanation.
- **Early-warning evidence is limited.** 93.1% of test positives are windows already under attack; only 122 test windows are true onsets (onset ROC-AUC 0.834, onset recall 43.4% at the frozen threshold). No lead-time analysis exists.
- **No calibrated probability.** Test ECE is 0.112 and Platt scaling made it worse.
- **No per-step attack probability.** The learned per-step risk head (Phase 9N) did not beat a trivial baseline and is excluded.
- The stage head predicts `INITIAL_ACCESS` although the test split has no true examples of it, and the data has no Reconnaissance or Exfiltration.
- **Unseen-attack generalization is not established.** In leave-one-family-out experiments (Transformer and logistic regression, not the World Model) Bot showed partial transfer (ROC-AUC 0.678 vs 0.771 seen) and Infiltration none (0.475 vs 0.879); every family is confounded with its capture day.
- The output is a generic attack-risk score over a fixed six-window (60 s) horizon, not an infiltration-specific probability.
- The demo runs on a stored replay window only: no live network detection, no PCAP input, no packet+flow fusion.

## Legacy pipeline (non-authoritative)

`run_pipeline.py`, `train.py`, `feature_engine/`, `ingestion/` and `manage.py capture_traffic` belong to an earlier live-capture prototype. Its feature vector is not the 157-feature schema above, its checkpoint (`models/attack_forecaster.pt`) is not shipped, and without it `AttackForecastService` falls back to a simple heuristic. Nothing in the authoritative demo depends on it.

## Security notes

- Do not expose the development server to the public internet.
- `db/settings.py` contains a development `SECRET_KEY` and `DEBUG = True`; replace both before any deployment.
- Restrict dashboard access to authorized operators.
- Treat model output as decision support and confirm incidents with logs and endpoint evidence.

## Project layout

```text
manage.py, db/                  Django project
dashboard/, api/                Authoritative page and JSON API
world_model/inference_service.py  AuthoritativeForecastService (Django-facing) and the legacy heuristic
world_model/                    also holds the non-authoritative "Track B" model and the SIH deck
experiments/                    Frozen phase code, inference engine, results and audits
scripts/                        Dataset builders and artifact verification
tests/                          Regression suites (Phases 9H-10B)
forecasting/, explainability/, feature_engine/, ingestion/   Supporting and legacy modules
```
