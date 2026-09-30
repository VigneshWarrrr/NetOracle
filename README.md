<<<<<<< HEAD
# NetOracle Risk Forecasting Platform

NetOracle is a Django-based network security platform focused on forecasting attack risk from engineered network traffic. It converts packet or flow data into temporal network states, runs a persisted PyTorch forecaster, builds a communication graph, and presents the result through a risk forecasting dashboard.

## Risk Forecasting

The forecasting path is:

```text
Network events or CIC-IDS CSV
        |
        v
CSVReaderService / Scapy capture
        |
        v
FlowFeatureEngine
        |
        v
TemporalFeatureEngine
        |
        v
NetworkGraphBuilder
        |
        v
models/attack_forecaster.pt
        |
        v
Risk forecast, attack chain, MITRE mapping, target analysis, dashboard
```

The dashboard provides:

- Current and forecasted risk scores
- Historical versus projected risk timeline
- Prediction horizon and confidence
- Predicted attack type
- Attack-chain progression
- MITRE ATT&CK stage and technique mapping
- Dynamic network graph
- Model comparison
- Explainability summary with possible next access step and precautions

The dashboard uses the persisted model at:

```text
models/attack_forecaster.pt
```

It does not create a random model during inference. If the model artifact is missing, the dashboard cannot provide a trained-model forecast.

## Project Layout

```text
manage.py                       Django entry point
run_pipeline.py                 Terminal forecasting pipeline
train.py                        Training entry point
models/                         Persisted model artifacts
world_model/                    PyTorch model and inference services
feature_engine/                 Flow, temporal, and graph features
forecasting/                    Attack-chain, MITRE, and victim logic
explainability/                 Model and graph explanation tools
ingestion/                      CSV, PCAP, Zeek, and live capture adapters
dashboard/                     Django dashboard views
templates/dashboard/            Risk forecast dashboard UI
static/css/                     Shared dashboard styling
datasets/                       Dataset package boundary
experiments/                    Model experiments and baselines
```

## Requirements

- Python 3.10 or newer
- Windows: Npcap for local packet capture
- Administrator privileges may be required for interface capture
- A working PyTorch installation compatible with the selected Python environment

The project has been tested with the local virtual environment named `myenv`.

## Installation

From the repository workspace root:

```powershell
cd "C:\VIGNESHWARAN\SIH 2026\Logged_IN"
.\myenv\Scripts\python.exe -m pip install -r .\Logged_In\requirements.txt
```

Or activate the environment first:

```powershell
cd "C:\VIGNESHWARAN\SIH 2026\Logged_IN\Logged_In"
..\myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Database Setup

```powershell
python manage.py migrate
python manage.py check
```

Create an administrator for the protected dashboard:

```powershell
python manage.py createsuperuser
```

## Run the Dashboard

```powershell
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

Open the risk forecast page at:

```text
http://127.0.0.1:8000/dashboard/forecast/
```

## Run the Terminal Pipeline

The terminal pipeline is useful for validating the forecasting path without starting Django:

```powershell
python run_pipeline.py ..\cic.csv --limit 200 --model models\attack_forecaster.pt
```

It reports the current risk, forecasted risk, predicted attack, horizon, likely target, pipeline counts, and model source.

## Train the Model

To create or replace the persisted model artifact:

```powershell
python train.py ..\cic.csv --output models\attack_forecaster.pt --limit 2000 --epochs 25
```

Inference loads the resulting file through `world_model/inference_service.py`.

## Capture Local Network Traffic

The application can capture IP packets directly from the machine where it runs. This does not require an external ingestion API, but it only sees traffic available to that machine's network interface.

On Windows:

1. Install Npcap.
2. Open an elevated PowerShell when required by the adapter.
3. Start Django in one terminal.
4. Start capture in a second terminal:

```powershell
python manage.py capture_traffic
```

Specify an interface when necessary:

```powershell
python manage.py capture_traffic --iface "Wi-Fi" --batch-size 20
```

Captured packets are normalized by `ingestion/live_capture.py`, processed by the feature engines, persisted in the feature-window store, and used for the model forecast.

A host normally sees its own traffic and traffic delivered to its interface. Monitoring every device on a switched network requires router visibility, a mirrored switch port, a gateway sensor, or agents on those devices.

## Data Sources

The repository includes `cic.csv` at the workspace level for repeatable pipeline testing. CIC-IDS data is used for training and offline validation. Live capture uses the active interface and does not require the CSV.

## Forecast Interpretation

Risk is a model probability-like score in the range `0.0` to `1.0`. The dashboard applies a conservative attack-label threshold. A low or borderline risk score is reported as benign rather than being labeled as a confirmed attack.

MITRE stage, attack-chain progression, graph target ranking, and precautionary actions are supporting interpretations of the model and engineered network state. They are not proof that a compromise has occurred.

## Development Checks

Run these before committing changes:

```powershell
python manage.py check
python -m py_compile run_pipeline.py train.py
python run_pipeline.py ..\cic.csv --limit 200 --model models\attack_forecaster.pt
```

## Security Notes

- Do not expose the development server directly to the public internet.
- Replace development secrets before deployment.
- Restrict packet capture and dashboard access to authorized operators.
=======
# NetOracle: attack-risk forecasting from network-flow windows

NetOracle (SIH 2026, problem statement SIH26153, *AI based Network Attack Forecasting from Network Traffic Data*) is a Django platform around a PyTorch **World Model** trained on the CSE-CIC-IDS2018 flow data. From the six most recent 10-second network-state windows it forecasts the next six windows.

> **Status of the demo.** The dashboard runs the model on a *validated replay window* taken from the held-out test split of the Phase 3.5 dataset. It does **not** perform live network detection. Packet/PCAP and packet+flow fusion are not implemented.

## Authoritative path and legacy path

| | Authoritative (the SIH demo) | Legacy, NON-AUTHORITATIVE |
|---|---|---|
| Inference | `experiments/inference_engine.py` (`NetOracleInferenceEngine`) | `world_model/inference_service.py::AttackForecastService` (a sigmoid-of-mean heuristic) |
| Django wiring | `dashboard/views.py`, `api/views.py` -> `world_model/inference_service.py::AuthoritativeForecastService` -> `NetOracleInferenceEngine` | `/dashboard/forecast/`, `run_pipeline.py`, `train.py`, `capture_traffic` |
| Checkpoint | `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt` | `models/attack_forecaster.pt` is **not shipped**; without it the legacy path only returns the heuristic |
| Status | frozen, audited (Phases 9K, 9L, 10A) | kept for compatibility only; do not treat its output as a NetOracle model prediction |

Also not part of the authoritative path: the "Track B" model in `world_model/` (`world_model.py`, `inference.py`, `build_project.py`, `src/streamlit.py`), and the Phase 9N per-step risk experiment (its verdict is RED and it is excluded).

```text
Django page / API
   -> world_model/inference_service.py::AuthoritativeForecastService   (loads the engine once per process)
   -> experiments/inference_engine.py::NetOracleInferenceEngine.predict()
   -> validate (6, 157) finite  ->  frozen scaler  ->  VectorWorldModelWithStageHead
   -> attack-risk score, six-step MITRE-stage trajectory, six predicted states, Gradient x Input attribution
```

## Model contract

**Input:** one array of shape `(6, 157)` (float, raw units): six consecutive 10-second windows `t-5 ... t`, 157 features each, in the exact column order of `data/windows/*.csv` (schema SHA-256 `f5117fda8518c9e373a4260ff65281800a4760d8af16bb97411ca8371b85bb73`). Wrong type, wrong shape or non-finite values are rejected; nothing is padded or reordered.

**Output** (JSON, from `NetOracleInferenceEngine.predict`):

- `whole_horizon_attack_probability.value` is the **attack-risk score** for the *whole* horizon `t+1 ... t+6`. It is one score, **not** a six-element per-step series, and it is **not a calibrated probability** (the key name is historical). Read it as a ranking score.
- `mitre_stage_trajectory` is a predicted stage for **each of the six steps** with a stage-class softmax confidence. There are six classes: `BENIGN`, `INITIAL_ACCESS`, `CREDENTIAL_ACCESS`, `LATERAL_MOVEMENT`, `COMMAND_AND_CONTROL`, `IMPACT`. The labels come from a reasoned mapping of CIC-IDS-2018 attack labels; they are not MITRE ATT&CK ground truth and not kill-chain inference.
- `future_state_rollout` is six predicted 157-feature network states.
- `explanations` is **post-hoc Gradient x Input** feature attribution (`explanations.future_attack_risk_prediction.explanation_method` reports `Gradient x Input` for every call). Attribution magnitudes are tiny when the score is saturated near 0 or 1 (for example 1e-6): a small number does not mean zero importance, compare relative values. SHAP is **not** used; see `requirements.txt`.
- `provenance` records the checkpoint / scaler / schema hashes, the Git commit, device and timing.

## Authoritative artifacts and SHA-256

| Role | Path | SHA-256 |
|---|---|---|
| Checkpoint (Phase 6B Run 1 backbone + trained Phase 7B stage head, 356,452 parameters) | `experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt` | `f9d16f1943aeeed4f355659b85e90cb6cf8cb94ae84d2fcbd473c59fa3342fe4` |
| Scaler (train-only `StandardScaler`, 157 features) | `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib` | `b3a0cea7f9d2d0fc59de163c8768116cbd95abbe3b30c127c44a2322f0c24881` |
| Backbone reference (its 39 tensors are bit-identical to those inside the checkpoint) | `experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt` | `6374a9c47d722215e64a3cae1b1e24c425a89dad26407b147b153af03404fb79` |

The checkpoint and scaler are committed to Git (a narrow `.gitignore` exception; the broad `*.pt` rule still applies to every other checkpoint).

**The window dataset is external to Git.** The engine reads `../data/windows/*.csv` (ten Phase 3.5 files, about 71 MB) to obtain the 157 feature names and the replay samples. Keep this layout:

```text
<workspace>/
  NetOracle/        <- this repository
  data/windows/     <- the ten *_TrafficForML_CICFlowMeter.csv window files (not in Git)
```

Each file's SHA-256 is in `experiments/results/phase10b_submission_remediation/authoritative_artifacts.json`. There is no immutable public download for these derived files: obtain them from the submission bundle, or rebuild them with `scripts/build_temporal_dataset.py` from the public CSE-CIC-IDS2018 CSVs (a rebuild is **not** verified to be byte-identical; compare against the recorded hashes). Check everything with:

```powershell
python scripts/verify_authoritative_artifacts.py
```

## Setup

Requires Python 3.10+. Tested environment: Python 3.14, Django 6.0.2, djangorestframework 3.16.1, torch 2.10.0+cu128, numpy 2.3.5, scikit-learn 1.8.0, reportlab 5.0.1 (a CUDA GPU is optional; CPU gives the same stage trajectories, score difference at most 2.2e-4). Loading the scaler under scikit-learn 1.8.0 prints an `InconsistentVersionWarning` because it was pickled with 1.9.1; the values were verified against a fresh train-only refit.

```powershell
python -m pip install -r requirements.txt   # reportlab is REQUIRED: server/views.py imports it at start-up
python manage.py migrate
python manage.py createsuperuser            # the dashboard is staff-only
python manage.py check
python manage.py runserver --noreload
```

Open `http://127.0.0.1:8000/`, sign in, then:

- `/dashboard/` describes the authoritative system and checks that its artifacts are present (it does not load the model).
- `/dashboard/forecast/authoritative/` runs the model on the replay window. **The first request loads the model and takes about 20 seconds**; later requests take milliseconds.
- `GET /api/authoritative-predict/?index=<n>` returns the full JSON for replay window `n` of the test split (`0` is a confident attack window, `1000` a benign one). Login is required.
- `/dashboard/forecast/` is the legacy, non-authoritative page.

## What is verified, and known limitations

Verified (Phase 10A audit, `experiments/results/phase10a_ai_core_final_audit/`): checkpoint and scaler hashes; the engine reproduces the frozen Phase 7B predictions on all 6,195 test samples (about 5e-13 under the frozen mixed-precision protocol; up to 2.4e-3 in the shipped fp32 path, with 6 of 37,170 stage cells flipping); deterministic inference; the stage trajectory really comes from the stage head.

Limitations, stated plainly:

- The **Temporal Transformer is the stronger binary detector** at the audited controlled-FPR operating point (PR-AUC 0.8638 vs 0.8487, test F1 0.8279 vs 0.7471). The World Model is kept for its future-state rollout, stage trajectory and explanation.
- **Early-warning evidence is limited.** 93.1% of test positives are windows already under attack; only 122 test windows are true onsets (onset ROC-AUC 0.834, onset recall 43.4% at the frozen threshold). No lead-time analysis exists.
- **No calibrated probability.** Test ECE is 0.112 and Platt scaling made it worse.
- **No per-step attack probability.** The learned per-step risk head (Phase 9N) did not beat a trivial baseline and is excluded.
- The stage head predicts `INITIAL_ACCESS` although the test split has no true examples of it, and the data has no Reconnaissance or Exfiltration.
- **Unseen-attack generalization is not established.** In leave-one-family-out experiments (Transformer and logistic regression, not the World Model) Bot showed partial transfer (ROC-AUC 0.678 vs 0.771 seen) and Infiltration none (0.475 vs 0.879); every family is confounded with its capture day.
- The output is a generic attack-risk score over a fixed six-window (60 s) horizon, not an infiltration-specific probability.
- No live network detection, no PCAP input, no packet+flow fusion. No Docker configuration is provided.

## Legacy pipeline (non-authoritative)

`run_pipeline.py`, `train.py`, `feature_engine/`, `ingestion/` and `manage.py capture_traffic` belong to an earlier live-capture prototype. Its feature vector is not the 157-feature schema above, its checkpoint (`models/attack_forecaster.pt`) is not shipped, and without it `AttackForecastService` falls back to a simple heuristic. Nothing in the authoritative demo depends on it.

## Reproducibility and provenance

The authoritative source (engine and every module it imports), the checkpoint, the scaler, the Django wiring, the tests and the audit reports are committed to Git; `git ls-files` proves what is tracked. The dashboard's provenance panel shows `git rev-parse HEAD` at run time. The remediation record is in `experiments/results/phase10b_submission_remediation/` (`phase10b_report.md`, `authoritative_artifacts.json`, `git_provenance.txt`).

## Tests

```powershell
python -m unittest tests.test_phase10b_remediation
python -m unittest tests.test_phase9k_integration tests.test_phase9l_django_integration
```

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

## Security notes

- Do not expose the development server to the public internet.
- Replace development secrets before deployment.
- Restrict dashboard access to authorized operators.
>>>>>>> 2d48ed3a754465266137017828c32b683cc44185
- Treat model output as decision support and confirm incidents with logs and endpoint evidence.
