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
- Treat model output as decision support and confirm incidents with logs and endpoint evidence.
