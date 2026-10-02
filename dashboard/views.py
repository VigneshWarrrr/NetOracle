from pathlib import Path

import numpy as np
from django.conf import settings
from django.shortcuts import redirect

from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from logs.models import LogEntry, LogSource
from alerts.models import Alert
from feature_engine.models import FeatureWindow
from feature_engine.services import ingest_live_events

from run_pipeline import run_pipeline
from world_model.inference_service import AttackForecastService, AuthoritativeForecastService


def format_attribution(value):
    """Phase 10B, presentation only. Never changes the underlying value.

    Gradient x Input importances can be ~1e-6 when the risk score is saturated
    near 0 or 1; rounding those to four decimals showed a misleading "0.0000".
    Values >= 1e-3 keep four decimals, smaller non-zero values use scientific
    notation, and only a genuine 0.0 renders as "0".
    """
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.4f}"
    return f"{value:.2e}"


def build_attribution_rows(features):
    """Rows for the explanation panel: rank, feature, a readable value, the exact
    raw value (for the tooltip) and the value relative to the top feature."""
    values = [float(item["importance"]) for item in features]
    top = max(values) if values else 0.0
    rows = []
    for rank, (item, value) in enumerate(zip(features, values), start=1):
        relative = (value / top * 100.0) if top > 0 else 0.0
        rows.append({
            "rank": rank,
            "feature": item["feature"],
            "display": format_attribution(value),
            "raw": repr(value),
            "relative_percent": f"{relative:.0f}",
        })
    return rows


# Mirrors the artifact paths pinned in experiments/inference_engine.py (a test asserts they are equal).
AUTHORITATIVE_CHECKPOINT_RELPATH = "experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt"
AUTHORITATIVE_SCALER_RELPATH = "experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib"
AUTHORITATIVE_WINDOWS_RELPATH = "../data/windows"
EXPECTED_WINDOW_CSV_COUNT = 10


def get_authoritative_artifact_status():
    """Cheap availability check for the landing page: file existence only, no model
    load (loading takes ~20 s and belongs to the forecast page)."""
    root = Path(settings.BASE_DIR)
    windows_dir = (root / AUTHORITATIVE_WINDOWS_RELPATH).resolve()
    csv_count = len(list(windows_dir.glob("*.csv"))) if windows_dir.is_dir() else 0
    items = [
        {"label": "Authoritative checkpoint (Phase 7B stage head on the Phase 6B Run 1 backbone)",
         "path": AUTHORITATIVE_CHECKPOINT_RELPATH, "present": (root / AUTHORITATIVE_CHECKPOINT_RELPATH).is_file()},
        {"label": "Authoritative scaler (Phase 6B Run 1, train-only StandardScaler)",
         "path": AUTHORITATIVE_SCALER_RELPATH, "present": (root / AUTHORITATIVE_SCALER_RELPATH).is_file()},
        {"label": f"Phase 3.5 window data (157-feature schema and replay samples; {EXPECTED_WINDOW_CSV_COUNT} CSV files)",
         "path": AUTHORITATIVE_WINDOWS_RELPATH, "present": csv_count == EXPECTED_WINDOW_CSV_COUNT},
    ]
    return {"authoritative_artifacts": items, "authoritative_artifacts_ready": all(item["present"] for item in items)}


def get_authoritative_forecast_context():
    """Phase 9L: context for the ONE authoritative, checkpoint-backed
    prediction path (experiments/inference_engine.py via
    world_model.inference_service.AuthoritativeForecastService). Distinct
    from get_forecast_context() below, which remains the pre-existing,
    NON-AUTHORITATIVE live-capture heuristic path (left unmodified -- its
    feature schema is incompatible with the authoritative model; see
    experiments/results/phase9l_django_integration/)."""
    try:
        result = AuthoritativeForecastService.predict_demo_sample(index=0)
        error = None
    except FileNotFoundError as exc:
        result = None
        error = f"Authoritative model checkpoint not found: {exc}"
    except (ValueError, TypeError) as exc:
        result = None
        error = f"Authoritative model input validation failed: {exc}"
    except Exception as exc:  # noqa: BLE001 -- surface any other inference failure clearly, never substitute a heuristic
        result = None
        error = f"Authoritative inference failed: {exc}"
    mitre_stage_steps = []
    attribution_rows = []
    attribution_method = None
    if result is not None:
        explanations = result.get("explanations") or {}
        attribution_rows = build_attribution_rows(
            (explanations.get("current_state_evidence") or {}).get("top_attack_risk_features", [])
        )
        attribution_method = (explanations.get("future_attack_risk_prediction") or {}).get("explanation_method")
        trajectory = result["mitre_stage_trajectory"]
        mitre_stage_steps = [
            {"horizon_label": label, "stage": stage, "confidence": confidence}
            for label, stage, confidence in zip(
                trajectory["horizon_labels"],
                trajectory["per_step_stage"],
                trajectory["per_step_confidence"],
            )
        ]
    return {
        "authoritative_available": result is not None,
        "authoritative_forecast": result,
        "authoritative_error": error,
        "mitre_stage_steps": mitre_stage_steps,
        "attribution_rows": attribution_rows,
        "attribution_method": attribution_method,
    }


def build_demo_legacy_forecast(model_path: Path):
    """Provide a deterministic demo forecast when no local capture exists.

    This keeps the legacy page usable in development and demos without requiring
    a live capture dataset or a persisted model checkpoint.
    """
    rng = np.random.default_rng(42)
    states = rng.normal(0.18, 0.75, size=(12, 8)).astype(np.float32)
    timestamps = [f"2026-01-01T00:{minute:02d}:00" for minute in range(0, 12)]
    feature_names = [f"feature_{index + 1}" for index in range(states.shape[1])]
    prediction = AttackForecastService(model_path=model_path).forecast(
        states,
        feature_names=feature_names,
        timestamps=timestamps,
        horizon_seconds=60,
        fallback_risk=0.68,
    )
    risk = float(prediction['forecasted_risk'])
    if risk >= 0.70:
        attack_type = 'Brute Force'
        mitre_stage = 'Credential Access'
        next_step = 'Possible progression toward credential-access activity on a nearby target.'
        precautions = [
            'Review authentication events on the predicted target before allowing access.',
            'Rate-limit outbound authentication activity and verify MFA coverage.',
        ]
        target = 'demo-victim.internal'
    else:
        attack_type = 'Benign'
        mitre_stage = 'Benign'
        next_step = 'No access attempt predicted within the current forecast horizon.'
        precautions = [
            'Continue routine monitoring and keep endpoint protections enabled.',
            'Investigate any new high-severity events before treating traffic as malicious.',
        ]
        target = None

    forecast = {
        **prediction,
        'attack_type': attack_type,
        'mitre_stage': mitre_stage,
        'likely_victim': target,
        'events': 0,
        'flows': 0,
        'network_states': states.shape[0],
        'graph_nodes': 0,
        'graph_edges': [],
        'mitre': {
            'stage': mitre_stage,
            'technique': 'Demo heuristic' if risk >= 0.70 else 'No suspicious behavior',
            'description': 'Fallback estimate generated for the legacy dashboard while no live capture is present.',
            'confidence': round(float(prediction['confidence']), 2),
        },
        'attack_chain': {
            'stages': ['Credential Access', 'Persistence'] if risk >= 0.70 else ['No suspicious activity detected'],
            'overall_risk': round(risk, 2),
        },
        'historical_timeline': prediction['risk_timeline'],
        'forecast_timeline': [
            {'label': timestamp, 'risk': round(float(risk), 4)}
            for timestamp in timestamps[:6]
        ],
        'model_comparison': [
            {'name': 'Trained world model', 'risk': round(float(prediction['forecasted_risk']), 2), 'selected': False},
            {'name': 'Heuristic baseline', 'risk': round(float(prediction['forecasted_risk']), 2), 'selected': True},
        ],
        'victim_prediction': {
            'victim': target,
            'probability': round(float(max(risk, 0.1)), 2),
            'reasons': ['Demo fallback from legacy dashboard state'],
            'ranked_targets': [{'target': target, 'score': round(float(risk), 2)}] if target else [],
        },
        'explainability_summary': {
            'next_access_step': next_step,
            'target': target,
            'attack_type': attack_type,
            'estimated_time_seconds': 30 if risk >= 0.70 else None,
            'precautions': precautions,
            'basis': ['fallback demo forecast', 'legacy heuristic projection'],
        },
        'stream_key': 'global',
        'live_events': 0,
    }
    return forecast


def get_forecast_context():
    model_path = settings.BASE_DIR / 'models' / 'attack_forecaster.pt'
    has_live_events = FeatureWindow.objects.filter(stream_key='global').exists()
    try:
        if has_live_events:
            forecast = ingest_live_events([], stream_key='global', model_path=model_path)
        else:
            forecast = build_demo_legacy_forecast(model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        forecast = build_demo_legacy_forecast(model_path)
        forecast_error = None
    else:
        forecast_error = None if forecast else 'Start capture_traffic to receive local network data.'
    context = {
        'forecast': forecast,
        'model_available': bool(forecast and forecast['model_source'] == 'trained_model'),
        'live_stream': bool(forecast),
        'forecast_enabled': True,
        'forecast_error': forecast_error,
    }
    if forecast:
        context['forecast_peak_percent'] = round(forecast['forecasted_risk'] * 100)
        context['forecast_json'] = forecast
    return context


class AdminDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/index.html'

    def dispatch(self, request, *args, **kwargs):
        if not (request.user.is_authenticated):
            return redirect('users:login')
        if not (request.user.is_staff or request.user.is_superuser):
            return redirect('users:dashboard')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_logs'] = LogEntry.objects.all().order_by('-timestamp')[:10]
        context['all_alerts'] = Alert.objects.all().order_by('-created_at')[:10]
        context['log_sources'] = LogSource.objects.all()
        context['total_logs'] = LogEntry.objects.count()
        context['active_alerts'] = Alert.objects.filter(status='active').count() if hasattr(Alert, 'status') else Alert.objects.count()
        # Phase 10B: the landing page describes the authoritative system; it no longer runs the
        # legacy heuristic forecaster (that path stays reachable only at /dashboard/forecast/).
        context.update(get_authoritative_artifact_status())
        return context


class NetworkRiskForecastView(AdminDashboardView):
    template_name = 'dashboard/forecast.html'

    def post(self, request, *args, **kwargs):
        return redirect('dashboard:forecast')

    def get_context_data(self, **kwargs):
        context = super(AdminDashboardView, self).get_context_data(**kwargs)
        context.update(get_forecast_context())
        return context


class AuthoritativeForecastView(AdminDashboardView):
    """Phase 9L: the ONE authoritative, checkpoint-backed forecast page.
    Wraps world_model.inference_service.AuthoritativeForecastService,
    which wraps experiments.inference_engine.NetOracleInferenceEngine
    (Phase 9K). See that class's docstring for why this page uses an
    already-validated test-split sample rather than live-captured events."""

    template_name = 'dashboard/authoritative_forecast.html'

    def post(self, request, *args, **kwargs):
        return redirect('dashboard:authoritative_forecast')

    def get_context_data(self, **kwargs):
        context = super(AdminDashboardView, self).get_context_data(**kwargs)
        context.update(get_authoritative_forecast_context())
        return context
