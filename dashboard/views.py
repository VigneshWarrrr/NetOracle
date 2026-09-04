from pathlib import Path

from django.conf import settings
from django.shortcuts import redirect

from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from logs.models import LogEntry, LogSource
from alerts.models import Alert
from feature_engine.models import FeatureWindow
from feature_engine.services import ingest_live_events

from run_pipeline import run_pipeline

def get_forecast_context():
    model_path = settings.BASE_DIR / 'models' / 'attack_forecaster.pt'
    has_live_events = FeatureWindow.objects.filter(stream_key='global').exists()
    try:
        if has_live_events:
            forecast = ingest_live_events([], stream_key='global', model_path=model_path)
        else:
            forecast = None
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        forecast = None
        forecast_error = str(error)
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
        context.update(get_forecast_context())
        return context


class NetworkRiskForecastView(AdminDashboardView):
    template_name = 'dashboard/forecast.html'

    def post(self, request, *args, **kwargs):
        return redirect('dashboard:forecast')

    def get_context_data(self, **kwargs):
        context = super(AdminDashboardView, self).get_context_data(**kwargs)
        context.update(get_forecast_context())
        return context
