from django.shortcuts import render, redirect

from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from logs.models import LogEntry, LogSource
from feature_engine.models import FeatureWindow
from alerts.models import Alert
from ai_models.services import forecaster
from ai_models.models import ForecastSettings
import json

def get_forecast_context():
    feature_windows = list(FeatureWindow.objects.order_by('-created_at')[:100])
    feature_windows.reverse()
    forecast = forecaster.forecast_sequence(
        [window.features for window in feature_windows]
    )
    context = {
        'forecast': forecast,
        'model_available': forecaster.available,
        'forecast_enabled': forecaster.enabled,
    }
    if forecast:
        context['forecast_peak_percent'] = round(forecast['risk_score'] * 100)
        context['forecast_windows'] = [
            {
                'probability': probability,
                'percent': round(probability * 100),
                'stage': stage,
            }
            for probability, stage in zip(
                forecast['attack_probabilities'], forecast['mitre_stages']
            )
        ]
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
        settings = ForecastSettings.get_current()
        settings.enabled = request.POST.get('forecast_enabled') == 'on'
        settings.save(update_fields=['enabled', 'updated_at'])
        return redirect('dashboard:forecast')

    def get_context_data(self, **kwargs):
        context = super(AdminDashboardView, self).get_context_data(**kwargs)
        context.update(get_forecast_context())
        return context
