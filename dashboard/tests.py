from django.test import TestCase

from dashboard.views import get_forecast_context
from feature_engine.models import FeatureWindow


class LegacyForecastFallbackTests(TestCase):
    def test_returns_demo_forecast_when_no_live_capture_exists(self):
        FeatureWindow.objects.all().delete()

        context = get_forecast_context()

        self.assertIsNotNone(context['forecast'])
        self.assertTrue(context['live_stream'])
        self.assertIn('forecasted_risk', context['forecast'])
        self.assertIn('model_comparison', context['forecast'])
        self.assertIsNone(context['forecast_error'])
