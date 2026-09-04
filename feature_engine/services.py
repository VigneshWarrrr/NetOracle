"""Publish live feature-engine output to the NetOracle forecasting pipeline."""

from .models import FeatureWindow


def publish_feature_window(features, stream_key='global'):
    """Persist one feature-engine window and evaluate it immediately."""
    if hasattr(features, 'to_dict'):
        features = features.to_dict()
    if not isinstance(features, dict):
        raise TypeError('features must be a dictionary or DataFrame row')

    window = FeatureWindow.objects.create(
        stream_key=stream_key,
        features=features,
    )

    from ai_models.services import forecaster

    forecast = forecaster.ingest_live_features(features, stream_key=stream_key)
    return window, forecast