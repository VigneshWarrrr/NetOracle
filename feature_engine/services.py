"""Publish live feature-engine output to the NetOracle forecasting pipeline."""

from .models import FeatureWindow
from world_model.inference_service import AttackForecastService

import polars as pl

from run_pipeline import run_pipeline_events


LIVE_EVENT_SCHEMA = {
    "timestamp": pl.String,
    "src_ip": pl.String,
    "dst_ip": pl.String,
    "src_port": pl.Int64,
    "dst_port": pl.Int64,
    "protocol": pl.String,
    "packet_count": pl.Int64,
    "byte_count": pl.Float64,
    "duration": pl.Float64,
    "tcp_flags": pl.String,
    "ttl": pl.Int64,
    "tcp_window_size": pl.Int64,
    "source_type": pl.String,
}


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

    numeric_values = [value for value in features.values() if isinstance(value, (int, float))]
    forecast = AttackForecastService().forecast([numeric_values]) if numeric_values else None
    return window, forecast


def ingest_live_events(events, stream_key="global", model_path=None):
    if isinstance(events, dict):
        events = [events]
    for event in events:
        FeatureWindow.objects.create(stream_key=stream_key, features=event)

    recent = list(
        FeatureWindow.objects.filter(stream_key=stream_key)
        .order_by("-created_at")[:200]
        .values_list("features", flat=True)
    )
    recent.reverse()
    if not recent:
        raise ValueError("No captured events are available.")
    events = pl.DataFrame(recent, schema_overrides=LIVE_EVENT_SCHEMA)
    result = run_pipeline_events(events, model_path)
    result["stream_key"] = stream_key
    result["live_events"] = len(recent)
    return result