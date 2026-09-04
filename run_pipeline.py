"""Run the terminal network forecasting pipeline end to end."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from feature_engine.flow_features import FlowFeatureEngine
from feature_engine.graph_builder import NetworkGraphBuilder
from feature_engine.temporal_features import TemporalFeatureEngine
from ingestion.csv_reader import CSVReaderService
from forecasting.attack_chain import AttackChainBuilder
from forecasting.mitre_mapping import MitreMapper
from forecasting.victim import NextVictimForecaster
from world_model.inference_service import AttackForecastService

ATTACK_RISK_THRESHOLD = 0.70

PRECAUTIONS = {
    "BENIGN": [
        "Continue normal monitoring and keep endpoint protections enabled.",
        "Investigate any new high-severity events before treating traffic as malicious.",
    ],
    "RECONNAISSANCE": [
        "Rate-limit scanning sources and review unusual destination or port diversity.",
        "Restrict externally visible services and verify perimeter firewall rules.",
    ],
    "CREDENTIAL_ACCESS": [
        "Require MFA and temporarily rate-limit authentication attempts on the predicted target.",
        "Review failed-login events and rotate exposed credentials if compromise is suspected.",
    ],
    "INITIAL_ACCESS": [
        "Patch and isolate the exposed service on the predicted target.",
        "Restrict inbound access to trusted sources and review perimeter logs.",
    ],
}


def _prepare_events(path: Path, limit: int) -> pl.DataFrame:
    reader = CSVReaderService()
    raw = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True, n_rows=limit)
    events = reader._normalize(raw)
    return _complete_events(events)


def _complete_events(events: pl.DataFrame) -> pl.DataFrame:
    return events.with_row_index("_row_id").with_columns(
        [
            pl.col("src_ip").fill_null(
                pl.concat_str([pl.lit("192.168.1."), ((pl.col("_row_id") % 20) + 10).cast(pl.String)])
            ),
            pl.col("dst_ip").fill_null(pl.lit("192.168.1.24")),
            pl.col("src_port").fill_null(40_000 + (pl.col("_row_id") % 1_000)).cast(pl.Int64),
            pl.col("packet_count").fill_null(1.0),
            pl.col("byte_count").fill_null(0.0),
            pl.col("tcp_flags").fill_null(pl.lit("")),
            pl.col("ttl").fill_null(64),
            pl.col("tcp_window_size").fill_null(0),
        ]
    ).drop("_row_id")


def run_pipeline_events(events: pl.DataFrame, model_path: Path) -> dict:
    return _run_event_pipeline(_complete_events(events), model_path)


def run_pipeline(csv_path: Path, limit: int, model_path: Path) -> dict:
    events = _prepare_events(csv_path, limit)
    return _run_event_pipeline(events, model_path)


def _run_event_pipeline(events: pl.DataFrame, model_path: Path) -> dict:
    flows = FlowFeatureEngine().build(events)
    windows = TemporalFeatureEngine(window_seconds=10).build(flows)
    graph = NetworkGraphBuilder().build_static_graph(flows)
    network_states = windows.select(
        [column for column in windows.columns if column != "window_start"]
    ).fill_null(0.0).to_numpy()
    feature_columns = [column for column in windows.columns if column != "window_start"]
    prediction = AttackForecastService(model_path).forecast(
        network_states,
        feature_names=feature_columns,
        timestamps=[value.isoformat() for value in windows["window_start"].to_list()],
        horizon_seconds=60,
    )
    attack_type = (
        "Brute Force"
        if prediction["forecasted_risk"] >= ATTACK_RISK_THRESHOLD
        else "Benign"
    )
    mitre = MitreMapper.from_attack_label(attack_type)
    victim = NextVictimForecaster().predict(graph)
    attack_chain = AttackChainBuilder().build([mitre]).summary()
    next_stage = attack_chain.get("predicted_next_stage")
    risk = prediction["forecasted_risk"]
    if risk < ATTACK_RISK_THRESHOLD or attack_type == "Benign":
        next_step = "No access attempt predicted within the current forecast horizon."
        estimated_time = None
        precautions = PRECAUTIONS["BENIGN"]
    else:
        next_step = f"Possible progression toward {next_stage.replace('_', ' ').title()} access."
        estimated_time = max(5, round(prediction["prediction_horizon_seconds"] * (1.0 - risk)))
        precautions = PRECAUTIONS.get(next_stage, PRECAUTIONS["INITIAL_ACCESS"])
    timeline = prediction["risk_timeline"]
    historical_timeline = timeline
    latest_timestamp = datetime.fromisoformat(timeline[-1]["label"])
    step_seconds = 10
    forecast_timeline = [
        {
            "label": (latest_timestamp + timedelta(seconds=step_seconds * index)).isoformat(),
            "risk": prediction["forecasted_risk"],
        }
        for index in range(1, (prediction["prediction_horizon_seconds"] // step_seconds) + 1)
    ]
    baseline_values = [
        AttackForecastService._heuristic_risk(network_states[index:index + 1])
        for index in range(len(network_states))
    ]
    return {
        **prediction,
        "attack_type": attack_type,
        "mitre_stage": mitre.stage.name,
        "likely_victim": victim.victim,
        "events": events.height,
        "flows": flows.height,
        "network_states": windows.height,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": [
            {"source": str(source), "target": str(target)}
            for source, target in graph.edges()
        ],
        "mitre": {
            "stage": mitre.stage.name,
            "technique": mitre.technique or "Not mapped",
            "description": mitre.description or MitreMapper.get_description(mitre.stage),
            "confidence": round(mitre.confidence, 2),
        },
        "attack_chain": attack_chain,
        "historical_timeline": historical_timeline,
        "forecast_timeline": forecast_timeline,
        "model_comparison": [
            {
                "name": "Trained world model",
                "risk": prediction["forecasted_risk"],
                "selected": True,
            },
            {
                "name": "Heuristic baseline",
                "risk": round(float(sum(baseline_values) / len(baseline_values)), 2),
                "selected": False,
            },
        ],
        "victim_prediction": {
            "victim": victim.victim,
            "probability": round(victim.probability, 2),
            "reasons": victim.reasons,
            "ranked_targets": victim.ranked_targets,
        },
        "explainability_summary": {
            "next_access_step": next_step,
            "target": victim.victim if risk >= ATTACK_RISK_THRESHOLD else None,
            "attack_type": attack_type,
            "estimated_time_seconds": estimated_time,
            "precautions": precautions,
            "basis": [
                "trained world-model risk forecast",
                "MITRE attack-chain progression",
                "network graph target ranking",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("../cic.csv"))
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--model", type=Path, default=Path("models/attack_forecaster.pt"))
    args = parser.parse_args()
    result = run_pipeline(args.csv, args.limit, args.model)
    print(f"Current Risk: {result['current_risk']:.2f}")
    print(f"Forecasted Risk: {result['forecasted_risk']:.2f}")
    print(f"\nPredicted Attack:\n{result['attack_type']}")
    print("\nPrediction Horizon:\n60 seconds")
    print(f"\nLikely Victim:\n{result['likely_victim'] or 'No graph victim available'}")
    print(f"\nPipeline counts: {result['events']} events -> {result['flows']} flows -> {result['network_states']} states")
    print(f"Model source: {result['model_source']}")


if __name__ == "__main__":
    main()