"""Train and persist the small production risk forecaster used by inference_service."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_pipeline import _prepare_events
from feature_engine.flow_features import FlowFeatureEngine
from feature_engine.temporal_features import TemporalFeatureEngine
from world_model.inference_service import AttackRiskForecaster


def train(csv_path: Path, output_path: Path, limit: int, epochs: int) -> None:
    events = _prepare_events(csv_path, limit)
    flows = FlowFeatureEngine().build(events)
    windows = TemporalFeatureEngine(window_seconds=10).build(flows)
    states = windows.select(
        [column for column in windows.columns if column != "window_start"]
    ).fill_null(0.0).to_numpy()
    if len(states) == 0:
        raise ValueError("The input did not produce any temporal network states.")

    states = states.astype("float32")
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    states = states / (1.0 + abs(states).max(axis=0, keepdims=True))
    features = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
    labels = torch.full((len(states), 1), 0.5, dtype=torch.float32)
    if "label" in events.columns:
        attack_ratio = float(
            events.select(
                (~events.get_column("label").cast(str).str.to_lowercase().eq("benign")).mean()
            ).item()
        )
        labels.fill_(attack_ratio)

    model = AttackRiskForecaster(states.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)
    print(f"Saved trained model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("../cic.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/attack_forecaster.pt"))
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train(args.csv, args.output, args.limit, args.epochs)


if __name__ == "__main__":
    main()
