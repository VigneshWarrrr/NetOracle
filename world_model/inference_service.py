from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:
    class AttackRiskForecaster(torch.nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, sequence: Any) -> Any:
            return self.network(sequence[:, -1, :])


class AttackForecastService:
    """Load the persisted forecaster and expose one prediction contract."""

    def __init__(self, model_path: str | Path = "models/attack_forecaster.pt") -> None:
        self.model_path = Path(model_path)
        self.model = self._load_model()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def forecast(
        self,
        network_states: Any,
        *,
        feature_names: Optional[list[str]] = None,
        timestamps: Optional[list[Any]] = None,
        horizon_seconds: int = 60,
        fallback_risk: Optional[float] = None,
    ) -> dict[str, Any]:
        sequence = np.asarray(network_states, dtype=np.float32)
        if sequence.size == 0:
            raise ValueError("network_states must contain at least one state")

        if self.model is not None:
            risk = float(np.clip(self._predict_with_model(sequence), 0.0, 1.0))
            source = "trained_model"
        else:
            risk = float(np.clip(
                fallback_risk if fallback_risk is not None else self._heuristic_risk(sequence),
                0.0,
                1.0,
            ))
            source = "heuristic_fallback"

        timeline = []
        for index in range(len(sequence)):
            point = float(np.clip(self._predict_with_model(sequence[index:index + 1]), 0.0, 1.0)) if self.model is not None else self._heuristic_risk(sequence[index:index + 1])
            timeline.append({
                "label": str(timestamps[index]) if timestamps and index < len(timestamps) else f"Window {index + 1}",
                "risk": round(point, 4),
            })

        latest = np.nan_to_num(sequence[-1], nan=0.0, posinf=0.0, neginf=0.0)
        names = feature_names or [f"Feature {index + 1}" for index in range(len(latest))]
        contributions = sorted(
            [
                {"name": names[index], "value": round(float(abs(value)), 4)}
                for index, value in enumerate(latest)
            ],
            key=lambda item: item["value"],
            reverse=True,
        )[:8]

        return {
            "current_risk": round(float(np.clip(risk * 0.74, 0.0, 1.0)), 2),
            "forecasted_risk": round(risk, 2),
            "confidence": round(float(max(risk, 1.0 - risk)), 2),
            "prediction_horizon_seconds": horizon_seconds,
            "risk_timeline": timeline,
            "top_features": contributions,
            "model_source": source,
        }

    def _load_model(self) -> Any:
        if torch is None or not self.model_path.exists():
            return None
        artifact = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(artifact, torch.nn.Module):
            artifact.eval()
            return artifact
        if isinstance(artifact, dict) and isinstance(artifact.get("model"), torch.nn.Module):
            artifact["model"].eval()
            return artifact["model"]
        raise ValueError(f"Unsupported model artifact at {self.model_path}; expected a torch module.")

    def _predict_with_model(self, sequence: np.ndarray) -> float:
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        sequence = sequence / (1.0 + np.abs(sequence).max(axis=0, keepdims=True))
        tensor = torch.from_numpy(sequence).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, dict):
            output = output.get("risk", output.get("attack_probability"))
        return float(torch.as_tensor(output).reshape(-1)[0].item())

    @staticmethod
    def _heuristic_risk(sequence: np.ndarray) -> float:
        latest = np.nan_to_num(sequence[-1], nan=0.0, posinf=0.0, neginf=0.0)
        return float(1.0 / (1.0 + np.exp(-np.mean(latest))))