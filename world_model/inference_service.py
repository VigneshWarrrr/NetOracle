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


class AuthoritativeForecastService:
    """Phase 9L: the ONE thin Django-facing wrapper around the authoritative,
    checkpoint-backed inference engine established in Phase 9K
    (experiments/inference_engine.py: NetOracleInferenceEngine).

    This class contains NO model logic of its own -- it only locates,
    lazily loads (once per process, class-level singleton), and calls the
    real engine, and shapes its already-JSON-serializable output for
    Django views/API responses. It never modifies the checkpoint, never
    fits/refits the scaler, and never runs with gradients enabled outside
    the engine's own explainability path.

    `AttackForecastService`/`AttackRiskForecaster` above are UNCHANGED and
    remain the (non-authoritative, heuristic-fallback) path for the
    existing live-capture feature pipeline (feature_engine.TemporalFeatureEngine),
    whose feature schema is NOT compatible with the authoritative model's
    157-feature CICFlowMeter-window schema (see
    experiments/results/phase9l_django_integration/ for why this phase did
    not attempt to reconcile the two schemas). That legacy path must never
    be presented as "the NetOracle model prediction" -- see
    experiments/results/phase9k_integration/track_reconciliation.md.
    """

    _engine = None  # process-level singleton; loaded lazily, once

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            import sys
            from pathlib import Path

            experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
            if str(experiments_dir) not in sys.path:
                sys.path.insert(0, str(experiments_dir))
            from inference_engine import NetOracleInferenceEngine  # local import: heavy (torch + checkpoint), loaded lazily

            cls._engine = NetOracleInferenceEngine()
        return cls._engine

    @classmethod
    def predict(
        cls,
        x_raw,
        *,
        source_file: str = "unknown",
        window_start: str = "unknown",
        top_k: int = 10,
        include_explanations: bool = True,
    ) -> dict:
        """x_raw: array-like [6,157], raw (unscaled) feature units, exact
        world_model_dataset.py column order. Raises ValueError/TypeError on
        invalid input (via the engine's own validate_feature_vector) --
        callers (views/API) are responsible for turning that into a clean
        HTTP error response; this method never substitutes a heuristic."""
        import numpy as np

        engine = cls.get_engine()
        x_raw = np.asarray(x_raw, dtype=np.float32)
        return engine.predict(
            x_raw,
            source_file=source_file,
            window_start=window_start,
            top_k=top_k,
            include_explanations=include_explanations,
        )

    @classmethod
    def predict_demo_sample(cls, index: int = 0) -> dict:
        """Runs a real prediction on an already-validated sample from the
        frozen Phase 3.5 TEST split (the exact mechanism Phase 9K used for
        its own deterministic-validation equivalence check). Used because
        this integration-only phase does not build a new CSV-upload/window-
        construction UI; it demonstrates the authoritative engine on real,
        already-audited data rather than on the incompatible live-capture
        feature schema (see class docstring)."""
        engine = cls.get_engine()
        sample = engine.get_test_sample(index)
        return cls.predict(
            sample["x_raw"],
            source_file=sample["source_file"],
            window_start=sample["window_start"],
        )