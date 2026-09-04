"""NetOracle model adapter for the Logged_In feature pipeline."""

from collections import defaultdict, deque
from pathlib import Path
import sys


class NetOracleForecaster:
    """Load NetOracle once and forecast risk from feature-engine output."""

    BASE_FEATURES = (
        "dst_port", "protocol", "flow_duration", "total_fwd_packets",
        "total_bwd_packets", "total_length_fwd", "total_length_bwd",
        "flow_bytes_per_sec", "flow_packets_per_sec", "flow_iat_mean",
        "flow_iat_std", "flow_iat_max", "syn_flag_count", "ack_flag_count",
        "fin_flag_count", "rst_flag_count", "psh_flag_count",
        "avg_packet_size", "fwd_header_length", "fwd_bwd_ratio",
        "syn_ack_ratio", "flag_entropy", "ttl_proxy", "payload_size_proxy",
        "dst_port_normalized",
    )

    def __init__(self, checkpoint_path=None):
        project_checkpoint = Path(__file__).resolve().parent / "models" / "best_world_model.pth"
        cloned_checkpoint = Path(__file__).resolve().parents[2] / "netoracle" / "NetOracle" / "models" / "best_world_model.pth"
        self.checkpoint_paths = [Path(checkpoint_path)] if checkpoint_path else [project_checkpoint, cloned_checkpoint]
        self._model = None
        self._torch = None
        self._histories = defaultdict(lambda: deque(maxlen=20))

    def _load(self):
        from .models import ForecastSettings

        if not ForecastSettings.get_current().enabled:
            return False
        if self._model is not None:
            return True
        checkpoint_path = next((path for path in self.checkpoint_paths if path.exists()), None)
        if checkpoint_path is None:
            return False
        try:
            import torch

            netoracle_root = Path(__file__).resolve().parents[2] / "netoracle" / "NetOracle"
            if str(netoracle_root) not in sys.path:
                sys.path.insert(0, str(netoracle_root))
            from src.world_model import NetworkWorldModel

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            self._model = NetworkWorldModel(
                input_dim=checkpoint["input_dim"],
                state_dim=config["model"]["state_dim"],
                hidden_dim=config["model"]["hidden_dim"],
                num_heads=config["model"]["num_heads"],
                num_layers=config["model"]["num_layers"],
                forecast_horizon=config["data"]["forecast_horizon"],
            )
            self._model.load_state_dict(checkpoint["model_state"])
            self._model.eval()
            self._torch = torch
            return True
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            self._model = None
            return False

    @classmethod
    def state_from_features(cls, features):
        """Convert one feature-engine row into NetOracle's 25x4+1 state."""
        import numpy as np

        if hasattr(features, "to_dict"):
            features = features.to_dict(as_series=False)
        if isinstance(features, dict):
            aliases = {
                "flow_duration": ("flow_duration", "flow_duration_seconds", "mean_flow_duration"),
                "total_fwd_packets": ("total_fwd_packets", "packet_count", "total_packets"),
                "total_length_fwd": ("total_length_fwd", "byte_count", "total_bytes"),
                "flow_bytes_per_sec": ("flow_bytes_per_sec", "bytes_per_second", "mean_byte_rate"),
                "flow_packets_per_sec": ("flow_packets_per_sec", "packets_per_second", "mean_packet_rate"),
                "avg_packet_size": ("avg_packet_size", "packet_size_mean", "mean_packet_size"),
                "flow_count": ("flow_count",),
            }
            values = []
            for name in cls.BASE_FEATURES:
                names = aliases.get(name, (name,))
                value = next((features.get(alias) for alias in names if features.get(alias) is not None), 0.0)
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else 0.0
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(0.0)
            return np.asarray(
                [item for value in values for item in (value, 0.0, value, value)] + [values[-1]],
                dtype=np.float32,
            )

        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 2:
            array = array[-1]
        if array.shape != (101,):
            raise ValueError("NetOracle features must contain exactly 101 values")
        return array

    def forecast(self, features, stream_key="global"):
        """Return forecast probabilities and risk, or ``None`` without a model."""
        if not self._load():
            return None
        import numpy as np

        history = self._histories[stream_key]
        history.append(self.state_from_features(features))
        if len(history) < 20:
            return None
        batch = self._torch.tensor(np.asarray(history), dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            output = self._model(batch)
        probabilities = output["attack_probs"][0].cpu().tolist()
        stages = output["mitre_logits"][0].argmax(dim=-1).cpu().tolist()
        return {
            "risk_score": float(max(probabilities)),
            "attack_probabilities": [float(value) for value in probabilities],
            "mitre_stages": [int(value) for value in stages],
        }

    def ingest_live_features(self, features, stream_key="global"):
        """Evaluate one live feature-engine output as it arrives."""
        return self.forecast(features, stream_key=stream_key)

    def forecast_sequence(self, feature_rows, stream_key="dashboard"):
        """Forecast from an ordered collection of recent feature-engine rows."""
        history = self._histories[stream_key]
        history.clear()
        result = None
        for feature_row in feature_rows:
            result = self.forecast(feature_row, stream_key=stream_key)
        return result

    @property
    def available(self):
        """Whether a compatible trained checkpoint can be loaded."""
        return self._load()

    @property
    def enabled(self):
        from .models import ForecastSettings

        return ForecastSettings.get_current().enabled


forecaster = NetOracleForecaster()