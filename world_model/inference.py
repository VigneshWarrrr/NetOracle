"""
NetOracle Real Traffic Inference

Real CIC CSV
-> timestamp filtering
-> 10-second network states
-> training normalization
-> last 200 seconds
-> recursive 50-second forecast
"""

import os

import numpy as np
import torch
import yaml

from .feature_extraction import FeatureExtractor
from .world_model import NetworkWorldModel


class NetOracleInference:

    def __init__(
        self,
        model_path="models/best_world_model.pth",
        config_path="configs/config.yaml",
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "[*] Inference device:",
            self.device
        )

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.input_dim = int(
            self.checkpoint["input_dim"]
        )

        self.feature_names = (
            self.checkpoint["feature_names"]
        )

        self.threshold = float(
            self.checkpoint.get(
                "threshold",
                0.5
            )
        )

        self.model = NetworkWorldModel(

            input_dim=self.input_dim,

            state_dim=int(
                self.config["model"]["state_dim"]
            ),

            hidden_dim=int(
                self.config["model"]["hidden_dim"]
            ),

            num_heads=int(
                self.config["model"]["num_heads"]
            ),

            num_layers=int(
                self.config["model"]["num_layers"]
            ),

            forecast_horizon=int(
                self.config["data"]["forecast_horizon"]
            ),

            num_mitre_stages=7,

            dropout=float(
                self.config["model"]["dropout"]
            ),

        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state"]
        )

        self.model.eval()

        self.extractor = FeatureExtractor(
            config_path
        )

        # Restore training-only normalization.
        self.extractor.scaler.mean_ = np.asarray(
            self.checkpoint["scaler_mean"],
            dtype=np.float64,
        )

        self.extractor.scaler.scale_ = np.asarray(
            self.checkpoint["scaler_scale"],
            dtype=np.float64,
        )

        self.extractor.scaler.var_ = (
            self.extractor.scaler.scale_ ** 2
        )

        self.extractor.scaler.n_features_in_ = (
            len(self.extractor.scaler.mean_)
        )

        self.extractor.scaler_fitted = True

    def prepare_csv(
        self,
        csv_path,
        start_time=None,
        end_time=None,
    ):

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        df = self.extractor.load_file(
            csv_path,
            start_time=start_time,
            end_time=end_time,
        )

        (
            states,
            observed_attacks,
            observed_stages,
        ) = self.extractor.dataframe_to_states(

            df,

            window_seconds=int(
                self.config["data"][
                    "time_window_seconds"
                ]
            ),
        )

        sequence_length = int(
            self.config["data"][
                "sequence_length"
            ]
        )

        if len(states) < sequence_length:

            raise ValueError(
                "Not enough 10-second windows. "
                f"Need {sequence_length}, got {len(states)}."
            )

        normalized = (
            self.extractor.scaler
            .transform(states)
            .astype(np.float32)
        )

        # Current network context:
        # latest 20 x 10 sec = 200 seconds.
        current_sequence = (
            normalized[
                -sequence_length:
            ]
        )

        x = torch.from_numpy(
            current_sequence
        ).float()

        x = (
            x.unsqueeze(0)
            .to(self.device)
        )

        return {
            "input_tensor": x,
            "states": states,
            "observed_attacks": observed_attacks,
            "observed_stages": observed_stages,
            "last_timestamp": df["Timestamp"].max(),
            "flow_count": len(df),
        }

    @torch.no_grad()
    def predict_timeline(
        self,
        csv_path,
        start_time=None,
        end_time=None,
        stride=5,
    ):
        """
        Scan through traffic chronologically.

        stride=5 means one prediction every
        5 x 10-second states = 50 seconds.
        """

        df = self.extractor.load_file(
            csv_path,
            start_time=start_time,
            end_time=end_time,
        )

        (
            states,
            observed_attacks,
            observed_stages,
        ) = self.extractor.dataframe_to_states(
            df,
            window_seconds=int(
                self.config["data"][
                    "time_window_seconds"
                ]
            ),
        )

        normalized = (
            self.extractor.scaler
            .transform(states)
            .astype(np.float32)
        )

        sequence_length = int(
            self.config["data"][
                "sequence_length"
            ]
        )

        probabilities = []
        stages = []
        observed = []
        indices = []

        for end_index in range(
            sequence_length,
            len(normalized),
            stride,
        ):

            sequence = normalized[
                end_index
                - sequence_length:
                end_index
            ]

            x = (
                torch.from_numpy(sequence)
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            output = self.model.simulate(x)

            # Maximum predicted infiltration risk
            # within the 50-second horizon.
            risk = float(
                output["attack_probs"][0]
                .max()
                .cpu()
            )

            # Stage associated with highest-risk
            # future step.
            future_probs = (
                output["attack_probs"][0]
            )

            highest_step = int(
                torch.argmax(
                    future_probs
                ).item()
            )

            stage = int(
                output[
                    "mitre_logits"
                ][0, highest_step]
                .argmax()
                .item()
            )

            probabilities.append(risk)

            stages.append(stage)

            observed.append(
                int(
                    observed_attacks[
                        end_index
                    ]
                )
                if end_index
                < len(
                    observed_attacks
                )
                else 0
            )

            indices.append(
                end_index
            )

        return {
            "risk_timeline":
                np.asarray(
                    probabilities
                ),

            "stage_timeline":
                np.asarray(
                    stages
                ),

            "observed_timeline":
                np.asarray(
                    observed
                ),

            "indices":
                np.asarray(
                    indices
                ),

            "threshold":
                self.threshold,
        }
    def predict_csv(
        self,
        csv_path,
        start_time=None,
        end_time=None,
    ):

        prepared = self.prepare_csv(
            csv_path,
            start_time,
            end_time,
        )

        output = self.model.simulate(
            prepared["input_tensor"]
        )

        attack_probabilities = (
            output["attack_probs"][0]
            .detach()
            .cpu()
            .numpy()
        )

        stage_logits = (
            output["mitre_logits"][0]
            .detach()
            .cpu()
        )

        stage_probabilities = (
            torch.softmax(
                stage_logits,
                dim=-1,
            )
            .numpy()
        )

        predicted_stages = (
            stage_probabilities
            .argmax(axis=-1)
        )

        alerts = (
            attack_probabilities
            >= self.threshold
        )

        return {
            "attack_probabilities":
                attack_probabilities,

            "predicted_stages":
                predicted_stages,

            "stage_probabilities":
                stage_probabilities,

            "alerts":
                alerts,

            "threshold":
                self.threshold,

            "last_observed_timestamp":
                prepared[
                    "last_timestamp"
                ],

            "observed_state_count":
                len(
                    prepared["states"]
                ),

            "observed_flow_count":
                prepared["flow_count"],

            "input_tensor":
                prepared[
                    "input_tensor"
                ],
        }