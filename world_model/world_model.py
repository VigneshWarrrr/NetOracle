"""
NetOracle World Model

Temporal Transformer + recursive latent transition dynamics.

The model:
1. Encodes observed network states.
2. Learns temporal context using causal self-attention.
3. Recursively simulates future latent states.
4. Decodes each future latent state into:
   - predicted network state
   - infiltration probability
   - MITRE ATT&CK stage

This implements a genuine K-step learned rollout.
"""

import math
from typing import Dict

import torch
import torch.nn as nn


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        if d_model % 2 == 0:

            pe[:, 1::2] = torch.cos(
                position * div_term
            )

        else:

            pe[:, 1::2] = torch.cos(
                position
                * div_term[:-1]
            )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = (
            x
            + self.pe[
                :, :x.size(1)
            ]
        )

        return self.dropout(x)


# ============================================================
# RESIDUAL LATENT TRANSITION
# ============================================================

class LatentTransition(nn.Module):
    """
    Learns:

        z(t+1) = z(t) + delta(z(t))

    Residual dynamics are generally more stable than
    directly generating a completely new latent vector.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        dropout: float,
    ):

        super().__init__()

        self.transition = nn.Sequential(

            nn.LayerNorm(
                state_dim
            ),

            nn.Linear(
                state_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                state_dim,
            ),
        )

        self.output_norm = nn.LayerNorm(
            state_dim
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        delta = self.transition(z)

        future_z = z + delta

        return self.output_norm(
            future_z
        )


# ============================================================
# WORLD MODEL
# ============================================================

class NetworkWorldModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 96,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        forecast_horizon: int = 5,
        num_mitre_stages: int = 7,
        dropout: float = 0.1,
    ):

        super().__init__()

        if (
            state_dim % num_heads
            != 0
        ):

            raise ValueError(
                "state_dim must be divisible "
                "by num_heads."
            )

        self.input_dim = (
            input_dim
        )

        self.state_dim = (
            state_dim
        )

        self.forecast_horizon = (
            forecast_horizon
        )

        self.num_mitre_stages = (
            num_mitre_stages
        )

        # ----------------------------------------------------
        # 1. OBSERVED NETWORK STATE ENCODER
        # ----------------------------------------------------

        self.state_encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                state_dim,
            ),

            nn.LayerNorm(
                state_dim
            ),
        )

        # ----------------------------------------------------
        # 2. TEMPORAL MODEL
        # ----------------------------------------------------

        self.positional_encoding = (
            PositionalEncoding(

                d_model=state_dim,

                max_len=512,

                dropout=dropout,
            )
        )

        transformer_layer = (
            nn.TransformerEncoderLayer(

                d_model=
                    state_dim,

                nhead=
                    num_heads,

                dim_feedforward=
                    hidden_dim,

                dropout=
                    dropout,

                activation=
                    "gelu",

                batch_first=
                    True,

                norm_first=
                    True,
            )
        )

        self.temporal_transformer = (
            nn.TransformerEncoder(

                transformer_layer,

                num_layers=
                    num_layers,

                norm=
                    nn.LayerNorm(
                        state_dim
                    ),
            )
        )

        # ----------------------------------------------------
        # 3. LEARNED WORLD DYNAMICS
        # ----------------------------------------------------

        self.transition_model = (
            LatentTransition(

                state_dim=
                    state_dim,

                hidden_dim=
                    hidden_dim,

                dropout=
                    dropout,
            )
        )

        # ----------------------------------------------------
        # 4. FUTURE STATE DECODER
        # ----------------------------------------------------

        self.state_decoder = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                input_dim,
            ),
        )

        # ----------------------------------------------------
        # 5. INFILTRATION RISK HEAD
        # ----------------------------------------------------

        self.attack_head = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim // 2,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

        # ----------------------------------------------------
        # 6. MITRE STAGE HEAD
        # ----------------------------------------------------

        self.stage_head = nn.Sequential(

            nn.Linear(
                state_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_mitre_stages,
            ),
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        self.apply(
            self._initialize_weights
        )

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    @staticmethod
    def _initialize_weights(
        module,
    ):

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.xavier_uniform_(
                module.weight
            )

            if (
                module.bias
                is not None
            ):

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # ENCODE HISTORICAL STATES
    # ========================================================

    def encode_history(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = (
            x.size(0)
        )

        sequence_length = (
            x.size(1)
        )

        encoded = self.state_encoder(

            x.reshape(
                -1,
                self.input_dim,
            )
        )

        encoded = encoded.reshape(

            batch_size,

            sequence_length,

            self.state_dim,
        )

        encoded = (
            self.positional_encoding(
                encoded
            )
        )

        # Causal mask:
        # state t cannot attend to future states.
        causal_mask = torch.triu(

            torch.full(

                (
                    sequence_length,
                    sequence_length,
                ),

                float("-inf"),

                device=x.device,
            ),

            diagonal=1,
        )

        context = (
            self.temporal_transformer(

                encoded,

                mask=causal_mask,
            )
        )

        return context

    # ========================================================
    # RECURSIVE WORLD-MODEL ROLLOUT
    # ========================================================

    def rollout_from_latent(
        self,
        latent_state: torch.Tensor,
    ):

        predicted_states = []

        attack_probabilities = []

        mitre_logits = []

        latent_rollout = []

        current_latent = (
            latent_state
        )

        for _ in range(
            self.forecast_horizon
        ):

            # -----------------------------------------------
            # P(z_{t+1} | z_t)
            # -----------------------------------------------

            next_latent = (
                self.transition_model(
                    current_latent
                )
            )

            # -----------------------------------------------
            # Decode future network state
            # -----------------------------------------------

            predicted_state = (
                self.state_decoder(
                    next_latent
                )
            )

            # -----------------------------------------------
            # Predict infiltration probability
            # -----------------------------------------------

            attack_logit = (
                self.attack_head(
                    next_latent
                )
                .squeeze(-1)
            )

            attack_probability = (
                torch.sigmoid(
                    attack_logit
                )
            )

            # -----------------------------------------------
            # Predict MITRE stage
            # -----------------------------------------------

            stage_logits = (
                self.stage_head(
                    next_latent
                )
            )

            predicted_states.append(
                predicted_state
            )

            attack_probabilities.append(
                attack_probability
            )

            mitre_logits.append(
                stage_logits
            )

            latent_rollout.append(
                next_latent
            )

            # Critical world-model step:
            #
            # Future prediction becomes the
            # starting state of the next transition.

            current_latent = (
                next_latent
            )

        predicted_states = (
            torch.stack(
                predicted_states,
                dim=1,
            )
        )

        attack_probabilities = (
            torch.stack(
                attack_probabilities,
                dim=1,
            )
        )

        mitre_logits = (
            torch.stack(
                mitre_logits,
                dim=1,
            )
        )

        latent_rollout = (
            torch.stack(
                latent_rollout,
                dim=1,
            )
        )

        return (
            predicted_states,
            attack_probabilities,
            mitre_logits,
            latent_rollout,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if x.ndim != 3:

            raise ValueError(
                "Input must have shape "
                "[batch, sequence, features]."
            )

        if (
            x.size(-1)
            != self.input_dim
        ):

            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dim}, "
                f"received {x.size(-1)}."
            )

        # Historical temporal context.
        context = (
            self.encode_history(x)
        )

        # Current learned latent world state.
        current_latent = (
            context[:, -1, :]
        )

        # Recursive future simulation.
        (
            predicted_states,
            attack_probs,
            mitre_logits,
            latent_rollout,
        ) = self.rollout_from_latent(
            current_latent
        )

        return {

            "predicted_states":
                predicted_states,

            "attack_probs":
                attack_probs,

            "mitre_logits":
                mitre_logits,

            "latent_rollout":
                latent_rollout,

            "historical_context":
                context,
        }

    # ========================================================
    # SIMULATION API
    # ========================================================

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
    ):

        """
        Explicit API used by the dashboard/inference layer
        for K-step future simulation.
        """

        self.eval()

        return self.forward(x)