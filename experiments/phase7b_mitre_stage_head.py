"""Phase 7B: MITRE Stage Head architecture -- frozen Phase 6B backbone + new head.

    X [B, 6, 157]
        --> frozen Phase 6B Run-1 backbone (state_encoder, temporal_context,
            transition x6, state_decoder, attack_head -- ALL INHERITED,
            UNCHANGED, LOADED FROM THE FROZEN RUN-1 CHECKPOINT)
        --> z_future [B, 6, 128]
        --> existing AttackForecastHead -> attack_logit [B]      (unchanged path)
        --> new MitreStageHead          -> stage_logits [B, 6, 6] (new path)

This file does not modify `phase6b_vector_world_model.py` in any way -- it
only imports `VectorWorldModel` and subclasses it. The subclass's `forward()`
duplicates the ~8-line call sequence of the base class's `forward()` (calling
the SAME inherited submodule instances -- not new logic), solely so that the
locally-computed `z_future` can also be returned and fed to the new head.
`state_encoder`, `temporal_context`, `transition`, `state_decoder`, and
`attack_head` are never redefined here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from phase6b_vector_world_model import (
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    FORECAST_HORIZON_WINDOWS,
    HISTORY_WINDOWS,
    INPUT_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    VectorWorldModel,
)

NUM_STAGE_CLASSES = 6  # the six MitreStage values with real coverage in this dataset (see phase7b_stage_targets.py)
STAGE_HEAD_PREFIX = "stage_head."


class MitreStageHead(nn.Module):
    """Per-horizon-step MITRE stage classifier.

    Shared weights are applied independently to each of the 6 predicted
    future latents z_future[:, k, :] (the same "apply one small module per
    timestep via nn.Linear's automatic broadcast over leading dims" pattern
    already used by StateDecoder in phase6b_vector_world_model.py). Input
    and output never touch the ground-truth future labels; both are pure
    functions of the model's own predicted rollout latents.
    """

    def __init__(self, d_model: int = D_MODEL, num_classes: int = NUM_STAGE_CLASSES, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        """z_future: [B, 6, D] -> stage_logits: [B, 6, num_classes]."""
        return self.net(z_future)


class VectorWorldModelWithStageHead(VectorWorldModel):
    """Adds a MitreStageHead sibling to the existing (unchanged) AttackForecastHead.

    Inherits `state_encoder`, `temporal_context`, `transition`, `state_decoder`,
    `attack_head` from VectorWorldModel unmodified (same nn.Module instances,
    same weights once a Phase 6B checkpoint is loaded). Only `stage_head` is new.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        history_windows: int = HISTORY_WINDOWS,
        forecast_horizon: int = FORECAST_HORIZON_WINDOWS,
        num_stage_classes: int = NUM_STAGE_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            history_windows=history_windows,
            forecast_horizon=forecast_horizon,
        )
        self.stage_head = MitreStageHead(d_model=d_model, num_classes=num_stage_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, 6, 157] -> (y_hat, attack_logit, stage_logits, z_future).

        Identical computation to VectorWorldModel.forward() for y_hat and
        attack_logit (same submodule calls, same order) -- z_future and
        stage_logits are the only additions. When the backbone is frozen
        (requires_grad=False on every inherited parameter), autograd
        automatically detaches y_hat/attack_logit/z_future from the graph
        (verified by test_frozen_backbone_blocks_gradient in tests.py) --
        only stage_logits (via stage_head's trainable parameters) carries a
        gradient. No future label ever appears on the right-hand side of any
        expression in this method.
        """
        z_hist = self.state_encoder(x)
        context = self.temporal_context(z_hist)
        z = context[:, -1, :]
        rollout = []
        for _ in range(self.forecast_horizon):
            z = self.transition(z)
            rollout.append(z)
        z_future = torch.stack(rollout, dim=1)
        y_hat = self.state_decoder(z_future)
        attack_logit = self.attack_head(z_future)
        stage_logits = self.stage_head(z_future)
        return y_hat, attack_logit, stage_logits, z_future


def load_frozen_backbone(checkpoint_path: Path, device: torch.device) -> VectorWorldModelWithStageHead:
    """Loads a Phase 6B `best_model.pt` checkpoint into a fresh
    VectorWorldModelWithStageHead via strict=False. Raises immediately
    (HARD STOP per the caller's protocol) if anything other than exactly the
    new `stage_head.*` parameters is missing, or if the checkpoint contains
    any key this architecture does not recognize.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VectorWorldModelWithStageHead().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    non_stage_head_missing = [key for key in missing_keys if not key.startswith(STAGE_HEAD_PREFIX)]
    if non_stage_head_missing:
        raise RuntimeError(
            f"Checkpoint load produced unexpected missing (non-stage_head) keys: {non_stage_head_missing}"
        )
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint contains keys not present in this architecture: {unexpected_keys}")
    if not any(key.startswith(STAGE_HEAD_PREFIX) for key in missing_keys):
        raise RuntimeError("Expected stage_head.* parameters to be missing from the Phase 6B checkpoint (they are new), but none were reported missing.")

    return model


def freeze_backbone(model: VectorWorldModelWithStageHead) -> None:
    """Sets requires_grad=False on every parameter except stage_head.*."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(STAGE_HEAD_PREFIX))


def assert_only_stage_head_trainable(model: VectorWorldModelWithStageHead) -> None:
    for name, parameter in model.named_parameters():
        expected = name.startswith(STAGE_HEAD_PREFIX)
        if parameter.requires_grad != expected:
            raise RuntimeError(
                f"Backbone-freeze invariant violated: parameter {name!r} has requires_grad="
                f"{parameter.requires_grad}, expected {expected}."
            )


def set_frozen_backbone_training_mode(model: VectorWorldModelWithStageHead) -> None:
    """Backbone stays in eval() (deterministic, no dropout noise, matching
    the exact Run-1 inference behavior) at all times; only the new
    stage_head is put in train() mode so its own dropout is active during
    stage-head training."""
    model.eval()
    model.stage_head.train()


def backbone_state_dict(model: VectorWorldModelWithStageHead) -> dict[str, torch.Tensor]:
    """Snapshot of every non-stage_head parameter/buffer, for before/after
    drift verification."""
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith(STAGE_HEAD_PREFIX)
    }


def backbone_unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    if set(before.keys()) != set(after.keys()):
        return False
    return all(torch.equal(before[key], after[key]) for key in before)


@torch.no_grad()
def predict_all(
    model: VectorWorldModelWithStageHead,
    x: np.ndarray,
    device: torch.device,
    use_amp: bool,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eval-mode forward pass over `x`. Returns (y_hat, attack_probability,
    stage_probability) as numpy arrays. attack_probability is sigmoid(attack_logit)
    (matching phase6b_vector_world_model.predict()); stage_probability is
    softmax(stage_logits, dim=-1), shape [N, 6, num_stage_classes]."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    y_hats: list[np.ndarray] = []
    attack_probabilities: list[np.ndarray] = []
    stage_probabilities: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            y_hat, attack_logit, stage_logits, _z_future = model(batch)
        y_hats.append(y_hat.float().cpu().numpy())
        attack_probabilities.append(torch.sigmoid(attack_logit.float()).cpu().numpy())
        stage_probabilities.append(torch.softmax(stage_logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(y_hats), np.concatenate(attack_probabilities), np.concatenate(stage_probabilities)
