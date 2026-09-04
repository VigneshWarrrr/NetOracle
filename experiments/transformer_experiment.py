from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class TransformerResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]


class PositionalEncoding(
    nn.Module
):
    """
    Adds temporal position information
    to the Transformer input.
    """

    def __init__(
        self,
        d_model: int,
        max_length: int = 500,
    ):

        super().__init__()

        encoding = torch.zeros(
            max_length,
            d_model,
        )

        positions = torch.arange(
            0,
            max_length,
            dtype=torch.float,
        ).unsqueeze(
            1
        )

        division_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float,
            )
            *
            (
                -math.log(
                    10000.0
                )
                / d_model
            )
        )

        encoding[
            :,
            0::2
        ] = torch.sin(
            positions
            * division_term
        )

        encoding[
            :,
            1::2
        ] = torch.cos(
            positions
            * division_term
        )

        encoding = encoding.unsqueeze(
            0
        )

        self.register_buffer(
            "encoding",
            encoding,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        sequence_length = x.size(
            1
        )

        return (
            x
            +
            self.encoding[
                :,
                :sequence_length,
                :
            ]
        )


class AttackTransformer(
    nn.Module
):
    """
    Transformer-based temporal
    attack forecasting model.

    Input:
        (batch, sequence_length, features)

    Output:
        attack logits
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=(
                    d_model * 4
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_projection(
            x
        )

        x = self.positional_encoding(
            x
        )

        encoded = self.transformer(
            x
        )

        last_state = encoded[
            :,
            -1,
            :
        ]

        last_state = self.dropout(
            last_state
        )

        return self.classifier(
            last_state
        )


class TransformerExperiment:
    """
    Training and evaluation wrapper for
    Transformer-based attack forecasting.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.0005,
        device: Optional[str] = None,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.num_classes = num_classes

        self.model = AttackTransformer(
            input_size=input_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(
            self.device
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    def train(
        self,
        X_train: Any,
        y_train: Any,
        epochs: int = 20,
        batch_size: int = 64,
    ) -> List[
        Dict[str, float]
    ]:

        dataset = self._create_dataset(
            X_train,
            y_train,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        history = []

        self.model.train()

        for epoch in range(
            epochs
        ):

            total_loss = 0.0

            correct = 0

            total = 0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(
                    self.device
                )

                y_batch = y_batch.to(
                    self.device
                )

                self.optimizer.zero_grad()

                logits = self.model(
                    X_batch
                )

                loss = self.criterion(
                    logits,
                    y_batch,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                total_loss += (
                    loss.item()
                    * X_batch.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                correct += (
                    predictions
                    == y_batch
                ).sum().item()

                total += (
                    y_batch.size(0)
                )

            epoch_loss = (
                total_loss / total
            )

            epoch_accuracy = (
                correct / total
            )

            metrics = {
                "epoch": epoch + 1,
                "loss": float(
                    epoch_loss
                ),
                "accuracy": float(
                    epoch_accuracy
                ),
            }

            history.append(
                metrics
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: "
                f"{epoch_accuracy:.4f}"
            )

        return history

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:

        probabilities = (
            self.predict_proba(
                X
            )
        )

        return np.argmax(
            probabilities,
            axis=1,
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:

        X_tensor = self._prepare_tensor(
            X
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                X_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return (
            probabilities
            .cpu()
            .numpy()
        )

    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    def evaluate(
        self,
        X_test: Any,
        y_test: Any,
    ) -> TransformerResult:

        y_test = np.asarray(
            y_test
        )

        predictions = self.predict(
            X_test
        )

        probabilities = self.predict_proba(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        roc_auc = self._calculate_auc(
            y_test,
            probabilities,
        )

        return TransformerResult(
            accuracy=float(
                accuracy
            ),

            precision=float(
                precision
            ),

            recall=float(
                recall
            ),

            f1_score=float(
                f1
            ),

            roc_auc=roc_auc,
        )

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save(
        self,
        path: str,
    ) -> None:

        torch.save(
            self.model.state_dict(),
            path,
        )

    def load(
        self,
        path: str,
    ) -> None:

        state_dict = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

    # --------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------

    def _create_dataset(
        self,
        X: Any,
        y: Any,
    ) -> TensorDataset:

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.int64,
        )

        if X.ndim != 3:

            raise ValueError(
                "Expected X shape: "
                "(samples, sequence_length, features)"
            )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.long,
        )

        return TensorDataset(
            X_tensor,
            y_tensor,
        )

    def _prepare_tensor(
        self,
        X: Any,
    ) -> torch.Tensor:

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 3:

            raise ValueError(
                "Expected shape: "
                "(samples, sequence_length, features)"
            )

        return torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

    # --------------------------------------------------
    # AUC
    # --------------------------------------------------

    def _calculate_auc(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> Optional[float]:

        try:

            unique_classes = np.unique(
                y_true
            )

            if len(unique_classes) < 2:

                return None

            if probabilities.shape[1] == 2:

                return float(
                    roc_auc_score(
                        y_true,
                        probabilities[:, 1],
                    )
                )

            return float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            )

        except Exception:

            return None