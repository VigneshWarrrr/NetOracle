from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class LSTMResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]


class AttackLSTM(nn.Module):
    """
    Temporal attack forecasting model.

    Input:
        (batch, sequence_length, features)

    Output:
        attack logits
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        lstm_output, _ = self.lstm(
            x
        )

        last_state = lstm_output[
            :,
            -1,
            :,
        ]

        last_state = self.dropout(
            last_state
        )

        return self.classifier(
            last_state
        )


class LSTMExperiment:
    """
    Training and evaluation wrapper for
    LSTM attack forecasting.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
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

        self.model = AttackLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
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
    ) -> List[Dict[str, float]]:

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
    ) -> LSTMResult:

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

        return LSTMResult(
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
    # MODEL SAVE
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
    # HELPERS
    # --------------------------------------------------

    def _create_dataset(
        self,
        X: Any,
        y: Any,
    ) -> TensorDataset:

        X_tensor = torch.tensor(
            np.asarray(
                X,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        y_tensor = torch.tensor(
            np.asarray(
                y,
                dtype=np.int64,
            ),
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

    def _calculate_auc(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> Optional[float]:

        try:

            if len(
                np.unique(y_true)
            ) < 2:

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