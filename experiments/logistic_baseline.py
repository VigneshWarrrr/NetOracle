from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    classification_report: str


class LogisticBaselineExperiment:
    """
    Traditional machine-learning baseline.

    This model predicts attack risk from a single
    network-state feature vector.

    It does NOT model temporal relationships.

    Therefore it is useful as a baseline for comparing
    LSTM and Transformer forecasting models.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[Any] = "balanced",
    ):

        self.scaler = StandardScaler()

        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

        self.is_fitted = False

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    def train(
        self,
        X_train: Any,
        y_train: Any,
    ) -> None:

        X_train = self._prepare_features(
            X_train
        )

        y_train = np.asarray(
            y_train
        )

        X_scaled = self.scaler.fit_transform(
            X_train
        )

        self.model.fit(
            X_scaled,
            y_train,
        )

        self.is_fitted = True

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:

        self._check_fitted()

        X = self._prepare_features(
            X
        )

        X_scaled = self.scaler.transform(
            X
        )

        return self.model.predict(
            X_scaled
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:

        self._check_fitted()

        X = self._prepare_features(
            X
        )

        X_scaled = self.scaler.transform(
            X
        )

        return self.model.predict_proba(
            X_scaled
        )

    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    def evaluate(
        self,
        X_test: Any,
        y_test: Any,
    ) -> BaselineResult:

        self._check_fitted()

        X_test = self._prepare_features(
            X_test
        )

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

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
        )

        return BaselineResult(
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

            classification_report=report,
        )

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    def get_feature_importance(
        self,
        feature_names: list[str],
    ) -> Dict[str, float]:

        self._check_fitted()

        coefficients = np.abs(
            self.model.coef_
        )

        if coefficients.ndim > 1:

            coefficients = coefficients.mean(
                axis=0
            )

        importance = {}

        for index, feature_name in enumerate(
            feature_names
        ):

            if index < len(coefficients):

                importance[
                    feature_name
                ] = float(
                    coefficients[index]
                )

        return dict(
            sorted(
                importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _prepare_features(
        self,
        X: Any,
    ) -> np.ndarray:

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim == 3:

            # For sequence input, use the latest
            # network state as the baseline input.
            X = X[:, -1, :]

        if X.ndim != 2:

            raise ValueError(
                "Expected feature shape "
                "(samples, features) or "
                "(samples, sequence, features)."
            )

        return X

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

    def _check_fitted(
        self,
    ) -> None:

        if not self.is_fitted:

            raise RuntimeError(
                "Train the model before "
                "calling predict or evaluate."
            )