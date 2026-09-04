from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True

except ImportError:

    torch = None
    nn = None

    TORCH_AVAILABLE = False


try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:

    shap = None

    SHAP_AVAILABLE = False


@dataclass
class ExplanationResult:
    """
    Stores the explanation of a model prediction.
    """

    prediction: float

    prediction_name: str

    top_features: List[Dict[str, Any]]

    feature_scores: Dict[str, float]

    method: str

    summary: str

    raw_values: Optional[np.ndarray] = None


class WorldModelExplainer:
    """
    Explains predictions produced by a PyTorch world model.

    Supports two approaches:

    1. SHAP-based explanation
    2. Gradient × Input fallback

    For temporal models, the explanation is aggregated across
    the sequence dimension so that the final result explains
    the importance of each network-state feature.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        device: Optional[str] = None,
    ):

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required. Install it using: "
                "pip install torch"
            )

        self.model = model

        self.feature_names = list(
            feature_names
        )

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def explain_attack_risk(
        self,
        sequence: Any,
        target_index: Optional[int] = None,
        top_k: int = 10,
        method: str = "auto",
    ) -> ExplanationResult:
        """
        Explains the attack-risk prediction.

        Parameters
        ----------
        sequence:
            Network-state sequence.

            Expected shape:

            (sequence_length, feature_count)

            or:

            (batch, sequence_length, feature_count)

        target_index:
            Optional output index.

        top_k:
            Number of most important features.

        method:
            "auto"
            "shap"
            "gradient"
        """

        tensor = self._prepare_tensor(
            sequence
        )

        prediction = self._get_attack_prediction(
            tensor,
            target_index,
        )

        if (
            method in {"auto", "shap"}
            and SHAP_AVAILABLE
        ):

            try:

                result = (
                    self._explain_with_shap(
                        tensor=tensor,
                        target_index=target_index,
                        top_k=top_k,
                    )
                )

                return result

            except Exception:

                if method == "shap":
                    raise

        return self._explain_with_gradients(
            tensor=tensor,
            target_index=target_index,
            top_k=top_k,
            prediction=prediction,
        )

    def explain_future_state(
        self,
        sequence: Any,
        future_feature_index: int,
        top_k: int = 10,
    ) -> ExplanationResult:
        """
        Explains which current features contributed to a
        predicted future network-state feature.
        """

        tensor = self._prepare_tensor(
            sequence
        )

        prediction_tensor = (
            self._forward_model(
                tensor
            )
        )

        output = (
            self._extract_future_state(
                prediction_tensor
            )
        )

        if output is None:

            raise ValueError(
                "The model did not return a future-state "
                "prediction."
            )

        if (
            future_feature_index < 0
            or future_feature_index
            >= output.shape[-1]
        ):

            raise IndexError(
                "future_feature_index is outside the "
                "future-state output."
            )

        target = output[
            ...,
            future_feature_index
        ].mean()

        prediction = float(
            target.detach()
            .cpu()
            .item()
        )

        return self._gradient_explanation(
            tensor=tensor,
            target=target,
            prediction=prediction,
            prediction_name=(
                f"Future {self.feature_names[future_feature_index]}"
                if future_feature_index
                < len(self.feature_names)
                else "Future State Feature"
            ),
            top_k=top_k,
        )

    # --------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------

    def _explain_with_shap(
        self,
        tensor: Any,
        target_index: Optional[int],
        top_k: int,
    ) -> ExplanationResult:

        background = torch.zeros_like(
            tensor
        )

        def model_function(
            x: Any,
        ) -> Any:

            x = torch.tensor(
                x,
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():

                output = (
                    self._forward_model(
                        x
                    )
                )

                attack_output = (
                    self._extract_attack_output(
                        output
                    )
                )

                if attack_output is None:

                    raise ValueError(
                        "Could not find attack prediction "
                        "in model output."
                    )

                probabilities = (
                    self._convert_to_probability(
                        attack_output
                    )
                )

                if (
                    target_index is not None
                    and probabilities.shape[-1] > 1
                ):

                    probabilities = probabilities[
                        ...,
                        target_index
                    ]

                return (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )

        background_np = (
            background
            .detach()
            .cpu()
            .numpy()
        )

        input_np = (
            tensor
            .detach()
            .cpu()
            .numpy()
        )

        explainer = shap.GradientExplainer(
            model_function,
            background_np,
        )

        shap_values = explainer.shap_values(
            input_np
        )

        values = np.asarray(
            shap_values
        )

        feature_scores = (
            self._aggregate_sequence_importance(
                values
            )
        )

        prediction = self._get_attack_prediction(
            tensor,
            target_index,
        )

        return self._build_result(
            feature_scores=feature_scores,
            prediction=prediction,
            prediction_name="Attack Risk",
            top_k=top_k,
            method="SHAP GradientExplainer",
            raw_values=values,
        )

    # --------------------------------------------------
    # GRADIENT EXPLANATION
    # --------------------------------------------------

    def _explain_with_gradients(
        self,
        tensor: Any,
        target_index: Optional[int],
        top_k: int,
        prediction: float,
    ) -> ExplanationResult:

        input_tensor = (
            tensor.clone()
            .detach()
            .requires_grad_(True)
        )

        output = (
            self._forward_model(
                input_tensor
            )
        )

        attack_output = (
            self._extract_attack_output(
                output
            )
        )

        if attack_output is None:

            raise ValueError(
                "Could not identify attack prediction "
                "from model output."
            )

        probability = (
            self._convert_to_probability(
                attack_output
            )
        )

        if probability.ndim == 0:

            target = probability

        elif (
            target_index is not None
            and probability.shape[-1] > 1
        ):

            target = probability[
                ...,
                target_index
            ].mean()

        elif probability.shape[-1] > 1:

            target = probability.max(
                dim=-1
            ).values.mean()

        else:

            target = probability.mean()

        return self._gradient_explanation(
            tensor=input_tensor,
            target=target,
            prediction=prediction,
            prediction_name="Attack Risk",
            top_k=top_k,
        )

    def _gradient_explanation(
        self,
        tensor: Any,
        target: Any,
        prediction: float,
        prediction_name: str,
        top_k: int,
    ) -> ExplanationResult:

        self.model.zero_grad(
            set_to_none=True
        )

        if tensor.grad is not None:

            tensor.grad.zero_()

        target.backward()

        gradients = (
            tensor.grad
            .detach()
        )

        importance = (
            gradients.abs()
            * tensor.detach().abs()
        )

        values = (
            importance
            .cpu()
            .numpy()
        )

        feature_scores = (
            self._aggregate_sequence_importance(
                values
            )
        )

        return self._build_result(
            feature_scores=feature_scores,
            prediction=prediction,
            prediction_name=prediction_name,
            top_k=top_k,
            method="Gradient × Input",
            raw_values=values,
        )

    # --------------------------------------------------
    # MODEL OUTPUT HANDLING
    # --------------------------------------------------

    def _forward_model(
        self,
        tensor: Any,
    ) -> Any:

        self.model.eval()

        return self.model(
            tensor
        )

    def _extract_attack_output(
        self,
        output: Any,
    ) -> Optional[Any]:
        """
        Attempts to support multiple model output formats.

        Supported:

        Tensor

        Tuple:
            (future_state, attack_output)

        Dict:
            {
                "attack_probability": ...,
                "attack_risk": ...,
                "attack_logits": ...
            }
        """

        if torch.is_tensor(
            output
        ):

            return output

        if isinstance(
            output,
            dict,
        ):

            possible_keys = [
                "attack_probability",
                "attack_probabilities",
                "attack_risk",
                "attack_logits",
                "risk",
                "risk_logits",
            ]

            for key in possible_keys:

                if key in output:

                    return output[key]

            return None

        if isinstance(
            output,
            (
                tuple,
                list,
            ),
        ):

            if len(output) >= 2:

                return output[1]

            if len(output) == 1:

                return output[0]

        return None

    def _extract_future_state(
        self,
        output: Any,
    ) -> Optional[Any]:

        if isinstance(
            output,
            dict,
        ):

            possible_keys = [
                "future_state",
                "next_state",
                "state_prediction",
                "future_state_prediction",
            ]

            for key in possible_keys:

                if key in output:

                    return output[key]

            return None

        if isinstance(
            output,
            (
                tuple,
                list,
            ),
        ):

            if len(output) >= 1:

                return output[0]

        return None

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def _get_attack_prediction(
        self,
        tensor: Any,
        target_index: Optional[int],
    ) -> float:

        with torch.no_grad():

            output = (
                self._forward_model(
                    tensor
                )
            )

            attack_output = (
                self._extract_attack_output(
                    output
                )
            )

            if attack_output is None:

                raise ValueError(
                    "Could not identify attack output."
                )

            probabilities = (
                self._convert_to_probability(
                    attack_output
                )
            )

            if probabilities.ndim == 0:

                return float(
                    probabilities
                    .cpu()
                    .item()
                )

            if (
                target_index is not None
                and probabilities.shape[-1] > 1
            ):

                value = probabilities[
                    ...,
                    target_index
                ].mean()

            elif probabilities.shape[-1] > 1:

                value = probabilities.max(
                    dim=-1
                ).values.mean()

            else:

                value = probabilities.mean()

            return float(
                value
                .cpu()
                .item()
            )

    def _convert_to_probability(
        self,
        output: Any,
    ) -> Any:

        output = output.float()

        if output.numel() == 1:

            value = output.reshape(
                -1
            )[0]

            if (
                value < 0
                or value > 1
            ):

                return torch.sigmoid(
                    output
                )

            return output

        if (
            output.min() < 0
            or output.max() > 1
        ):

            return torch.softmax(
                output,
                dim=-1,
            )

        return output

    # --------------------------------------------------
    # FEATURE AGGREGATION
    # --------------------------------------------------

    def _aggregate_sequence_importance(
        self,
        values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Aggregates:

        batch
        sequence

        into one importance score per feature.
        """

        values = np.asarray(
            values
        )

        values = np.abs(
            values
        )

        while values.ndim > 1:

            values = values.mean(
                axis=0
            )

        if values.ndim == 0:

            values = np.array(
                [float(values)]
            )

        feature_scores = {}

        feature_count = min(
            len(self.feature_names),
            len(values),
        )

        for index in range(
            feature_count
        ):

            feature_scores[
                self.feature_names[index]
            ] = float(
                values[index]
            )

        return feature_scores

    # --------------------------------------------------
    # RESULT BUILDING
    # --------------------------------------------------

    def _build_result(
        self,
        feature_scores: Dict[
            str,
            float,
        ],
        prediction: float,
        prediction_name: str,
        top_k: int,
        method: str,
        raw_values: Optional[
            np.ndarray
        ] = None,
    ) -> ExplanationResult:

        ranked = sorted(
            feature_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_features = []

        maximum = (
            ranked[0][1]
            if ranked
            else 1.0
        )

        if maximum == 0:
            maximum = 1.0

        for feature, score in ranked[
            :top_k
        ]:

            normalized = (
                score
                / maximum
            )

            top_features.append(
                {
                    "feature": feature,

                    "importance": float(
                        score
                    ),

                    "relative_importance": float(
                        normalized
                    ),

                    "percentage": round(
                        normalized * 100,
                        2,
                    ),
                }
            )

        summary = (
            self._generate_summary(
                prediction=prediction,
                prediction_name=prediction_name,
                top_features=top_features,
            )
        )

        return ExplanationResult(
            prediction=float(
                prediction
            ),

            prediction_name=prediction_name,

            top_features=top_features,

            feature_scores=feature_scores,

            method=method,

            summary=summary,

            raw_values=raw_values,
        )

    def _generate_summary(
        self,
        prediction: float,
        prediction_name: str,
        top_features: List[
            Dict[str, Any]
        ],
    ) -> str:

        if not top_features:

            return (
                f"{prediction_name}: "
                f"{prediction:.4f}. "
                "No feature attribution "
                "was available."
            )

        important = ", ".join(
            item["feature"]
            for item in top_features[:3]
        )

        return (
            f"{prediction_name}: "
            f"{prediction:.4f}. "
            f"The strongest contributing "
            f"features were: {important}."
        )

    # --------------------------------------------------
    # INPUT PREPARATION
    # --------------------------------------------------

    def _prepare_tensor(
        self,
        sequence: Any,
    ) -> Any:

        if torch.is_tensor(
            sequence
        ):

            tensor = (
                sequence
                .float()
                .to(
                    self.device
                )
            )

        else:

            tensor = torch.tensor(
                np.asarray(
                    sequence,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=self.device,
            )

        if tensor.ndim == 2:

            tensor = tensor.unsqueeze(
                0
            )

        if tensor.ndim != 3:

            raise ValueError(
                "Expected sequence shape "
                "(sequence_length, features) "
                "or "
                "(batch, sequence_length, features)."
            )

        return tensor