"""
NetOracle Static Baseline

Logistic Regression baseline for fair comparison against
the Temporal World Model.

IMPORTANT:
- Uses the exact same TRAIN / VALIDATION / TEST splits.
- Uses ONLY the latest observed network state.
- Has no temporal history.
- Predicts T+1 attack probability.
- Threshold is selected using validation only.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaselineModel:

    def __init__(self):

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )

        self.threshold = 0.5

    # =====================================================
    # STATIC REPRESENTATION
    # =====================================================

    @staticmethod
    def prepare(data):

        X, _, y_attack, _ = data

        # Baseline receives ONLY S_t.
        #
        # World Model receives:
        # S_(t-19) ... S_t
        #
        # Therefore Logistic Regression has
        # no temporal sequence information.

        X_static = X[:, -1, :]

        # Predict T+1 only.
        y = y_attack[:, 0].astype(int)

        return (
            X_static,
            y,
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def fit(
        self,
        train_data,
    ):

        X_train, y_train = (
            self.prepare(
                train_data
            )
        )

        print(
            "[*] Training Logistic Regression..."
        )

        print(
            "    Samples:",
            len(X_train),
        )

        print(
            "    Features:",
            X_train.shape[1],
        )

        print(
            "    Attack ratio:",
            f"{100 * y_train.mean():.2f}%",
        )

        self.model.fit(
            X_train,
            y_train,
        )

    # =====================================================
    # PROBABILITIES
    # =====================================================

    def predict_probability(
        self,
        data,
    ):

        X, y = self.prepare(
            data
        )

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        return (
            probabilities,
            y,
        )

    # =====================================================
    # THRESHOLD SELECTION
    # =====================================================

    def select_threshold(
        self,
        validation_data,
        max_fpr=None,
    ):

        probabilities, targets = (
            self.predict_probability(
                validation_data
            )
        )

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.arange(
            0.01,
            0.991,
            0.01,
        ):

            predictions = (
                probabilities
                >= threshold
            ).astype(int)

            if max_fpr is not None:

                cm = confusion_matrix(
                    targets,
                    predictions,
                    labels=[0, 1],
                )

                tn, fp, _, _ = (
                    cm.ravel()
                )

                fpr = (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else 0.0
                )

                if fpr > max_fpr:
                    continue

            current_f1 = f1_score(
                targets,
                predictions,
                zero_division=0,
            )

            if current_f1 > best_f1:

                best_f1 = (
                    current_f1
                )

                best_threshold = (
                    float(
                        threshold
                    )
                )

        self.threshold = (
            best_threshold
        )

        return (
            best_threshold,
            float(best_f1),
        )

    # =====================================================
    # METRICS
    # =====================================================

    def evaluate(
        self,
        data,
        threshold=None,
    ):

        if threshold is None:

            threshold = (
                self.threshold
            )

        probabilities, targets = (
            self.predict_probability(
                data
            )
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            targets,
            predictions,
        )

        precision = precision_score(
            targets,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            targets,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            targets,
            predictions,
            zero_division=0,
        )

        if (
            len(
                np.unique(targets)
            )
            > 1
        ):

            auc = roc_auc_score(
                targets,
                probabilities,
            )

        else:

            auc = 0.5

        cm = confusion_matrix(
            targets,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        return {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1),

            "auc_roc":
                float(auc),

            "false_positive_rate":
                float(fpr),

            "threshold":
                float(threshold),

            "true_negative":
                int(tn),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "true_positive":
                int(tp),
        }