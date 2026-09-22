# Phase 4 Baseline Comparison

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.584603 | 0.805445 | 0.677481 | 0.845412 | 0.849503 | 0.227662 | 0.564704 | 1420 | 3423 | 1009 | 343 |
| lstm | 0.430169 | 0.894498 | 0.580954 | 0.853651 | 0.885276 | 0.471345 | 0.000768 | 1577 | 2343 | 2089 | 186 |

Thresholds were selected using validation predictions only. The test set was used once for final evaluation.

The better model should be determined from the reported forecasting metrics, with PR-AUC prioritized because the target is imbalanced.
