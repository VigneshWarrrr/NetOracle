# Phase 5 Temporal Transformer Baseline

This is one controlled Transformer experiment using the unchanged Phase 3.5 dataset interface and six-state forecasting target. The Transformer uses `pos_weight=1.0` and mean pooling only; existing Phase 4 artifacts were read only.

## Transformer verification

- Input: `6 x 157`
- Sequence aggregation: mean pooling over all six Transformer outputs
- Parameters: `286081`
- Device: `cuda:0`
- First training batch: `cuda:0`
- Best epoch: `2`
- Training duration: `19.135` seconds

## Comparison

| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | test | 0.584603 | 0.805445 | 0.677481 | 0.845412 | 0.849503 | 0.227662 | 0.564703837884 | 1420 | 3423 | 1009 | 343 |
| weighted_lstm | test | 0.430169 | 0.894498 | 0.580954 | 0.853651 | 0.885276 | 0.471345 | 0.00076760322554 | 1577 | 2343 | 2089 | 186 |
| unweighted_lstm | validation | 0.630005 | 0.910494 | 0.744714 | 0.800445 | 0.833157 | 0.384679 | 0.000555350503419 | 2360 | 2217 | 1386 | 232 |
| unweighted_lstm | test | 0.424242 | 0.857629 | 0.567674 | 0.857640 | 0.885502 | 0.462996 | 0.000555350503419 | 1512 | 2380 | 2052 | 251 |
| temporal_transformer | validation | 0.729199 | 0.730324 | 0.729761 | 0.815627 | 0.838105 | 0.195115 | 0.043264452368 | 1893 | 2900 | 703 | 699 |
| temporal_transformer | test | 0.617992 | 0.806580 | 0.699803 | 0.863815 | 0.876298 | 0.198330 | 0.043264452368 | 1422 | 3553 | 879 | 341 |

## Interpretation

Thresholds were selected from validation predictions only; test labels were not used for threshold selection. Mean pooling over the Transformer output sequence is the sole aggregation method and is not configurable in this experiment.

PR-AUC and ROC-AUC describe ranking, while F1, FPR, and recall describe the selected operating point. Fixed diagnostic thresholds are in the threshold sweep files. No model is declared superior from a single metric.
