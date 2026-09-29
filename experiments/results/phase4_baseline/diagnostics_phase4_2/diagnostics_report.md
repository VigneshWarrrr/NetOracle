# Phase 4.2 Inference and Threshold Diagnostics

This evaluation used the existing Phase 4 models only. No training, retraining, calibration, dataset modification, split change, or checkpoint overwrite was performed.

## Verification

- Reconstructed input shape: `(6, 157)`.
- Numeric feature count: `157`.
- LSTM parameters: `279169` (expected `279169`).
- LSTM inference device: `cuda:0`.
- Validation/test samples: `6195` / `6195`.

## Reproduced thresholds

| Model | Recorded Phase 4 threshold | Reproduced threshold | Absolute difference |
|---|---:|---:|---:|
| logistic | 0.564703837884 | 0.564703837884 | 0 |
| lstm | 0.00076760322554 | 0.00076760322554 | 0 |

Thresholds were reproduced from validation predictions with the original candidate rule. Test probabilities were not used for selection.

## Validation operating points

### logistic

| Operating point | Threshold | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| original_validation_f1 | 0.564703837884 | 0.674781 | 0.684414 | 0.679563 | 0.237302 |
| validation_max_f1 | 0.564703837884 | 0.674781 | 0.684414 | 0.679563 | 0.237302 |
| validation_max_f2 | 2.83738064995e-42 | 0.418672 | 1.000000 | 0.590231 | 0.998890 |
| validation_recall_near_90 | 1.85683647968e-07 | 0.395893 | 0.900077 | 0.549912 | 0.988066 |
| validation_recall_near_80 | 0.227817750462 | 0.534950 | 0.800154 | 0.641212 | 0.500416 |
| validation_min_fpr_recall_at_least_80 | 0.227817750462 | 0.534950 | 0.800154 | 0.641212 | 0.500416 |

### lstm

| Operating point | Threshold | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| original_validation_f1 | 0.00076760322554 | 0.614969 | 0.922454 | 0.737963 | 0.415487 |
| validation_max_f1 | 0.00076760322554 | 0.614969 | 0.922454 | 0.737963 | 0.415487 |
| validation_max_f2 | 0.000499395304359 | 0.602583 | 0.935957 | 0.733152 | 0.444074 |
| validation_recall_near_90 | 0.00108281173743 | 0.620314 | 0.900077 | 0.734456 | 0.396336 |
| validation_recall_near_80 | 0.00336466892622 | 0.626586 | 0.800154 | 0.702813 | 0.343047 |
| validation_min_fpr_recall_at_least_80 | 0.00336466892622 | 0.626586 | 0.800154 | 0.702813 | 0.343047 |

Full requested fixed-threshold results are in `threshold_sweep_validation.csv` and `threshold_sweep_test.csv`.

## Original-threshold test generalization

| Model | Threshold | Precision | Recall | F1 | FPR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.564703837884 | 0.584603 | 0.805445 | 0.677481 | 0.227662 | 1420 | 3423 | 1009 | 343 |
| lstm | 0.00076760322554 | 0.430169 | 0.894498 | 0.580954 | 0.471345 | 1577 | 2343 | 2089 | 186 |

## Calibration diagnostic

Brier scores are diagnostic only; probabilities were not calibrated or changed.

| Split | Logistic Brier | LSTM Brier |
|---|---:|---:|
| validation | 0.21769693 | 0.22846958 |
| test | 0.17058633 | 0.13955112 |

Lower Brier score indicates better combined probability accuracy and calibration relative to the observed labels. Brier score alone does not separate calibration from discrimination.

## Answers

1. **Is the LSTM ranking useful?** Yes. Its reproduced test PR-AUC and ROC-AUC remain `0.853651` and `0.885276`, above Logistic Regression's `0.845412` and `0.849503`. At the original threshold it sacrifices precision/FPR for recall, so ranking quality and operating-point quality are different here.
2. **Is the very low threshold caused by class weighting/score scale?** The evidence supports a score-scale and class-weight contribution, but cannot isolate causality. The LSTM uses training-only `pos_weight=7.209185`; its validation F1 threshold is `0.0007676032`, while the Logistic threshold is `0.5647038`. The LSTM's validation Brier score is also worse (`0.228470` versus `0.217697`), consistent with poorer probability interpretation on validation.
3. **How much does performance improve as the LSTM threshold increases?** On test, LSTM FPR falls from `0.471345` at `0.0007676` to `0.397112` at `0.01`, `0.219765` at `0.10`, `0.123646` at `0.50`, and `0.062049` at `0.90`. Recall changes from `0.894498` to `0.820760`, `0.762904`, `0.745888`, and `0.736812` respectively. At threshold `0.90`, LSTM F1 is `0.778543`, higher than its original-threshold F1 `0.580954`, although this fixed test comparison is diagnostic and does not select a final threshold.
4. **Is the LSTM producing excessive benign scores?** Not globally relative to Logistic Regression. On test, LSTM benign mean/median/P75 are `0.136698/0.000399/0.071871`, versus Logistic's `0.279946/0.135731/0.525725`. However, the LSTM threshold is below its benign median, so `47.13%` of benign samples cross the original threshold. The problem is the operating threshold and benign upper tail, not uniformly higher benign scores.
5. **Does source-day shift explain the high FPR?** It contributes strongly. At the original threshold, LSTM FPR is `1.000` on Friday-23-02, `0.9948` on Thursday-22-02, `0.7801` on Tuesday-20-02, and `0.6914` on Wednesday-21-02. On the all-negative Friday-23-02 test partition, LSTM marks every sample positive. Other days are much better, including `0.0298` on Thursday-01-03 and `0.0693` on Thursday-15-02. This is direct evidence that source-day regimes concentrate the false positives.
6. **Does the validation-selected threshold generalize to test?** Only partially. LSTM validation at its selected threshold has precision `0.614969`, recall `0.922454`, F1 `0.737963`, and FPR `0.415487`; test becomes precision `0.430169`, recall `0.894498`, F1 `0.580954`, and FPR `0.471345`. Logistic changes less severely in FPR, from validation `0.237302` to test `0.227662`.
7. **Is calibration a significant issue?** It is significant for threshold use, but not uniformly worse by Brier score. Validation Brier is worse for LSTM (`0.228470` versus `0.217697`), while test Brier is better (`0.139551` versus `0.170586`). Because Brier mixes calibration and discrimination and class priors differ, reliability diagrams or calibration curves would be needed for a definitive calibration claim. No probabilities were modified.

## Conclusion

**F. combination**, dominated by threshold/score-scale mismatch and source-day distribution shift, with class weighting as a plausible contributor. The LSTM has useful ranking signal and is not simply assigning higher scores to every benign sample. Its very low validation-selected threshold makes benign upper-tail scores operationally expensive, and the cost is amplified on particular test source days. The evidence does not establish architecture as the primary cause, and it does not isolate class weighting from score-scale behavior.

## Recommended next experiment

Run one controlled **LSTM class-weight ablation**: retrain the identical architecture once with unweighted `BCEWithLogitsLoss` (`pos_weight=1`), keeping the same data, preprocessing, splits, seed policy, validation threshold selection, and test evaluation. This isolates whether the training class weight is driving the low score scale and source-day false-positive behavior.
