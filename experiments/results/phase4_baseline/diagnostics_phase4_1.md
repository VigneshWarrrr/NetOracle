# Phase 4.1 Diagnostic Analysis

This is a read-only analysis of the existing Phase 4 artifacts. No model was trained or retrained, no dataset was modified, and no existing artifact was overwritten.

## Evidence inspected

- `metrics.json`
- `config.json`
- `confusion_matrix.csv`
- `training_log.csv`
- `comparison_report.md`
- `logistic_regression/model.joblib`
- `logistic_regression/scaler.joblib`
- `lstm/best_model.pt`
- `lstm/scaler.joblib`
- The existing Phase 3.5 partition CSVs, read-only, for split/source-day prevalence context

The saved LSTM checkpoint contains only `model_state_dict`. No validation/test prediction probabilities, hard predictions, sample identifiers, source-day identifiers paired with predictions, or per-sample scores are saved.

## Validation-selected thresholds

The exact thresholds recorded in `metrics.json` are:

| Model | Threshold-selection source | Threshold |
|---|---|---:|
| Logistic Regression | Validation predictions, F1 selection | 0.5647038379 |
| LSTM | Validation predictions, F1 selection | 0.0007676032 |

The LSTM threshold is approximately 736 times lower than the Logistic Regression threshold. This is a major score-scale/operating-point difference, not a like-for-like comparison at threshold 0.5.

## Fixed-threshold sweep

A sweep at `0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90` cannot be computed from the existing artifacts. The necessary per-sample probabilities were not saved, and the aggregate confusion matrices cannot be inverted to recover them.

The existing test metrics therefore support comparison only at the separately selected thresholds above. No new threshold was selected from the test set.

## Existing test behavior

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.584603 | 0.805445 | 0.677481 | 0.845412 | 0.849503 | 0.227662 | 1420 | 3423 | 1009 | 343 |
| LSTM | 0.430169 | 0.894498 | 0.580954 | 0.853651 | 0.885276 | 0.471345 | 1577 | 2343 | 2089 | 186 |

At its selected operating point, the LSTM produces 1,080 more false positives and 157 fewer false negatives than Logistic Regression. Its FPR is about 2.07 times higher. The LSTM nevertheless has better ranking metrics: PR-AUC is higher by 0.008238 and ROC-AUC by 0.035772.

This pattern is consistent with a model that ranks examples reasonably well but is operated at a very permissive point on its score curve. It does not prove that benign examples systematically receive higher LSTM scores because the score distributions are not available.

## Validation versus test behavior

The available training log contains validation PR-AUC, but not validation FPR, precision, recall, threshold, or score distributions:

- Best recorded validation PR-AUC: `0.805096` at epoch 9.
- Training continued to epoch 14 because the configured patience was 5.
- Validation PR-AUC declined after epoch 9, reaching `0.765906` at epoch 14.
- Training loss continued declining from `0.429194` at epoch 1 to `0.056918` at epoch 14.
- The saved checkpoint was selected at epoch 9.

This is evidence of a training/validation divergence after epoch 9 and is compatible with overfitting pressure. It is not a direct test-overfitting proof because validation and test score distributions were not saved. The test LSTM PR-AUC (`0.853651`) is higher than the best logged validation PR-AUC (`0.805096`), so the available aggregate values do not show simple ranking degradation from validation to test.

Logistic Regression has no recorded validation history beyond its final selected threshold, so a comparable validation-to-test calibration analysis is unavailable.

## Class-prior and temporal distribution shift

The split class proportions are:

| Split | Positive | Negative | Positive rate |
|---|---:|---:|---:|
| Train | 3,571 | 25,744 | 12.18% |
| Validation | 2,592 | 3,603 | 41.84% |
| Test | 1,763 | 4,432 | 28.46% |

Relative to training, validation has a `+29.66` percentage-point positive-rate shift and test has a `+16.28` percentage-point shift. Test is `13.38` percentage points less positive than validation. This makes a threshold selected for F1 on validation potentially too permissive for the less-positive test distribution, especially for a model whose score scale is shifted by class weighting.

The LSTM used training-only `pos_weight = 7.209185`, equal to training negatives divided by training positives. Logistic Regression used `class_weight="balanced"`, which applies the same training-prior balancing principle. These choices are appropriate for ranking/recall under imbalance, but neither output is guaranteed to be calibrated to the validation or test deployment prior.

The LSTM threshold of `0.0007676032` is particularly strong evidence that its output scale is not calibrated as an ordinary probability at the selected operating point. The artifact does not contain reliability curves, Brier score, expected calibration error, or raw score distributions, so the calibration diagnosis remains qualitative.

## Source-day dependence

Per-source-day model metrics cannot be computed because predictions are not paired with source-day/sample identifiers. The existing dataset does show substantial source-day regime dependence:

| Source day | Train positive rate | Validation positive rate | Test positive rate |
|---|---:|---:|---:|
| Friday-02-03-2018 | 34.65% | 55.26% | 100.00% |
| Friday-16-02-2018 | 0.87% | 44.25% | 0.00% |
| Friday-23-02-2018 | 16.06% | 58.71% | 0.00% |
| Thuesday-20-02-2018 | 3.19% | 54.79% | 3.61% |
| Thursday-01-03-2018 | 19.48% | 55.42% | 0.00% |
| Thursday-15-02-2018 | 0.00% | 28.26% | 27.47% |
| Thursday-22-02-2018 | 9.56% | 53.22% | 9.73% |
| Wednesday-14-02-2018 | 18.39% | 36.42% | 54.48% |
| Wednesday-21-02-2018 | 6.35% | 0.00% | 44.83% |
| Wednesday-28-02-2018 | 11.72% | 20.72% | 48.98% |

Several test source days are all-negative while another is all-positive. This confirms strong temporal/source composition shift, but aggregate metrics cannot tell whether the LSTM's extra false positives are concentrated in particular days.

## Diagnostic assessment

### Overfitting

**Some evidence, not conclusive.** Training loss keeps falling while validation PR-AUC peaks at epoch 9 and then fluctuates lower. Early stopping selected the peak checkpoint. There is no saved per-example validation/test output to determine whether overfitting specifically caused the FPR increase.

### Poor probability calibration

**Strongly plausible.** The LSTM's F1-selected threshold is `0.0007676032`, far below 0.5, and the class-weighted objective changes the score prior. Calibration metrics and score histograms are absent, so this cannot be quantified from the current artifacts.

### Threshold mismatch

**Strong evidence.** The LSTM's validation-selected threshold is extremely permissive relative to Logistic Regression, and validation has a much higher positive rate than test. The resulting test operating point favors recall and incurs 2,089 false positives.

### Temporal distribution shift

**Strong evidence.** Train, validation, and test positive rates differ substantially, and source-day rates vary from 0% to 100% in the test partitions.

### Source-day dependence

**Dataset-level evidence only.** Source-day composition is highly dependent, but model-specific dependence cannot be measured without per-sample predictions and source IDs.

### Class-weight-induced overprediction

**Plausible contributing factor.** The LSTM uses training-prior `pos_weight=7.209185` while validation and test have much higher positive rates than training. The weighting encourages recall, and combined with the very low selected threshold it is consistent with overprediction. It cannot be isolated from threshold/calibration effects using aggregate metrics.

## Conclusion

**D. Combination**

The strongest supported explanation is a combination of threshold/calibration behavior and temporal/class-prior distribution shift, with possible contribution from class weighting and post-epoch-9 overfitting pressure. The artifacts do not support a definitive claim that the LSTM architecture itself is the root cause, nor do they prove that benign examples systematically receive higher LSTM scores.

## Single most useful next experiment

Run one **read-only inference-and-calibration evaluation** using the already selected checkpoints and the unchanged validation/test samples, saving per-sample model probabilities together with split, source-day, and sample identifiers. Generate validation and test reliability/calibration metrics, score histograms by true class, fixed-threshold curves, and per-source-day confusion matrices. This directly distinguishes threshold/calibration failure from source-day generalization failure without changing the model, target, preprocessing, or splits.
