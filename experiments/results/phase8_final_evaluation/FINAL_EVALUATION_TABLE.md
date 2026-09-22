# Phase 8A: Shared Controlled-FPR Model Comparison

Post-hoc comparison over already-persisted, already-frozen validation/test probabilities. No model was loaded, retrained, or modified. Threshold selected on VALIDATION ONLY via `choose_threshold_recall_at_fpr` (target validation FPR <= 0.05), applied to TEST exactly once.

**The 5% constraint applies to VALIDATION only.** Actual test FPR (reported below) differs per model because validation and test have different class prevalence (see interpretation).

## A. Threshold-independent ranking metrics (already-existing, unaltered)

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8454 | 0.8495 |
| Weighted LSTM | 0.8537 | 0.8853 |
| Unweighted LSTM | 0.8576 | 0.8855 |
| Temporal Transformer | 0.8638 | 0.8763 |
| Vector World Model (Run 1) | 0.8487 | 0.8720 |

## B. Shared operating-point metrics (validation FPR <= 5% constraint, applied to test)

| Model | Test Precision | Test Recall | Test F1 | Test FPR (actual) | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8899 | 0.7379 | 0.8068 | 0.0363 | 1301 | 4271 | 161 | 462 |
| Weighted LSTM | 0.7344 | 0.7419 | 0.7381 | 0.1067 | 1308 | 3959 | 473 | 455 |
| Unweighted LSTM | 0.7292 | 0.7544 | 0.7416 | 0.1115 | 1330 | 3938 | 494 | 433 |
| Temporal Transformer | 0.9214 | 0.7516 | 0.8279 | 0.0255 | 1325 | 4319 | 113 | 438 |
| Vector World Model (Run 1) | 0.7546 | 0.7396 | 0.7471 | 0.0957 | 1304 | 4008 | 424 | 459 |

## Validation-side threshold detail

| Model | Threshold | Validation FPR | Validation Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.909587 | 0.0494 | 0.4502 |
| Weighted LSTM | 0.654246 | 0.0486 | 0.4811 |
| Unweighted LSTM | 0.084459 | 0.0500 | 0.4792 |
| Temporal Transformer | 0.34954 | 0.0494 | 0.4506 |
| Vector World Model (Run 1) | 0.335242 | 0.0497 | 0.4726 |

## Fairness checks

- Same threshold-selection rule used for all models: True
- Threshold selected using validation only: True
- Test labels never used to select threshold: True
- Test probabilities never used during threshold selection: True
- Sample counts match expected partitions: True
- Positive prevalence identical across all models per split: True
- Predictions align to the same samples: True
- Overall fairness status: **PASS**

## Vector World Model Run 1 -- integrity reference check

- Recomputation vs. documented approximate reference values: **PASS**
- Recomputation vs. Run 1's own stored `metrics.json` values (exact): **PASS**

## Interpretation (evidence, not exaggeration)

1. Best PR-AUC: **Temporal Transformer**
2. Best ROC-AUC: **Unweighted LSTM**
3. Best validation recall under the shared constraint: **Weighted LSTM**
4. Best test F1 under the shared constraint: **Temporal Transformer**
5. Lowest test FPR: **Temporal Transformer**
6. World Model vs. Transformer (same protocol): PR-AUC higher = False, ROC-AUC higher = False, test F1 higher = False, test FPR lower = False

**Claim:** "The World Model provides better early-warning forecasting at controlled false-positive rates than conventional baselines."

**Verdict: NOT SUPPORTED**

The World Model does not clearly outrank the Transformer on both threshold-independent ranking metrics under this comparison; the claim is not supported by this evidence.

Validation and test have different positive prevalence (validation ~41.8%, test ~28.5% positive, both fixed by the chronological Phase 3.5 split, not by this script). A threshold chosen to hit <=5% FPR on validation does NOT transfer to ~5% FPR on test for any of the five models -- every model's test FPR below is substantially different from the 5% validation constraint, confirming the constraint does not transfer across the prevalence shift. This is a property of the dataset split, identical for all five models, not a flaw in any one model.

No winner is declared on the basis of F1 alone; see the full metric set above before drawing conclusions.

