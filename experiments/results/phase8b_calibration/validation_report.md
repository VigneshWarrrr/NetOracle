# Phase 8B -- Attack-Risk Calibration

Overall status: **PASS**

Calibrators (Platt scaling, isotonic regression) fit on VALIDATION probabilities/labels only; applied (transform only) to TEST exactly once. Method selected per model via validation-only Brier comparison (see calibration_metrics.json -> calibration_selection.rule). This phase does not define, reselect, or apply any classification threshold -- Phase 8A's threshold and results are untouched.

## Results

| Model | Raw test Brier | Raw test ECE | Method | Calibrated test Brier | Calibrated test ECE | PR-AUC unchanged | ROC-AUC unchanged |
|---|---:|---:|---|---:|---:|---|---|
| Temporal Transformer | 0.0771 | 0.0741 | platt | 0.1037 | 0.1598 | True | True |
| Vector World Model (Run 1) | 0.1143 | 0.1121 | platt | 0.1365 | 0.1697 | True | True |

## Interpretation

- Temporal Transformer: raw test ECE=0.0741 (validation ECE=0.2100), raw test Brier=0.0771. Materially miscalibrated (test ECE > 0.05 threshold, stated explicitly, not fit to this result). Selected calibration method: platt (test Brier does not improve by -0.0266, test ECE does not improve by -0.0857). PR-AUC unchanged after calibration: True. ROC-AUC unchanged: True. Validation-to-test ECE gap: 0.1359 (NOT stable under the chronological prevalence shift, by a 0.05 ECE-gap threshold).
- Vector World Model (Run 1): raw test ECE=0.1121 (validation ECE=0.2170), raw test Brier=0.1143. Materially miscalibrated (test ECE > 0.05 threshold, stated explicitly, not fit to this result). Selected calibration method: platt (test Brier does not improve by -0.0222, test ECE does not improve by -0.0576). PR-AUC unchanged after calibration: True. ROC-AUC unchanged: True. Validation-to-test ECE gap: 0.1049 (NOT stable under the chronological prevalence shift, by a 0.05 ECE-gap threshold).

### Suitability for dashboard/SOC decision support

Calibrated probabilities may be shown as a supporting, approximate risk indicator, but should NOT be presented to a SOC analyst as a statistically guaranteed likelihood -- calibration here is fit on validation and only checked (not re-fit) on test, the ECE gap between validation and test reflects the same chronological prevalence shift documented in Phase 8A, and no confidence interval or statistical guarantee is computed anywhere in this stack. Ranking (PR-AUC/ROC-AUC, unaffected by calibration) remains the more defensible basis for prioritization than the raw calibrated probability value taken literally.

## Plots

- `plots/temporal_transformer_reliability_test.svg`
- `plots/temporal_transformer_probability_histogram_test.svg`
- `plots/vector_world_model_run1_reliability_test.svg`
- `plots/vector_world_model_run1_probability_histogram_test.svg`

- Calibrators fit on validation only: True
- Test labels never used for fitting: True
- PR-AUC/ROC-AUC unchanged after calibration (all models): True
- Frozen files (Phase 5/6B-ablation/8A) unchanged: True
- Deterministic repeated computation: True

