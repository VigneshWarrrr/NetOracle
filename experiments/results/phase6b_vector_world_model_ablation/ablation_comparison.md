# Phase 6B Controlled Improvement Ablation

Architecture is unchanged (VectorWorldModel imported unmodified). Two controlled runs differ ONLY in feature preprocessing; both use the same improved threshold-selection rule (maximize recall subject to validation FPR <= 5%, validation-only) and the same relaxed early-stopping budget (max_epochs=60, patience=10). The existing F1-max validation threshold is reported for comparison only and is not the operating point. Test labels were never used for threshold selection; the selected threshold is applied to test exactly once (no test sweep).

## Run summary

| Run | Parameters | Best epoch | Epochs run | Training time (s) | Primary threshold | Val FPR @ threshold | Val recall @ threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1_existing_scaling_improved_threshold | 347806 | 22 | 32 | 148.585 | 0.335242 | 0.049681 | 0.472608 |
| run2_robust_pathological_scaling_improved_threshold | 347806 | 27 | 37 | 170.286 | 0.597667 | 0.048571 | 0.493056 |

## Validation / test metrics at the PRIMARY threshold (recall @ FPR<=5%)

| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run1_existing_scaling_improved_threshold (validation) | 0.872507 | 0.472608 | 0.613113 | 0.825969 | 0.852195 | 0.049681 | 1225 | 3424 | 179 | 1367 |
| run1_existing_scaling_improved_threshold (test) | 0.754630 | 0.739648 | 0.747064 | 0.848722 | 0.871996 | 0.095668 | 1304 | 4008 | 424 | 459 |
| run2_robust_pathological_scaling_improved_threshold (validation) | 0.879560 | 0.493056 | 0.631891 | 0.804161 | 0.820883 | 0.048571 | 1278 | 3428 | 175 | 1314 |
| run2_robust_pathological_scaling_improved_threshold (test) | 0.723659 | 0.650596 | 0.685185 | 0.802646 | 0.833935 | 0.098827 | 1147 | 3994 | 438 | 616 |
| phase5_temporal_transformer (test, its own selected threshold) | 0.617992 | 0.806580 | 0.699803 | 0.863815 | 0.876298 | 0.198330 | 1422 | 3553 | 879 | 341 |

## Reference only: existing F1-max validation threshold (NOT the operating point)

| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run1_existing_scaling_improved_threshold (validation) | 0.638968 | 0.860340 | 0.733311 | 0.825969 | 0.852195 | 0.349709 | 2230 | 2343 | 1260 | 362 |
| run1_existing_scaling_improved_threshold (test) | 0.430663 | 0.817357 | 0.564103 | 0.848722 | 0.871996 | 0.429829 | 1441 | 2527 | 1905 | 322 |
| run2_robust_pathological_scaling_improved_threshold (validation) | 0.588720 | 0.910108 | 0.714957 | 0.804161 | 0.820883 | 0.457397 | 2359 | 1955 | 1648 | 233 |
| run2_robust_pathological_scaling_improved_threshold (test) | 0.389546 | 0.841180 | 0.532496 | 0.802646 | 0.833935 | 0.524368 | 1483 | 2108 | 2324 | 280 |

## Per-horizon state prediction error (test, standardized feature scale -- NOTE: run 2's scale for the 4 pathological features differs from run 1's by construction, so raw MSE magnitudes are not directly comparable feature-for-feature between runs; the reduction in total test MSE is the intended effect being measured)

| Horizon | Run 1 MSE | Run 1 MAE | Run 2 MSE | Run 2 MAE |
|---|---:|---:|---:|---:|
| t+1 | 2133.208496 | 1.422490 | 18.515575 | 0.432570 |
| t+2 | 2134.165771 | 1.420788 | 18.520626 | 0.427180 |
| t+3 | 2136.859619 | 1.423689 | 18.527905 | 0.425310 |
| t+4 | 2140.120850 | 1.428593 | 18.554844 | 0.426430 |
| t+5 | 2140.458496 | 1.431466 | 18.580826 | 0.429295 |
| t+6 | 2140.666504 | 1.432818 | 18.584560 | 0.430080 |

## Training loss components at the best (checkpointed) epoch

| Run | Best epoch | Train state loss (MSE) | Train attack loss (BCE) | Val state loss (MSE) |
|---|---:|---:|---:|---:|
| run1_existing_scaling_improved_threshold | 22 | 0.419862 | 0.140124 | 1.824706 |
| run2_robust_pathological_scaling_improved_threshold | 27 | 0.405088 | 0.117884 | 1.841695 |

No test-label information was used to select any threshold. No architecture change was made in either run (both import VectorWorldModel unmodified from phase6b_vector_world_model.py). Improvement claims should be read from the primary-threshold table and the ranking metrics (PR-AUC/ROC-AUC), not from any test sweep.

