# Phase 9N: Native Per-Step Attack-Risk Forecasting

Status: **RED**

The per-step TARGET and LEAKAGE-FREE construction ARE defensible (Steps 1-3 passed cleanly, reusing Phase 7B's already-validated per-step labels), and the architecture change is clean (frozen backbone, verified byte-identical before/after training; frozen whole-horizon metrics reproduced exactly before and after). But the trained head does not demonstrate forecasting value: mean PR-AUC 0.8397 is beaten by the trivial current-window-persistence baseline (0.9253) at EVERY single one of the 6 horizon steps, not just on average, and within-sample predicted-vs-true trajectory correlation is near-zero (-0.0552). This phase therefore cannot claim a defensible native per-step forecasting capability from the tested approach (frozen-backbone + shared linear head) -- the failure is best explained by state-representation information loss in the frozen backbone's latent rollout, not by dataset unsupportability of per-step targets themselves (see Section 16).

## 1. Executive Summary

Phase 9N added a NEW, non-authoritative PerStepAttackRiskHead on top of the frozen Phase 6B Run-1 backbone (never modified -- verified byte-identical before/after training) to test whether NetOracle can produce a genuine [risk(t+1)...risk(t+6)] trajectory rather than one whole-horizon scalar. Per-step binary targets were derived by reusing Phase 7B's already-validated per-step stage targets (y_k = stage != BENIGN), avoiding any new target-construction leakage surface. Leakage audit: PASS. Preflight/postflight frozen-metric reproduction: PASS/PASS. Learned-model mean PR-AUC across the 6 steps: 0.8397 vs persistence-baseline 0.9253 and whole-horizon-broadcast-baseline 0.8364.

## 2. Existing Whole-Horizon Limitation

The authoritative World Model (Phase 6B Run-1 + Phase 7B/9K) produces exactly one scalar: P(attack somewhere in t+1..t+6). It has no native per-step head; Phase 9K/9L explicitly refused to fabricate per-step values by broadcasting this scalar. This phase asks whether a genuine per-step head can be added.

## 3. Per-Step Target Definition

`y_k = 1 if per_step_stage_name[:, k] != "BENIGN" else 0` for k=0..5 (t+1..t+6), derived by reusing (unmodified, read-only) Phase 7B's already-validated `phase7b_stage_targets.read_stage_targets`. This is a GENERIC any-covered-attack target, not Infiltration-specific -- see Section 15.

## 4. Target Construction

Leakage audit status: **PASS**
- train: n=29315, positive per step (t+1..t+6)=[2904, 2902, 2900, 2899, 2898, 2897]
- validation: n=6195, positive per step (t+1..t+6)=[2054, 2059, 2064, 2070, 2075, 2080]
- test: n=6195, positive per step (t+1..t+6)=[1643, 1639, 1636, 1633, 1629, 1625]

## 5. Leakage Audit

- none

Stress checks:
```json
{
  "train_current_vs_t1_identical_fraction": 0.9893910967081698,
  "validation_current_vs_t1_identical_fraction": 0.9491525423728814,
  "test_current_vs_t1_identical_fraction": 0.9843422114608555,
  "terminal_windows_excluded_by_construction": true,
  "scaler_path": "C:\\AKSHAY\\Akshay\\SIH FOLDER MAIN\\NetOracle\\experiments\\results\\phase6b_vector_world_model_ablation\\run1_existing_scaling\\model\\scaler.joblib",
  "scaler_reused_unmodified_from_run1": true,
  "scaler_sha256": "b3a0cea7f9d2d0fc59de163c8768116cbd95abbe3b30c127c44a2322f0c24881",
  "all_checks_pass": true
}
```

## 6. Architecture

`VectorWorldModelWithPerStepRiskHead(VectorWorldModel)` -- frozen Phase 6B Run-1 backbone (state_encoder, temporal_context, transition x6, state_decoder, attack_head, all inherited unmodified) + new `PerStepAttackRiskHead` (shared Linear->GELU->Dropout->Linear(1) applied independently to each of the 6 predicted latents z_future[:,k,:]), producing `[B,6]` risk logits. Only `per_step_risk_head.*` parameters are trainable (frozen-backbone invariant verified: True).

## 7. Loss Function

`mean_k BCEWithLogitsLoss(logit_k, y_k)` with a per-step `pos_weight` tensor computed from TRAIN split labels only: `[9.094696998596191, 9.101654052734375, 9.108620643615723, 9.112107276916504, 9.115596771240234, 9.119089126586914]`. No state-reconstruction term is included (backbone is frozen; state_loss/attack_loss are computed for logging only and excluded from backward()) -- this is the smaller, more defensible design given the already-audited Phase 7B precedent, so no lambda hyperparameter search was needed.

## 8. Baselines

- Baseline A (current-window persistence): broadcasts the CURRENT window's own attack status.
- Baseline B (whole-horizon broadcast): copies the existing scalar P(attack in horizon) across all 6 steps -- included ONLY to demonstrate why this is not a substitute for genuine per-step prediction, never as a legitimate solution.

| model | mean PR-AUC | mean ROC-AUC | mean F1 |
|---|---:|---:|---:|
| learned_model | 0.8396716439957784 | 0.875076406420544 | 0.667669135800332 |
| baseline_a_persistence | 0.9252555549368808 | 0.9714062763474117 | 0.9563431977779304 |
| baseline_b_whole_horizon_broadcast | 0.8363784526003154 | 0.8681274680127281 | 0.6793808052121334 |

## 9. Training Protocol

- Seed: 42; Device: cuda:0
- Parameter count (trainable): 8321
- Best epoch: 1; duration: 21.03s
- Checkpoint sha256: cf02d6f2444aca40...
- Scaler sha256 (reused Run-1 scaler): b3a0cea7f9d2d0fc...
- Preflight (reproduce frozen Run-1 metrics before training): **PASS**

## 10. Per-Step Results (PRIMARY -- learned model, test split)

| step | n | prevalence | threshold | precision | recall | f1 | pr_auc | roc_auc | fpr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t+1 | 6195 | 0.2652 | 0.0289 | 0.486 | 0.786 | 0.601 | 0.831 | 0.8728329805202237 | 0.301 |
| t+2 | 6195 | 0.2646 | 0.0390 | 0.535 | 0.780 | 0.635 | 0.838 | 0.8741034893007953 | 0.244 |
| t+3 | 6195 | 0.2641 | 0.0455 | 0.610 | 0.778 | 0.684 | 0.842 | 0.8752959700873792 | 0.178 |
| t+4 | 6195 | 0.2636 | 0.0488 | 0.628 | 0.783 | 0.697 | 0.843 | 0.8758876477130898 | 0.166 |
| t+5 | 6195 | 0.2630 | 0.0499 | 0.626 | 0.784 | 0.696 | 0.843 | 0.8762599129283704 | 0.167 |
| t+6 | 6195 | 0.2623 | 0.0524 | 0.623 | 0.783 | 0.694 | 0.842 | 0.8760784379734052 | 0.168 |

## 11. Horizon Degradation

```json
{
  "pr_auc_step1": 0.830667971276842,
  "pr_auc_step6": 0.8415258863746643,
  "pr_auc_monotonic_non_increasing": false
}
```

## 12. Trajectory Diagnostics

```json
{
  "mean_absolute_step_to_step_change": 0.002371380804106593,
  "fraction_of_sequences_non_constant": 0.9657788539144472,
  "mean_within_sample_predicted_vs_target_trajectory_correlation": -0.05519856232326369,
  "n_samples_with_defined_correlation": 280,
  "note": "Diagnostic only -- not a substitute for the per-step PR-AUC/ROC-AUC/recall metrics above. No lead-time claim is made; lead time was not explicitly evaluated in this phase."
}
```

## 13. Whole-Horizon Comparison

```json
{
  "assumption": "independence across the 6 future steps (explicitly NOT justified as the true joint probability -- future windows are temporally correlated in this dataset)",
  "formula": "1 - prod_k(1 - p_k)",
  "mean_absolute_error_vs_existing_whole_horizon_attack_head": 0.06097659468650818,
  "pearson_correlation_vs_existing_whole_horizon_attack_head": 0.9663577292960697,
  "conclusion": "Reported as a diagnostic comparison only. The authoritative whole-horizon checkpoint and its output are NOT replaced or modified by this derived aggregation."
}
```

## 14. Probability / Calibration Interpretation

Recommended wording: **"predicted attack-risk score in [0,1]"** (expected calibration error 0.1057).

## 15. SIH Requirement Mapping

```json
{
  "delivered_in_this_phase": "A -- generic future attack-risk per step (any covered MITRE stage / any current_attack in that specific future window)",
  "NOT_delivered": "B -- Infiltration-specific per-step probability (would require a separately trained/evaluated model on the Infiltration-only target; not attempted here per 'smallest defensible experiment')",
  "also_not_delivered": "C -- this phase's output is per-step, distinct from the existing whole-horizon scalar (still produced, unchanged, by the frozen backbone's own attack_head)",
  "per_step_generic_attack_prevalence_test": [
    0.2652138821630347,
    0.2645682001614205,
    0.26408393866020985,
    0.2635996771589992,
    0.262953995157385,
    0.2623083131557708
  ],
  "per_step_infiltration_prevalence_test": [
    0.05036319612590799,
    0.05020177562550444,
    0.05004035512510089,
    0.049878934624697335,
    0.04971751412429379,
    0.049556093623890234
  ],
  "infiltration_fraction_of_generic_positives_per_step": [
    0.1898965307364577,
    0.18974984746796827,
    0.18948655256723718,
    0.18922229026331905,
    0.18907305095150398,
    0.18892307692307692
  ],
  "conclusion": "The trained per-step head answers 'what is the risk of ANY covered-MITRE-stage attack at t+k', not 'what is the risk of INFILTRATION specifically at t+k'. Where the PS wording 'infiltration probability over the next K windows' is read literally (infiltration-specific), this phase's model does NOT satisfy it; where it is read as a stand-in for generic 'attack-risk time series', this phase directly satisfies the SHAPE and TEMPORAL semantics of the requirement (a genuine, non-broadcast [B,6] risk trajectory) with the caveat that the underlying risk is generic, not infiltration-specific."
}
```

## 16. Failure Analysis

Learned model mean PR-AUC=0.8397 vs Baseline A (persistence) mean PR-AUC=0.9253 vs Baseline B (whole-horizon broadcast) mean PR-AUC=0.8364. Persistence beats the learned model at EVERY single horizon step (per-step PR-AUC gap, learned minus persistence: [-0.1188, -0.0878, -0.0808, -0.0763, -0.081, -0.0688]), not just on average -- this is a uniform, not step-dependent, shortfall. The learned model only marginally beats the whole-horizon-broadcast non-solution baseline (yes, by 0.0033 mean PR-AUC), meaning most of what the new head captures is close to the existing scalar, not genuine per-step structure. Trajectory diagnostics reinforce this: mean step-to-step change in predicted risk is tiny (0.00237) and the within-sample correlation between the predicted and true per-step trajectories is near-zero/slightly negative (-0.0552) -- the head has not learned to track the SHAPE of genuine per-step risk change, only a roughly-average risk level. Positive-class prevalence per step (train): [0.09906191378831863, 0.0989936888217926, 0.09892546385526657, 0.09889135509729385, 0.09885723888874054, 0.09882313013076782] -- prevalence is nearly flat across steps (attacks in this dataset tend to be sustained over many consecutive 10s windows), which is exactly why a trivial persistence heuristic is so strong here: recent attack status is highly autocorrelated with near-future attack status, and the frozen backbone's latent rollout (optimized originally for state reconstruction + one whole-horizon scalar, not for preserving this autocorrelation signal per step) does not expose that information as usefully to a shallow linear head as the raw current-window flag does directly. This points to STATE REPRESENTATION / frozen-backbone information loss (the latent rollout smooths away the sharp, highly-autocorrelated attack signal) as the primary limiting factor, not target sparsity (prevalence is high, ~26% per step) or class imbalance (pos_weight correctly compensates) -- a jointly fine-tuned backbone or a head with direct access to the current-window attack flag would be the natural next experiment, but is out of scope here (frozen-backbone-only was this phase's deliberately minimal, defensible design).

## 17. Limitations

- Target is GENERIC attack-risk (any covered MITRE stage), not Infiltration-specific -- see Section 15.
- Backbone is frozen (only the new head is trained); a jointly fine-tuned backbone was not attempted (smallest defensible experiment).
- Per-step thresholds are selected independently per step on validation; no joint multi-step calibration was performed.
- No lead-time evaluation was performed; no 'early warning' claim is made.
- This is one seed; no multi-seed variance estimate.
- New checkpoint exists ONLY under C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase9n_per_step_risk -- the authoritative Phase 9K/9L production checkpoint and Django inference path are unmodified.

## 18. Claim-Safety Statement

None of the following are claimed anywhere in this report or its artifacts: "predicts the exact next attack", "predicts attacker actions", "causal attacker progression", "infiltration probability" (the delivered target is generic attack-risk, stated explicitly in Section 15), "calibrated probability" without calibration evidence (see Section 14's wording decision), "early warning" (no lead-time evaluation was performed), "generalizes to unseen attacks" (out of scope, see Phase 9M).

## 19. Final Verdict

**RED**

The per-step TARGET and LEAKAGE-FREE construction ARE defensible (Steps 1-3 passed cleanly, reusing Phase 7B's already-validated per-step labels), and the architecture change is clean (frozen backbone, verified byte-identical before/after training; frozen whole-horizon metrics reproduced exactly before and after). But the trained head does not demonstrate forecasting value: mean PR-AUC 0.8397 is beaten by the trivial current-window-persistence baseline (0.9253) at EVERY single one of the 6 horizon steps, not just on average, and within-sample predicted-vs-true trajectory correlation is near-zero (-0.0552). This phase therefore cannot claim a defensible native per-step forecasting capability from the tested approach (frozen-backbone + shared linear head) -- the failure is best explained by state-representation information loss in the frozen backbone's latent rollout, not by dataset unsupportability of per-step targets themselves (see Section 16).

STOP AFTER PHASE 9N. Do NOT begin uncertainty, counterfactual defense, GNN, packet fusion, or further novelty work.
