# Phase 9H: Multimodal State Fusion Baseline

Verdict: **RED**

## 1. Executive Summary

Variant A (flow-only control, 347,806 parameters) was trained fresh using the exact Phase 6B VectorWorldModel architecture/protocol, reaching test PR-AUC=0.8682, ROC-AUC=0.8775, F1=0.8439 at a validation-selected threshold targeting FPR<=0.05. Variants B, C, and D could NOT be constructed: an empirical cross-correlation search across 9 UTC-offset hypotheses found no credible timestamp correspondence (best |r|=0.0879 at +0h, indistinguishable from noise) between the Phase 9D/9F single-host PCAP data and the Phase 6B flow-window CSV, compounded by a fundamental population mismatch (2 hosts vs. whole-subnet flow aggregates). Per this phase's own hard constraint, both modalities were stopped rather than guessed.

## 2. Research Question

Does validated packet/event and temporal graph information add measurable predictive value beyond the existing flow-only Vector World Model, under strict temporal-leakage and alignment rules?

## 3. Exact A/B/C/D Experiment Definitions

- **A (flow-only control)**: the existing Phase 6B VectorWorldModel architecture and training protocol, retrained fresh on the identical Phase 3.5/6B dataset interface (`world_model_dataset.read_world_model_samples`), input X=[N,6,157]. TRAINED.
- **B (flow + packet/event)**: intended to add compact Phase 9G B2 event statistics aligned to the 6 observed timesteps. NOT CONSTRUCTED -- see Section 5.
- **C (flow + graph/window)**: intended to add compact Phase 9E 10-second graph statistics aligned to the 6 observed timesteps. NOT CONSTRUCTED -- see Section 5.
- **D (flow + event + graph/window)**: B and C combined. NOT CONSTRUCTED -- see Section 5.

## 4. Dataset and Split Definition

- CONFIRMED: identical Phase 3.5/6B interface reused unmodified (`world_model_dataset.read_world_model_samples`), `data/windows/*.csv`.
- CONFIRMED: counts train=29315, validation=6195, test=6195 (matches `EXPECTED_SPLIT_COUNTS` = {'train': 29315, 'validation': 6195, 'test': 6195}).
- CONFIRMED: dataset validation status = `PASS`.
- CONFIRMED: X=[N,6,157], Y=[N,6,157]; forecasting target = `future_attack_within_horizon` (any current_attack in t+1..t+6), horizon = 6 windows of 10s (unchanged from Phase 4/5/6B).

## 5. Temporal Alignment Methodology

For each candidate UTC-offset hypothesis (0-8 hours added to the CSV's naive local window_start), bucket both the Phase 9D+9F combined PCAP packet counts (true UTC, from scapy packet.time / pcap file semantics) and the Wednesday CSV's flow_count into aligned 10-second UTC buckets, then compute the Pearson correlation over the overlapping time range. A real, correct offset should produce a materially higher correlation than incorrect offsets, since both series would then be describing genuinely overlapping real-world activity.

| offset (hours) | overlap windows | Pearson r |
|---:|---:|---:|
| +0 | 155 | 0.0879 |
| +1 | 456 | 0.0766 |
| +2 | 789 | 0.0158 |
| +3 | 1090 | 0.0028 |
| +4 | 1393 | 0.0195 |
| +5 | 1687 | 0.0313 |
| +6 | 1987 | -0.0055 |
| +7 | 2293 | 0.0189 |
| +8 | 2591 | -0.0153 |

- DERIVED: best offset = +0h, |r| = 0.0879
- DECISION: **STOP** -- Best offset (+0h) only reaches |r|=0.0879 (floor is 0.3); this is not distinguishable from noise given n=155 overlapping 10s windows. No offset hypothesis produced a credible correlation peak.
- Even if a correct offset had been found, the Phase 9D/9F PCAPs cover only 2 specific hosts (172.31.69.22 and the capWIN-...-172.31.64.89 endpoint), while the CSV's flow_count/flow-level features aggregate ALL flows across the entire monitored subnet in each window. A 2-host packet/graph feature would therefore be a structurally different, far narrower quantity than what the flow-level window represents, independent of the timestamp question.

## 6. Leakage Safeguards

- Scaler (StandardScaler) fit on TRAIN X only; identical transform applied to X/Y across splits.
- pos_weight for the attack loss computed from TRAIN labels only.
- Threshold selected via `choose_threshold_recall_at_fpr` (Phase 8A's validation-only operating-point procedure) on VALIDATION probabilities only, at target FPR <= 0.05; frozen and applied exactly once to TEST.
- Test set never used to select the model, threshold, features, or preprocessing.
- `validate_world_model_samples` (Phase 6B's own independent audit) re-run and its status reported in Section 4 -- checks span-level split-boundary containment, 10s-contiguous alignment, and label recomputation, all unmodified from Phase 6B.
- No PCAP packet was aligned to any individual CSV row at any point (Section 5's method operates on 10-second AGGREGATE bucket counts only, and even that aggregate-level alignment was rejected).

## 7. Exact Event Feature List

NONE -- Variant B (flow + packet/event) was not constructed. Per the temporal alignment investigation in Section 5, no defensible correspondence between Phase 9G's event representation (built from 2 single-host PCAPs) and the Phase 6B flow windows could be established, so no event feature was selected or extracted for training.

## 8. Exact Graph/Window Feature List

NONE -- Variant C (flow + graph/window) was not constructed, for the same reason as Section 7.

## 9. Architecture Comparison

Only Variant A was constructed. Its architecture is the unmodified Phase 6B `VectorWorldModel` (StateEncoder -> TemporalContextEncoder -> LatentTransition x6 -> StateDecoder + AttackForecastHead), imported read-only.
- d_model=128, heads=4, layers=2, ff_dim=256, dropout=0.2

## 10. Parameter Counts

- A (flow-only): **347,806** parameters
- B, C, D: N/A -- not constructed

## 11. Hardware/GPU Verification

- CONFIRMED: CUDA available = `True`
- CONFIRMED: GPU name = `NVIDIA GeForce RTX 4060 Laptop GPU`
- CONFIRMED: first training batch device = `cuda:0`
- CONFIRMED: model parameters verified on CUDA (assertion passed during training)
- OBSERVED: peak GPU memory allocated = 38.2 MiB

## 12. Training Protocol

- Reused unmodified from Phase 6B: Adam optimizer, lr=0.001, batch_size=128, max_epochs=25, patience=5 (early stop on validation PR-AUC), loss = 1.0*MSE(state) + 1.0*BCEWithLogits(attack, pos_weight=train-derived), mixed precision (AMP) when CUDA available.
- Best epoch: 1; training duration: 25.10s; inference duration (validation+test): 0.459s
- Smoke test status: `PASS`

## 13. Validation Results

- Threshold selection: {'threshold': 0.7100254893302917, 'achieved_validation_fpr': 0.048570635581459895, 'achieved_validation_recall': 0.4185956790123457}
- PR-AUC=0.756635, ROC-AUC=0.775631, precision=0.861111, recall=0.418596, F1=0.563344, FPR=0.048571
- Brier=0.211051, ECE=0.158526

## 14. Test Results

- PR-AUC=0.868170, ROC-AUC=0.877507, precision=0.950284, recall=0.758934, F1=0.843898, FPR=0.015794
- Confusion matrix: {'tn': 4362, 'fp': 70, 'fn': 425, 'tp': 1338}
- Brier=0.129802, ECE=0.162204
- Bootstrap 95% CI (n=2000 resamples): {'n_iterations_requested': 2000, 'n_iterations_used': 2000, 'pr_auc': {'mean': 0.8682638032650718, 'ci_lower_2.5pct': 0.8550466390656092, 'ci_upper_97.5pct': 0.8810547726291467}, 'roc_auc': {'mean': 0.8775050763682, 'ci_lower_2.5pct': 0.864568526820369, 'ci_upper_97.5pct': 0.8894551615057091}, 'f1': {'mean': 0.843946722706847, 'ci_lower_2.5pct': 0.8297265035764231, 'ci_upper_97.5pct': 0.8573228847986358}, 'recall': {'mean': 0.7590153544144027, 'ci_lower_2.5pct': 0.7379767123634797, 'ci_upper_97.5pct': 0.7786711272713989}}

## 15. Ablation Deltas

N/A. B-A, C-A, D-A cannot be computed because B, C, and D were never constructed (Section 5). This is itself the headline finding of this phase: the ablation could not be run as originally specified because the additional modalities could not be safely aligned to the flow-level dataset, not because a trained model underperformed.

## 16. Complexity/Cost Comparison

- A: 347,806 params, 25.10s training, 0.459s validation+test inference.
- B/C/D: no cost incurred (not constructed).

## 17. Statistical Interpretation

The central research question -- whether packet/event/graph information adds measurable predictive value beyond the flow-only representation -- could not be tested empirically in this phase, because the additional modalities could not be safely constructed as model inputs. This is a distinct and, in one sense, a stronger negative finding than 'we tested it and found no improvement': it means the specific data available (2 single-host PCAP captures from one day) does not currently support even attempting the ablation without either guessing a timezone offset with no statistical support, or fabricating a correspondence between per-host packet traces and whole-subnet flow aggregates. Per the phase's own scientific-interpretation guidance, this is closest to outcome 5 (mixed/small evidence, prefer a conservative verdict) taken to its logical extreme: there is no evidence of predictive value because no valid experiment could be run, which must not be mistaken for evidence AGAINST predictive value in a fairer future experiment with better-aligned data.

## 18. Limitations

- Only Variant A could be trained; the core B vs C vs D ablation this phase was designed to answer was not executed.
- The temporal-alignment investigation is itself limited: only Pearson correlation over 10-second aggregate buckets was tested, across a bounded set of 9 whole-hour offset hypotheses (0-8h); a non-integer-hour offset, or a genuinely weak-but-real correlation obscured by the 2-host-vs-whole-subnet volume mismatch, cannot be ruled out -- only that THIS test found no usable signal.
- No documented timezone metadata exists anywhere in this repository for the CICFlowMeter-derived Timestamp/window_start column; this gap in the source dataset's own documentation, not a limitation introduced by this phase, is the root cause of the alignment difficulty.
- Variant A's numbers are a fresh, independently retrained run (new seed-consistent training), not a reuse of the frozen Phase 6B checkpoint -- expect small numeric differences from Phase 6B's own persisted metrics.json even though the code and protocol are unmodified.
- Bootstrap confidence intervals use standard resampling with replacement over the test set only; they characterize sampling variability of Variant A's OWN metrics and say nothing about variants that were not built.

## 19. Novelty Interpretation

**A. Existing/known technique:**
- Flow-level sequence-to-sequence forecasting with a Transformer/latent-rollout architecture (Phase 6B's own VectorWorldModel) is a known technique class, reused unmodified here.
- Timestamp-based dataset alignment/leakage auditing (split-boundary containment, cross-split duplicate detection) is standard ML engineering practice, not novel to this phase.

**B. Potential NetOracle system differentiator:**
- The intended multimodal fusion of flow state + packet/event + graph/window state into one forecasting model remains NetOracle's stated architectural differentiator -- but this phase demonstrates it is NOT YET achievable with the currently available packet-capture coverage (2 hosts, 1 day), which is itself useful information for scoping what data would be needed.

**C. What this experiment actually demonstrates:**
- A rigorous, transparent, and negative result: the specific PCAP artifacts available from Phases 9D/9F cannot be safely temporally aligned to the Phase 6B flow-level dataset.
- A working, GPU-verified, leakage-audited retraining of the flow-only baseline under the exact existing protocol, usable as a stable reference point for any FUTURE attempt at this ablation once better-aligned or broader packet-capture coverage exists.

**D. What remains unverified:**
- Whether packet/event or graph/window information would add predictive value GIVEN properly aligned, broader-coverage capture data -- entirely untested here.
- Whether a learned graph or event encoder (GNN, sequence model, or otherwise) would outperform simple aggregate statistics -- moot until alignment is solved.

## 20. Recommendation for Phase 10

Before attempting this ablation again: (1) establish authoritative timezone metadata for the CICFlowMeter Timestamp column (e.g. by locating original dataset documentation, or by capturing a small new PCAP alongside a fresh CICFlowMeter run with known clock settings), and/or (2) obtain PCAP coverage for a larger fraction of hosts active in a given window (not just 1-2 endpoints), so that packet/graph features approximate the same population the flow-level window aggregates describe. Only once both are resolved should Phase 9H's original A/B/C/D ablation be re-attempted; a learned graph/event encoder or GNN is NOT scientifically justified as a next step from this phase's evidence, because no modality-vs-baseline comparison was possible at all.

## 21. Final Verdict: RED

STOP AFTER PHASE 9H.
