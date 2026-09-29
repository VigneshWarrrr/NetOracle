# Phase 9J: SIH Compliance + Scientific Gap Audit

Overall project status: **YELLOW**

Requirement matrix: 6 GREEN / 6 YELLOW / 3 RED (of 15)

## Central Finding: Dual, Unreconciled World Model Implementations

**The repository contains TWO divergent, unreconciled World Model implementations, and neither is currently wired into a working, checkpoint-backed demo.**

- Track A (audited): VectorWorldModel (experiments/phase6b_vector_world_model.py) -- Rigorously trained, evaluated under a shared fair protocol (Phase 8A), leakage-audited (Phase 3.5/world_model_dataset.validate_world_model_samples), calibration-audited (Phase 8B), explainability-audited (Phase 7C), MITRE-stage-audited (Phase 7B). Real checkpoints exist (experiments/results/phase6b_vector_world_model*/model/best_model.pt, confirmed present).
  Gap: Never imported or called by any Django-reachable code path (confirmed by repo-wide grep for VectorWorldModel: 16 hits, all under experiments/). It is not the model behind any live demo.
- Track B (unaudited demo): NetworkWorldModel (world_model/world_model.py) -- Backs a fully-built 5-tab Streamlit dashboard (world_model/src/streamlit.py, 954 lines: Real Traffic Forecast, Benchmark, Explainability, MITRE ATT&CK, Architecture tabs) with its own feature pipeline (world_model/feature_extraction.py, reads raw CIC-IDS-2018 CSVs directly, NOT data/windows/*.csv) and its own 7-class MITRE taxonomy (world_model/config.yaml: Benign, Reconnaissance, Initial Access, Execution, Lateral Movement, Command & Control, Exfiltration).
  Gap: No checkpoint exists anywhere in the repository (models/best_world_model.pth absent, confirmed by verify_repository_facts) -- the Streamlit app currently fails to load the model and shows an error. Its 'Benchmark' tab's hardcoded comparison numbers (F1=0.6941, AUC=0.8323 for 'World Model') are traced to world_model/evaluation_results.json, which used a NON-STANDARD split (files literally named Wednesday-28-INFIL-TRAIN.csv / -INFIL-TEST.csv that do not exist in data/windows/ or data/raw/ -- an ad-hoc, undocumented split of the single Infiltration day) and shows a scientifically suspicious pattern: precision=0, recall=0, F1=0 on validation at ALL 5 forecast horizons at the selected threshold, yet recall 0.79-0.85 on test -- a validation/test inversion consistent with leakage or an invalid evaluation methodology. These numbers MUST NOT be cited as evidence of anything.
- A separate, better-designed (but still unaudited) experiment also exists: world_model/zero_shot_results.json trains on 5 attack-free-of-Infiltration days, validates on a 6th, and tests on the ENTIRE held-out Wednesday-28-02-2018 Infiltration day (a genuine leave-one-attack-family-out design). Result: validation AUC 0.9999 collapses to test AUC 0.37-0.43 (worse than random) with <1.5% recall at every horizon on the wholly unseen attack family. This is a believable, useful PRELIMINARY negative signal for unseen-attack generalization (Step 5 of this audit), but was never run through the audited/leakage-checked Phase 3.5 dataset interface, so it is cited here only as motivating evidence for a properly audited repeat experiment (see roadmap P1), never as a validated finding.
- The Django dashboard (dashboard/views.py -> feature_engine.services.ingest_live_events -> run_pipeline.run_pipeline_events -> forecasting.mitre_mapping/attack_chain/victim + world_model.inference_service.AttackForecastService) is a THIRD code path, distinct from both tracks above. Its model loader (world_model/inference_service.py:92-94) looks for models/attack_forecaster.pt, which also does not exist, so it falls back to a 2-layer sigmoid heuristic (world_model/inference_service.py:13-24, 113-115) -- reporting model_source: 'heuristic_fallback'. Its MITRE/attack-chain/victim outputs come from forecasting.attack_chain.AttackChainBuilder and forecasting.victim.NextVictimForecaster, which are real, tested, production-reachable, but were DELIBERATELY EXCLUDED from Phase 7D's audited MITRE trajectory narrative (experiments/tests.py: test_attack_chain_builder_never_used verifies AttackChainBuilder.predict_next_stage() is never called by the audited research scripts) -- meaning the live dashboard's stage/chain output is a separate, less-rigorously-validated heuristic than what Phases 7B/7D actually validated.
- Three different MITRE taxonomies coexist: (1) PS requires 5 named stages (Reconnaissance, Initial Access, Lateral Movement, Command & Control, Exfiltration); (2) Phase 7B's audited 6 classes are BENIGN, CREDENTIAL_ACCESS, LATERAL_MOVEMENT, COMMAND_AND_CONTROL, IMPACT, INITIAL_ACCESS -- missing Reconnaissance and Exfiltration, includes two classes the PS doesn't name; (3) world_model/config.yaml (unaudited track) has 7 classes including Reconnaissance and Exfiltration but is untrained/unvalidated. No single audited taxonomy in this repository fully covers the PS's 5 named stages.

## SIH Requirement Matrix

See `requirement_matrix.csv` for the full table. Summary by status:

**GREEN (6)**: R3, R4, R9, R10, R13, R14
**YELLOW (6)**: R1, R2, R6, R7, R8, R15
**RED (3)**: R5, R11, R12

## Required vs. Novelty Classification

See `sih_gap_audit_report.json` (novelty_matrix) for the full table. Recommendations:

- **GNN (graph neural network over network state)**: C -- Nice-to-have, blocked on data (not buildable on currently valid inputs)
- **Temporal GNN**: C -- lower priority than plain GNN, same data blocker
- **Packet+flow fusion**: D -- Novelty theater / reject for now; already tested, RED verdict twice; do not rebuild until new data resolves the Phase 9I blockers
- **Temporal Transformer**: A -- Mandatory compliance, already satisfied
- **World Model (learned state-transition dynamics)**: A -- Mandatory compliance, already satisfied; INTEGRATION is the real remaining work
- **Uncertainty estimation**: B -- Strong differentiator
- **Multi-step (native per-step) attack-risk forecasting**: A -- Mandatory compliance gap, highest-priority fix
- **MITRE stage trajectory**: A -- Mandatory compliance, mostly satisfied, needs taxonomy reconciliation
- **Explainability (gradient attribution / SHAP-compatible)**: A -- Mandatory compliance, needs production wiring only
- **Counterfactual defense simulation**: B -- Strong differentiator IF scoped narrowly (see Step 8); otherwise reject the broad framing
- **Reinforcement learning**: D -- Novelty theater / reject
- **LLM chatbot**: D -- Novelty theater / reject
- **Blockchain**: D -- Novelty theater / reject based on PS text as fetched; FLAG FOR HUMAN VERIFICATION with mentors given the theme name, since SIH themes can bundle multiple problem statements under one category
- **GAN / synthetic data**: D -- Novelty theater / reject
- **Federated learning**: D -- Novelty theater / reject

## Top Scientific Gaps (ranked)

1. **No integrated, checkpoint-backed inference engine or demo (R11/R12)** -- The PS's core deliverables (infiltration prediction engine, demonstration interface) are unmet even though the underlying trained model exists -- this is the single biggest gap between 'we built something good' and 'we can show it'
2. **No audited unseen-attack-family generalization result (R5)** -- PS explicitly requires generalization beyond signature memorization; preliminary unaudited evidence suggests this may currently FAIL (near-random AUC on held-out Infiltration)
3. **No native per-step attack-risk forecast (R6)** -- PS wording implies a K-indexed probability series, not one whole-horizon scalar
4. **MITRE taxonomy doesn't fully cover the PS's 5 named stages (R7)** -- Direct, easily-checked PS wording match
5. **No calibrated/trustworthy uncertainty (compounds Phase 8B's finding)** -- SOC operators need to know how much to trust a probability number; Phase 8B already showed naive calibration fails under prevalence shift
6. **Explainability not wired to any reachable interface (R8)** -- PS calls this non-negotiable; the capability exists but judges cannot see it run
7. **Packet-level requirement vs. invalid current PCAP pairing (R2)** -- PS explicitly wants packet-level features; current evidence says this can't be validly combined with the flow dataset

## Unseen Attack Generalization Feasibility

Recommended protocol: Leave-one-attack-family-out, built on top of the EXISTING Phase 3.5 canonical windowed dataset interface (world_model_dataset.py), not a new pipeline

Can the dataset support this claim? PARTIALLY. The dataset supports building and running the experiment (attack families are cleanly day-bounded per the existing Phase 2 audit), but the SMALL NUMBER of distinct attack families limits how strong or general any resulting claim can be -- a single held-out family (e.g. Infiltration) is a real, useful data point, not proof of generalization to attack types outside CIC-IDS-2018 entirely.

## Native Per-Step Infiltration Probability Feasibility

- Existing labels support it: True
- Backbone reusable: Yes -- the existing rollout already produces 6 distinct latent states z(t+1)...z(t+6) (experiments/phase6b_vector_world_model.py: rollout list before mean-pooling). The AttackForecastHead currently DISCARDS this per-step structure by mean-pooling before the final linear layer -- a per-step head would apply the same small MLP to EACH z(t+k) independently instead of the pooled mean.
- 1 - P(BENIGN) derived from the MITRE stage head is NOT a substitute for this and must never be presented as native per-step attack probability -- it answers a different question (which stage is most likely) and was fit with a different, stage-classification loss, not an attack/no-attack loss.

## Uncertainty Recommendation

Primary: Conformal prediction. It requires no architecture change, no retraining, trivial compute cost, and directly targets the exact failure mode Phase 8B already documented (naive probability calibration degrading under prevalence shift) with a method that has formal coverage guarantees rather than just empirical hope.

Fallback: Monte Carlo dropout, as a cheap secondary signal (epistemic-uncertainty flag) if conformal prediction's coverage checks fail under the project's specific distribution shift.

## Counterfactual Defense Feasibility

SCIENTIFICALLY SAFE ONLY IN A NARROW FORM: (1) restrict interventions to feature-space edits explicitly identified by the explainability output (not arbitrary host/IP actions, which aren't representable anyway), (2) always label the output 'simulated trajectory under a hypothetical feature-state edit', never 'predicted effect of this defensive action', (3) report the nearest-neighbor plausibility check alongside every counterfactual result. If this narrow framing cannot be maintained in the demo UI, DO NOT build this capability -- the broader framing risks a serious, easily-critiqued overclaim.

## Final Architecture Proposal

```
INPUT (CSV/PCAP) [CURRENT: CSV only, GREEN]
  -> STATE REPRESENTATION (157-dim feature vector) [CURRENT, GREEN]
  -> TEMPORAL MODEL (self-attention over 6 observed states) [CURRENT, GREEN]
  -> WORLD MODEL (LatentTransition, recursive x6 rollout) [CURRENT, GREEN]
  -> K-STEP ROLLOUT (z(t+1)...z(t+6)) [CURRENT, GREEN]
  -> PER-STEP ATTACK RISK (new per-step head on existing rollout) [PLANNED, P1]
  -> MITRE STAGE TRAJECTORY (per-step, needs taxonomy fix) [CURRENT + PLANNED FIX, P1]
  -> UNCERTAINTY (conformal prediction wrapper) [PLANNED, P2]
  -> EXPLAINABILITY (gradient attribution, needs wiring) [CURRENT + PLANNED WIRING, P0]
  -> OPTIONAL COUNTERFACTUAL SIMULATION (narrow feature-space form only) [OPTIONAL FUTURE, P2/reject-if-broad]
  -> OFFLINE SOC INTERFACE (ONE interface, wired to the audited checkpoint) [PLANNED, P0]

```

Packet/graph branch: NOT included as a required production branch. Phase 9H/9I established the currently-available PCAP data cannot be validly paired with the flow dataset (RED, RED). It remains a documented, evidence-based OPTIONAL FUTURE branch, contingent on new data per Phase 9I's recommendation -- never presented as part of the current production architecture.

## Final Demo Requirements

User provides: A CIC-IDS-2018-format flow CSV (the one input type that is fully validated end-to-end today); Optionally, a raw PCAP file -- accepted for packet-level feature EXTRACTION and display only, explicitly NOT fed into the forecasting model (per the R2/Phase 9I finding), to avoid silently reintroducing the invalid pairing

| # | Display element | Status |
|---|---|---|
| 1 | current network state | SUPPORTED -- direct readout of the last observed 157-dim state |
| 2 | current risk | SUPPORTED -- current-window classification-style score, already validated (Phase 4/5/6B) |
| 3 | future risk trajectory | PARTIALLY SUPPORTED until the per-step head (P1) lands -- today only a single whole-horizon number exists |
| 4 | predicted attack stage | SUPPORTED -- per-step MITRE stage exists (Phase 7B/7D), with documented taxonomy-coverage caveats |
| 5 | stage progression | SUPPORTED -- six-step stage trajectory is genuine (Phase 7D) |
| 6 | uncertainty | NOT SUPPORTED until conformal prediction (P2) is built |
| 7 | top driving features | SUPPORTED once wired (Phase 7C attribution exists, just needs a UI hook -- P0) |
| 8 | flagged flows windows | SUPPORTED -- straightforward once the engine is wired (threshold already validated, Phase 8A) |
| 9 | optional counterfactual result | OPTIONAL, only if narrowly scoped per Step 8 -- otherwise omit entirely rather than oversell it |
| 10 | evidence provenance | SUPPORTED -- every phase's report already documents this discipline; the demo should carry it through (e.g. label the benchmark tab's source explicitly, unlike the current unaudited Streamlit numbers) |

## Claims Audit

See `claims_audit.csv` for the full table.

| Claim | Verdict |
|---|---|
| Predicts future attacks | PARTIALLY SUPPORTED |
| Forecasts attacker progression | PARTIALLY SUPPORTED |
| Predicts infiltration probability | PARTIALLY SUPPORTED |
| Generalizes to unseen attacks | NOT SUPPORTED |
| Uses packet-level information | NOT SUPPORTED |
| Uses a network graph | NOT SUPPORTED |
| Uses SHAP | NOT SUPPORTED |
| Learns attacker kill-chain progression | NOT SUPPORTED |
| Provides calibrated probabilities | NOT SUPPORTED |
| Supports counterfactual defense | NOT SUPPORTED |
| Outperforms baselines | PARTIALLY SUPPORTED |
| Works offline | SUPPORTED |

## Prioritized Roadmap

See `roadmap.csv` for the full table with purpose/evidence/dependencies/risk.

**P0**:
- Reconcile the two World Model implementations; retire or clearly quarantine the unaudited world_model/ Streamlit track
- Build ONE integrated inference engine that loads the actual audited VectorWorldModel checkpoint and calls the actual audited MITRE mapping + explainability code
- Wire Phase 7C's gradient-attribution explainability into the same integrated engine/demo

**P1**:
- Run a properly audited leave-one-attack-family-out generalization experiment
- Add a native per-step attack-probability head to the existing VectorWorldModel backbone
- Reconcile the MITRE stage taxonomy to cover all 5 PS-named stages

**P2**:
- Add conformal-prediction uncertainty wrapper
- Build a narrowly-scoped counterfactual feature-space simulation (only if the narrow framing from Step 8 can be maintained in the UI)

**P3**:
- Document the open-source license explicitly

**REJECT**:
- Packet+flow fusion / GNN on current PCAP data
- Reinforcement learning, LLM chatbot, GAN/synthetic data, federated learning, blockchain (pending PS verification)

## Experiment Specifications (P0/P1/P2)

### Leave-one-attack-family-out generalization (P1)
- Hypothesis: The VectorWorldModel architecture, trained without exposure to a given attack family, generalizes poorly to that family at test time (motivated by the preliminary, unaudited zero_shot_results.json signal)
- Acceptance criterion: Documented, reproducible result either way -- a negative result (poor generalization) is an acceptance criterion in itself, per this project's established scientific discipline
- Failure criterion: N/A -- both outcomes are informative; only a leakage-audit failure would count as an implementation failure
- Expected artifact: experiments/results/phaseXX_unseen_attack_generalization/ with metrics.json, leakage-audit report, and comparison against the Phase 6B baseline

### Native per-step attack-probability head (P1)
- Hypothesis: A per-step head applied independently to each rollout latent z(t+1)...z(t+6) can produce meaningfully differentiated per-step attack probabilities, rather than one pooled whole-horizon number
- Acceptance criterion: Per-step head achieves materially better per-step metrics than the broadcast baseline, especially at short horizons (t+1, t+2) where more precise timing should be easiest
- Failure criterion: Per-step head performs no better than the broadcast baseline at any horizon -- would suggest the whole-horizon pooling wasn't actually losing much information
- Expected artifact: experiments/results/phaseXX_per_step_risk/ with per-step metrics.json and a direct comparison table against Phase 6B's whole-horizon numbers

### MITRE taxonomy reconciliation (P1)
- Hypothesis: CIC-IDS-2018's attack labels can support a taxonomy mapping that covers Reconnaissance and Exfiltration (the two PS-named stages currently missing from Phase 7B's 6-class taxonomy)
- Acceptance criterion: All 5 PS-named stages have at least some validation/test examples under the revised mapping, OR an explicit, evidenced statement of which stages CIC-IDS-2018 genuinely cannot support
- Failure criterion: If CIC-IDS-2018's labels genuinely cannot support Reconnaissance/Exfiltration as distinct classes, document this as a dataset limitation rather than forcing an unsupported mapping
- Expected artifact: experiments/results/phaseXX_mitre_taxonomy_reconciliation/ with the revised mapping and its evidence

### Conformal-prediction uncertainty wrapper (P2)
- Hypothesis: Split conformal prediction, calibrated on a held-out calibration set, achieves its nominal coverage rate on the test set even under the prevalence shift Phase 8B documented
- Acceptance criterion: Empirical coverage on test matches the nominal target within a small, pre-specified tolerance
- Failure criterion: Coverage falls meaningfully short of nominal on test -- would indicate the exchangeability assumption is violated under the real distribution shift, and must be reported as such, not hidden
- Expected artifact: experiments/results/phaseXX_conformal_uncertainty/ with coverage-check plots/tables

## Repository Facts Verified Programmatically This Run

- Commit hash: `4e54c7533fc327331d1ea44a609a52a6010424e0`
- Audited-track checkpoints present: {'experiments/results/phase4_baseline/lstm/best_model.pt': True, 'experiments/results/phase5_temporal_transformer/transformer/best_model.pt': True, 'experiments/results/phase6b_vector_world_model/model/best_model.pt': True, 'experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt': True, 'experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt': True, 'experiments/results/phase9h_multimodal_baseline/variant_a_model/best_model.pt': True}
- Unaudited-track checkpoints confirmed absent: {'models/attack_forecaster.pt': True, 'models/best_world_model.pth': True}

## Recommended Next Phase

Phase 9K should implement ONLY the P0 items (dual-implementation reconciliation, integrated inference engine, explainability wiring) -- no new modeling, pure integration of already-validated pieces. The P1 scientific experiments (unseen-attack generalization, per-step risk head, MITRE taxonomy reconciliation) should each be their own subsequent phase, following the same audit-then-implement discipline established since Phase 9H.

STOP AFTER PHASE 9J. Do not begin 9K automatically.
