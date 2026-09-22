# Phase 9K: Authoritative World Model Reconciliation + Inference Engine

Status: **GREEN**

## 1. Executive Summary

Phase 9K reconciled the two World Model implementations discovered in Phase 9J, designating the audited experiments/phase6b_vector_world_model.py + phase7b_mitre_stage_head.py + Run1 checkpoint as the single authoritative model, and built ONE inference engine (experiments/inference_engine.py) exposing its rollout, whole-horizon attack probability, 6-step MITRE stage trajectory, and gradient-based explainability through a single documented contract with full provenance. Deterministic validation against frozen Phase 6B Run1 and Phase 7B test-set predictions passed on all checked samples (attack-probability equivalence within 1e-3, exact stage-name match, deterministic repeated inference, correct [6,157] attribution shape). No training, no threshold changes, no checkpoint modification occurred. The engine is not yet wired into the Django dashboard -- that remains explicit future work.

## 2. Track A vs Track B Reconciliation

See `track_reconciliation.md` for the full decision document. Summary: Track A (experiments/phase6b_vector_world_model.py + phase7b_mitre_stage_head.py) is authoritative; Track B (world_model/world_model.py) is quarantined by documentation, not deleted, and was confirmed to have no Django-reachable import path.

## 3. Authoritative Model Decision

`VectorWorldModelWithStageHead` (experiments/phase7b_mitre_stage_head.py), backbone from experiments/phase6b_vector_world_model.py.

## 4. Exact Checkpoint Used

- `phase6b_run1_checkpoint`: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase6b_vector_world_model_ablation\run1_existing_scaling\model\best_model.pt` (sha256 `6374a9c47d722215...`)
- `phase6b_run1_scaler`: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase6b_vector_world_model_ablation\run1_existing_scaling\model\scaler.joblib` (sha256 `b3a0cea7f9d2d0fc...`)
- `phase7b_stage_head_checkpoint`: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase7b_mitre_stage_head\model\best_stage_head.pt` (sha256 `f9d16f1943aeeed4...`)
- `phase7c_shap_explainer_module`: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\explainability\shap_explainer.py` (sha256 `7045f572bd9922c4...`)

## 5. Inference Contract

See `inference_contract.json` for the full schema. Top-level keys: input_metadata, current_state, future_state_rollout, whole_horizon_attack_probability, mitre_stage_trajectory, explanations, provenance.

## 6. Preprocessing Contract

- Feature count: 157 (world_model_dataset.py canonical order)
- Scaler: `C:\AKSHAY\Akshay\SIH FOLDER MAIN\NetOracle\experiments\results\phase6b_vector_world_model_ablation\run1_existing_scaling\model\scaler.joblib` (loaded, NEVER refit)
- Feature schema sha256: `f5117fda8518c9e3...`

## 7. World-Model Outputs

- probability of attack somewhere within the forecast horizon (t+1..t+6) -- a single scalar, NOT a per-step P(t+1), P(t+2), ... series. The model has no native per-step attack-probability head; fabricating one from this scalar (e.g. by broadcasting or by 1-P(BENIGN) from the stage head) is explicitly out of scope for this phase and must never be presented as native per-step risk.

## 8. MITRE Outputs

- MITRE-stage classification derived from a reasoned mapping of CIC-IDS-2018 dataset attack labels (forecasting/mitre_mapping.py), NOT authoritative MITRE ATT&CK ground truth and NOT causal attacker kill-chain inference. Only 6 stage classes have real coverage in this dataset: ['BENIGN', 'INITIAL_ACCESS', 'CREDENTIAL_ACCESS', 'LATERAL_MOVEMENT', 'COMMAND_AND_CONTROL', 'IMPACT']. INITIAL_ACCESS has zero validation/test examples in the established Phase 3.5 split (see Phase 7B metrics.json). Stages not represented in the dataset at all: ['RECONNAISSANCE', 'EXECUTION', 'PERSISTENCE', 'PRIVILEGE_ESCALATION', 'DEFENSE_EVASION', 'DISCOVERY', 'COLLECTION', 'EXFILTRATION'].

## 9. Explainability Integration

- post-hoc gradient-based feature attribution with SHAP-compatible explainability infrastructure (explainability/shap_explainer.py); actual method used is reported per-call, never assumed
- Never claims SHAP; reports the actual method used per call (see explanations.future_attack_risk_prediction.explanation_method in a real result)

## 10. Provenance

Every prediction includes model identifier, checkpoint path+hash, scaler path+hash, feature-schema hash, code commit hash, inference timestamp/duration, device, and prediction semantics text (see `inference_contract.json`).

## 11. Offline Input Contract

CSV/window -> world_model_dataset.read_world_model_samples() -> [6,157] raw feature array -> NetOracleInferenceEngine.predict(). PCAP input is explicitly NOT supported by this engine (Phase 9H/9I: RED) -- no PCAP-to-flow extractor was built in this phase.

## 12. UI Integration Status

NOT WIRED to the Django dashboard in this phase. Per Step 11's instruction to prefer a thin integration layer over risky rewrites: experiments/inference_engine.py is a standalone, fully importable module with zero Django dependency, ready to be called from world_model/inference_service.py or a new dashboard view in a future phase. The remaining UI work is: (1) replace/augment AttackForecastService's heuristic fallback with a call into NetOracleInferenceEngine.predict(), (2) map its structured output onto the existing dashboard template fields, (3) handle the CSV/window input contract from an uploaded file. None of this was done in Phase 9K to avoid risky architectural changes to the live Django app in an integration-only phase.

## 13. Deterministic Equivalence Results

- Samples checked: 5
- Run1 attack-probability equivalence (tolerance 0.001): True
- Phase 7B attack-probability equivalence: True
- Phase 7B stage-name exact match: True
- Repeated-inference determinism (tolerance 1e-06): True
- Attribution shape [6,F] for every sample: True
- **Overall: True**

| # | source_file | window_start | engine_prob | frozen_run1_prob | frozen_7b_prob | stages_match |
|---|---|---|---:|---:|---:|---|
| 0 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:12:50 | 0.999852 | 0.99985229969 | 0.99985229969 | True |
| 1 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:00 | 0.999850 | 0.999849915504 | 0.999849915504 | True |
| 2 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:10 | 0.999847 | 0.999846458435 | 0.999846458435 | True |
| 3 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:20 | 0.999869 | 0.999869704247 | 0.999869704247 | True |
| 4 | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 2018-03-02 11:13:30 | 0.999870 | 0.999869704247 | 0.999869704247 | True |

## 14. Performance

- Device: cuda:0
- GPU latency (no explanations): 3.64 ms
- GPU latency (with explanations): 42.08 ms
- Peak GPU memory: 18.78 MiB
- CPU latency (no explanations): 0.0026664999895729125

## 15. Tests

See the final response for exact pass counts across Phase 9K and the required 9G/9H/9I/9J/7B/7C/7D/8A/8B regressions.

## 16. Frozen-Artifact Integrity

Prior result directories unchanged: **True**

## 17. Remaining Limitations

- The inference engine is not yet called from any Django view -- it exists as a standalone, tested module only.
- Only a CSV/window input contract is supported; PCAP ingestion was explicitly out of scope (Phase 9H/9I: RED) and was not revisited.
- No per-step attack probability, uncertainty, or counterfactual capability exists (explicitly deferred to later phases per this phase's constraints).
- Track B (world_model/) remains on disk, unmodified, quarantined by documentation only -- full retirement (file deletion) was explicitly out of scope.
- A harmless sklearn version-mismatch warning appears when unpickling the scaler (pickled with scikit-learn 1.9.1, current environment has 1.8.0); predictions were verified correct despite the warning (see deterministic validation), but this should be resolved by re-pickling with the current sklearn version in a future environment-hygiene pass.

## 18. Explicitly Unsupported Capabilities

- SHAP (actual method is gradient x input with SHAP-compatible infrastructure)
- Native per-step attack probability (only whole-horizon P(attack in t+1..t+6))
- Calibrated probabilities (Phase 8B finding stands unchanged)
- Causal attacker kill-chain inference (MITRE stages are a reasoned label mapping)
- Unseen-attack generalization claims (not tested in this phase)
- PCAP-derived model input (Phase 9H/9I RED verdicts stand unchanged)
- Uncertainty quantification of any kind
- Counterfactual simulation of any kind

## 19. Recommended Phase 9L

Phase 9L should wire experiments/inference_engine.py into world_model/inference_service.py (or a new, thin Django view) so the live dashboard displays real model output instead of the heuristic fallback -- pure integration, no new modeling, following the same audit-then-integrate discipline used since Phase 9H. It should NOT attempt per-step risk, uncertainty, or counterfactual work -- those remain separate, independently audited future phases per Phase 9J's roadmap.

STOP AFTER PHASE 9K. Do not begin 9L.
