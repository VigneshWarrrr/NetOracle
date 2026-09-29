# Phase 10A: NetOracle AI-Core Final Audit

Audit date: 2026-09-20. Scope: WORLD MODEL + AI CORE. **Audit only -- nothing was trained, fixed, retrained, or modified.**

## Gate

**AI_CORE_SUBMISSION_READY = NO**

Findings: 11 GREEN, 23 YELLOW, **6 RED**. Claims audited: 32 (12 GREEN / 13 YELLOW / 7 RED).

The numerical core is sound: the authoritative checkpoint, scaler, feature schema and engine were verified end-to-end and reproduce the frozen results exactly (finding F1), deterministically (G1), with the MITRE trajectory demonstrably coming from the stage head (H1). The answer is NO because of six defects that sit around the core rather than inside it -- packaging/provenance, the demo's start-up and explanation panel, and text that contradicts the implementation (README, landing page, deck). Each is small; none needs a change to the model weights or checkpoint (I3 changes what the explainer differentiates, and must be re-validated).

### RED items (must fix before submission)

- **C1 -- README documents the LEGACY pipeline as the forecasting path.** Fix: Rewrite README around the authoritative path (inputs, checkpoint, scaler, data dependency, how to run, limitations); demote or delete the legacy instructions.
- **C2 -- Post-login landing page still shows stale legacy status.** Fix: Point the landing page's forecast panel at the authoritative service or replace the stale hero/status copy; remove the five-window wording.
- **I2 -- SHAP is claimed in the SIH deck but never executes.** Fix: Correct the deck wording to 'gradient x input feature attribution' (or make real SHAP work and re-validate 7C); do not ship the current claim.
- **I3 -- Attributions saturate: the featured demo sample's explanation panel shows ten 0.0000 values.** Fix: Attribute the pre-sigmoid logit (or normalise/scale importances) and re-validate against Phase 7C; at minimum display relative importance instead of absolute 4-decimal values.
- **O1 -- Django cannot start in this environment (reportlab declared but not installed).** Fix: pip install -r requirements.txt on the demo machine and re-run manage.py check (no code change).
- **O2 -- The authoritative AI core is not in version control and the checkpoints are gitignored.** Fix: Commit the core; ship checkpoint + scaler (un-ignore or release artifact) and a data/feature-schema bundle; re-tag so provenance points at the committed state.

## Method

Read-only. Every module on the authoritative path was read in full or in its relevant classes (Phase 6B model, Phase 7B head, 9K engine, 9L service/views/API/template, 7C explain_sample, the explainer's method-selection and gradient code); legacy, Track-B and dead modules were inspected for reachability and labeling only, not audited internally. Numerical claims were re-measured by `experiments/phase10a_ai_core_audit.py` (hash/tensor comparison, full 6,195-sample test reproduction against frozen Phase 7B/Run-1 predictions, determinism incl. CPU-vs-GPU, causal stage-head perturbation, explanation study on 300 windows, 4-thread concurrency test, repository/claim scans incl. the SIH deck text, environment/git checks, fresh-process cold start, rendered Django pages/API under auth). The one shim used (a stub `reportlab` in a temp directory) exists only to render pages on this machine and was never placed in the repo. All evidence is in `evidence.json`; this report is generated from it by `experiments/phase10a_write_report.py`.

## A. Exact authoritative AI-core execution path

### A1 [GREEN] Authoritative execution path is fully traceable input -> output

Django page/API -> world_model.inference_service.AuthoritativeForecastService (lazy class-level singleton) -> experiments.inference_engine.NetOracleInferenceEngine.predict() -> validate (6,157) finite -> frozen Run-1 StandardScaler -> VectorWorldModelWithStageHead forward (frozen Run-1 backbone + Phase 7B stage head) -> sigmoid/softmax -> inverse-scaled rollout -> Phase 7C explain_sample() -> JSON dict. Every hop is documented with file:line in authoritative_path.md.

Evidence: authoritative_path.md; experiments/inference_engine.py:157-305; world_model/inference_service.py:119-200

## B. Only one authoritative model/inference path

### B1 [GREEN] Exactly one Django-reachable path produces the authoritative outputs

Only experiments/inference_engine.py builds the model from best_stage_head.pt and the Run-1 scaler for the production surface. dashboard/views.py, api/views.py and the template call it only through AuthoritativeForecastService. No production module imports Track B (world_model/world_model.py) or Phase 9N (0 hits for per_step_risk/phase9n in engine, service, views, api, templates).

Evidence: path_inventory.phase9n_refs_in_authoritative_engine_or_django = 0 hits; AST import guards in tests/test_phase9k_integration.py and tests/test_phase9l_django_integration.py

## C. Legacy / duplicate / heuristic / Track-B / dead paths that could be mistaken for authoritative

### C1 [RED] README documents the LEGACY pipeline as the forecasting path

README.md states the system 'runs a persisted PyTorch forecaster' at models/attack_forecaster.pt (a file that does not exist; models/ contains only __init__.py), lists FlowFeatureEngine -> TemporalFeatureEngine -> 'Predicted attack type', attack-chain progression, next access step and precautions as dashboard outputs, and gives train.py / run_pipeline.py as the way to train and run. None of that is the authoritative model (Phase 6B Run-1 + 7B stage head + 9K engine). A reader following the README will run the heuristic fallback and see a different system from the one audited.

Evidence: README.md lines 25, 46, 132, 142, 189 (legacy_heuristic_refs = 26 hits); models/ directory contains only __init__.py

Recommended action (not implemented): Rewrite README around the authoritative path (inputs, checkpoint, scaler, data dependency, how to run, limitations); demote or delete the legacy instructions.

### C2 [RED] Post-login landing page still shows stale legacy status

The first page a logged-in staff user sees (/dashboard/) renders 'Waiting for trained checkpoint', 'Forecast readiness -- Model offline', 'Checkpoint unavailable', 'projected across the next five observation windows' and 'Five-window rollout / Projected attack progression'. The authoritative model is working and its horizon is SIX windows; the page tells a judge the model is offline. (The per-window probability block is a dead loop over a context variable that no view sets.)

Evidence: rendered_pages.landing_dashboard_index.text (evidence.json); templates/dashboard/index.html lines 95-150

Recommended action (not implemented): Point the landing page's forecast panel at the authoritative service or replace the stale hero/status copy; remove the five-window wording.

### C3 [YELLOW] Legacy heuristic path is still reachable and partly mislabeled

AttackForecastService (sigmoid-of-mean heuristic, model_source='heuristic_fallback') is still reachable at /dashboard/forecast/ when live-capture events exist and via run_pipeline.py / feature_engine.services. The page carries a NON-AUTHORITATIVE banner (added in 9L), but its chart badge still reads 'Live model output', and run_pipeline.forecast_timeline repeats one flat risk value across future points (the pattern Phase 9N/9L forbid). Not reachable in a demo without a running capture; acceptable only if left unused.

Evidence: world_model/inference_service.py:27-116; run_pipeline.py:107-143; templates/dashboard/forecast.html:60

Recommended action (not implemented): Keep unused in the demo; ideally retire or hard-label the badge.

### C4 [YELLOW] Track B shares the `world_model/` package with the authoritative service

world_model/ contains BOTH the authoritative Django-facing service (inference_service.py) and Track B (world_model.py NetworkWorldModel, inference.py, build_project.py, streamlit dashboard, evaluation_results.json, zero_shot_results.json, 'Presentation SIH.pptx'). Verified never imported by the authoritative path, but a reviewer opening world_model/ will find an unaudited model with impressive-looking numbers (Phase 9J: ad-hoc split, validation/test metric inversion). Ownership is ambiguous even though behaviour is not.

Evidence: path_inventory.track_b_refs; Phase 9J findings

Recommended action (not implemented): Add a top-level note or move Track B under a clearly named legacy folder before submission.

### C5 [YELLOW] Dead heuristic 'infiltration probability' and placeholder chain logic remain in the tree

forecasting/infiltration.py::InfiltrationForecaster (hand-set thresholds 0.65/0.80, exported from forecasting/__init__.py) is referenced nowhere else; AttackChainBuilder.predict_next_stage() is a +1 placeholder used only by the legacy pipeline/streamlit. Neither is the model; both could be mistaken for the PS's 'infiltration probability'.

Evidence: forecasting/infiltration.py; forecasting/attack_chain.py; grep: InfiltrationForecaster has no call sites

Recommended action (not implemented): Delete or clearly mark as unused.

### C6 [YELLOW] Several same-named checkpoints on disk

best_model.pt exists for plain Phase 6B (sha 364276bd3f63...), Run-1 (sha 6374a9c47d72...), Run-2, Phase 4/5, plus the Phase 9N head. The engine hard-pins the Run-1 + Phase 7B paths, so the code cannot pick the wrong one; the risk is human (loading 'best_model.pt' from the wrong folder).

Evidence: artifact_integrity.run1_vs_plain_6b_checkpoints_identical = False; 23 checkpoint/scaler files under experiments/results

## D. Checkpoint / scaler compatibility

### D1 [GREEN] Checkpoint and scaler are the frozen, mutually compatible artifacts

best_stage_head.pt sha256 f9d16f1943aeeed4..., Run-1 best_model.pt 6374a9c47d722215..., scaler b3a0cea7f9d2d0fc... all equal the Phase 9K manifest. The Phase 7B checkpoint's 39 backbone tensors are bit-identical (torch.equal) to Run-1; only the 4 stage_head tensors (8646 params) are new (total 356452 params). The scaler has n_features_in_=157, finite mean/scale, no zero-variance feature (min scale 0.0041), and its mean/scale equal a fresh train-only refit exactly (max rel. diff 0.0e+00) while differing hugely from a train+val+test fit (max rel. diff 80.5) -- i.e. it is train-only.

Evidence: artifact_integrity; schema_and_scaler

### D2 [YELLOW] Scaler pickled under a newer scikit-learn than the runtime

The scaler was pickled with scikit-learn 1.9.1; this environment runs 1.8.0, so every load emits InconsistentVersionWarning. Values are verified correct (exact match to a train-only refit), so the effect is cosmetic here, but requirements.txt is unpinned (scikit-learn>=1.3), so a different demo machine may warn or, on an older release, fail to unpickle.

Evidence: artifact_integrity.scaler; supplement.scaler_pickled_with_sklearn; environment.requirements_txt_pins_exact_versions = False

Recommended action (not implemented): Pin scikit-learn (and torch/numpy) in requirements.txt to the tested versions; rehearse on the demo machine.

## E. Feature ordering and dimensionality

### E1 [GREEN] Feature dimensionality and ordering are correct and enforced

157 unique features, none from META_COLUMNS, identical order to the Phase 4 config's feature_columns list; schema hash f5117fda8518c9e3... equals the engine's provenance hash. X shape (6,157) float32 in all splits (29315/6195/6195); engine.validate_feature_vector rejects wrong type/shape/non-finite input (tests in 9K/9L).

Evidence: schema_and_scaler

### E2 [YELLOW] Feature schema is not stored with the checkpoint

The engine derives feature names/order at start-up by re-reading all ten Phase 3.5 CSVs (data/windows, ~70.9 MB) via read_world_model_samples; the checkpoint and scaler carry no feature names. Ordering is correct today (verified against the Phase 4 config) but is coupled to that external dataset directory.

Evidence: environment.windows_dir_used_by_engine; experiments/inference_engine.py:165-175

Recommended action (not implemented): Persist the 157-name list (and its hash) beside the checkpoint.

## F. Inference engine output == frozen checkpoint

### F1 [GREEN] Engine output corresponds exactly to the frozen checkpoint under the frozen numerical protocol

Running the loaded engine model with the same mixed-precision protocol Phase 7B used reproduces all 6195 frozen test predictions to max |dp| = 5.0e-13 with 0/37170 stage cells different. The engine is therefore the frozen model, not an approximation of it.

Evidence: supplement.amp_True; full_test_reproduction

### F2 [YELLOW] The shipped fp32 engine path differs from the frozen predictions at the 1e-3 level

The engine itself runs fp32 (no autocast). Against the frozen Phase 7B predictions_test.csv over the FULL test split: max |dp| = 2.40e-03 (mean 3.6e-05), stage-step exact match 99.984% (37164/37170); 6 samples have one argmax flip at a decision boundary. Phase 9K quoted a 1e-3 tolerance validated on 5 samples; on the full split the maximum is above it. Cause proven by the AMP re-run above (fp16 vs fp32 arithmetic), not by a checkpoint or preprocessing difference.

Evidence: full_test_reproduction; supplement.amp_False

Recommended action (not implemented): State the tolerance as 'fp16-vs-fp32 numerical, <= 2.5e-3' in submission text; no code change needed.

## G. Deterministic inference

### G1 [GREEN] Inference is deterministic

300 samples: repeated calls and a freshly constructed engine are bit-identical (max prob diff 0.0); CPU vs GPU max |dp| = 2.2e-04 with 100% identical stage trajectories; explanations are bit-identical on repeat and top-10 features identical CPU vs GPU. Concurrency: 4 threads x 3 rounds x 16 calls -> 0 exceptions, 0 result mismatches vs sequential.

Evidence: determinism; thread_safety

### G2 [YELLOW] Singleton has no lock (first-request race)

AuthoritativeForecastService.get_engine() checks `_engine is None` without a lock; two simultaneous first requests on the threaded dev server would each construct an engine (~20 s, double GPU memory). Not observed to corrupt results (concurrency test above ran on an already-loaded engine).

Evidence: thread_safety.singleton_has_no_lock; world_model/inference_service.py:145-156

Recommended action (not implemented): Warm the engine at start-up or add a lock.

## H. MITRE trajectory really comes from the stage head

### H1 [GREEN] The MITRE trajectory is produced by the authoritative stage head

For a test sample the engine's six stage names equal argmax(model.stage_head(z_future)) computed independently. Perturbing ONLY stage_head weights changes 510/600 stage cells across a 100-sample batch while the whole-horizon attack logit is bit-identical (attack head independent of stage head). Class space is the 6 covered MitreStage values; no rule-based or heuristic stage is substituted in the engine.

Evidence: stage_head

### H2 [YELLOW] Stage-trajectory quality is limited and must be presented as such

Phase 7B test: accuracy 0.839, macro-F1 0.609, stage-confidence ECE 0.103. INITIAL_ACCESS has ZERO test support, yet the engine predicts it in 2530/37170 (6.8%) of test step-cells -- by construction every one of those is unverifiable/incorrect on this split. Labels come from a reasoned CIC-label->MITRE mapping (Infiltration -> LATERAL_MOVEMENT is 'heuristic_uncertain'); the PS's five stages (Reconnaissance, Initial Access, Lateral Movement, C2, Exfiltration) are only partly covered (no Reconnaissance/Exfiltration in the data; Credential Access and Impact are extra). The dashboard already prints these caveats.

Evidence: full_test_reproduction.predicted_stage_step_counts_over_test; phase7b metrics.json; docs/MITRE_MAPPING_AUDIT.md

## I. Explainability: real method, no false SHAP claim

### I1 [GREEN] The reported attribution method is the one actually executed, and is not SHAP

Across 300 sampled test windows both the attack-risk and stage explanations report method ['Gradient × Input'] (Gradient x Input). The dashboard heading says 'Gradient-based feature attribution' and its HTML contains no 'SHAP'. The engine reports the method per call rather than assuming it.

Evidence: explainability

### I2 [RED] SHAP is claimed in the SIH deck but never executes

The shap package is installed (v0.50.0), but explain_attack_risk(method='auto') tries SHAP first and its GradientExplainer path raises `No module named 'tensorflow'`; the broad `except Exception` swallows it and the code silently falls back to Gradient x Input (forcing method='shap' raises). Yet 'Presentation SIH.pptx' slide 3 promises 'Explainability: SHAP values + feature importance'. The implemented method is Gradient x Input; the deck (and any derivative pitch text) is inaccurate. (No 'SHAP' claim appears in the UI or API; the API's faithful_claims_only note says 'gradient-based or SHAP attribution'.)

Evidence: explainability.explicit_shap_call; claims_scan.presentation_hits.shap: slide 3 ('SHAP values + feature importance', tech stack 'SHAP'), slide 4 ('SHAP-based explainability'), slide 5 ('Explainability: SHAP'); explainability/shap_explainer.py:110-165

Recommended action (not implemented): Correct the deck wording to 'gradient x input feature attribution' (or make real SHAP work and re-validate 7C); do not ship the current claim.

### I3 [RED] Attributions saturate: the featured demo sample's explanation panel shows ten 0.0000 values

The explainer back-propagates the sigmoid PROBABILITY (explainability/shap_explainer.py:_convert_to_probability -> target.backward()), whose gradient vanishes when the model is confident. For the demo sample (index 0, p=0.9999) the largest top-10 importance is ~1e-6, so the template's floatformat:4 renders all ten as 0.0000 (verified in the rendered page). Over 300 sampled windows the top-1 importance displays as 0.0000 in 30% of cases and in 44% of confident-attack (>0.99) cases; median top-1 importance is 1.2e-04 for those. The ranking is still deterministic and identical CPU vs GPU, but it is computed from noise-level magnitudes and the demo shows no usable numbers -- the panel looks broken exactly on the attack case a judge will see. The frozen Phase 7C records show the same ~1e-6 magnitudes for all five of its samples, so this is a property of the audited method, not of the Django integration.

Evidence: explainability; rendered_pages.authoritative_forecast.text (EXPLANATION block); phase7c records/sample_0001..0005.json: top attack-risk importances 9.7e-07, 1.3e-06, 2.4e-06, 2.4e-06, 2.4e-06 (verified)

Recommended action (not implemented): Attribute the pre-sigmoid logit (or normalise/scale importances) and re-validate against Phase 7C; at minimum display relative importance instead of absolute 4-decimal values.

### I4 [YELLOW] Method selection is silent and environment-dependent; UI text overstates where the library is disclosed

Because of the swallowed exception the method depends on what is installed (a machine with tensorflow + shap could silently switch to 'SHAP GradientExplainer' and change every attribution; not tested here). The authoritative page also says the 'exact underlying attribution library ... is available verbatim in explanations.faithful_claims_only' -- that note only says 'gradient-based or SHAP'; the exact method is in explanation_method.

Evidence: explainability.faithful_claims_note; templates/dashboard/authoritative_forecast.html (EXPLANATION panel)

Recommended action (not implemented): Report explanation_method on the page; remove the silent fallback or log it.

## J. Whole-horizon score vs per-step risk

### J1 [GREEN] Whole-horizon score is clearly distinguished from per-step risk

The API exposes exactly one risk field, whole_horizon_attack_probability {value, semantics}, whose semantics string says 'a single scalar, NOT a per-step P(t+1)...'. The six-step outputs are (a) predicted STATE vectors and (b) MITRE STAGE labels with a stage-class softmax 'confidence' -- neither is labeled as risk. The page's limitations panel repeats 'No native per-step attack probability'.

Evidence: rendered_pages.api_keys; rendered_pages.api_top_level_key_names_containing_risk_or_prob

### J2 [YELLOW] Per-step stage 'confidence' sits next to t+k labels

The MITRE table prints t+1..t+6 with a number (0.997...0.999). It is labeled confidence in the template but is the softmax of the stage class (uncalibrated, ECE 0.103) and could be misread as a per-step attack probability.

Evidence: rendered_pages.authoritative_forecast.text

Recommended action (not implemented): Add 'stage-class confidence, not attack risk' beside the column.

## K. Phase 9N excluded from the authoritative path

### K1 [GREEN] Phase 9N per-step risk is excluded from the authoritative path

0 references to per_step_risk / PerStepAttackRisk / phase9n / best_per_step in engine, service, views, api or templates; the 9N head exists only under experiments/results/phase9n_per_step_risk/ and is never loaded. 9N's own verdict is RED (mean PR-AUC 0.840 vs persistence 0.925 vs whole-horizon broadcast 0.836).

Evidence: path_inventory.phase9n_refs_in_production_code = 0

### K2 [YELLOW] 9N's persistence baseline is an oracle comparison

The 9N 'current-window persistence' baseline uses the ground-truth current_attack label, which does not exist at inference (it is not among the 157 features). It is a valid GATE for 9N's RED verdict but should not be quoted as 'a trivial baseline the deployed system could run'.

Evidence: forecast_vs_detection.oracle_persistence_uses_ground_truth_current_label_unavailable_at_inference

## L. Phase 9M unseen-family results represented honestly

### L1 [GREEN] 9M unseen-family result is represented honestly where it appears

The authoritative page states 'No unseen-attack-generalization claim'; no template, README or doc claims zero-day/unseen/novel detection (claims scan: only that disclaimer matched).

Evidence: claims_scan.unseen_or_zero_day

### L2 [YELLOW] Unseen-attack generalization is not demonstrated for the authoritative model

Phase 9M verdict YELLOW: Bot ROC-AUC unseen 0.678 vs seen 0.771 (CLAIM B, partial); Infiltration 0.475 vs 0.879 (CLAIM C, none). Those were Transformer/LogReg models on 2 of 6 families, and every family is confounded with capture day. The authoritative World Model itself was never leave-one-family-out evaluated. The PS phrase 'generalize to unseen attack patterns' is therefore unmet/unproven and must be presented as a limitation.

Evidence: phase9m_unseen_attack_audit.json

## M. Calibration not presented as reliable probabilities

### M1 [YELLOW] Scores are labeled 'probability' although calibration is poor

Phase 8B: World Model Run-1 raw test ECE 0.1121 (validation 0.217), raw test Brier 0.1143; the selected Platt calibrator made test calibration WORSE (ECE 0.1697, Brier 0.1365). The page shows 'Whole-horizon attack probability 0.9999' and the API key is whole_horizon_attack_probability; the page's limitations panel says 'No calibrated probabilities' and the README calls it 'probability-like'. Nothing claims calibration, but the headline label contradicts the disclaimer.

Evidence: phase8b_calibration/validation_report.md; rendered_pages.authoritative_forecast.text

Recommended action (not implemented): Relabel as 'attack-risk score (uncalibrated, 0-1)' in UI/API docs.

## N. Known limitations relevant to SIH

### N1 [YELLOW] World Model is not the best whole-horizon detector

Phase 8A (shared controlled-FPR comparison): World Model Run-1 PR-AUC 0.8487 / test F1 0.7471 / test FPR 0.0957 vs Temporal Transformer 0.8638 / 0.8279 / 0.0255. The claim 'World Model provides better early-warning forecasting than conventional baselines' is NOT SUPPORTED (recorded by 8A). The authoritative model is justified by what it adds (six-step state rollout, stage trajectory, attribution), not by beating the Transformer; no submission text may claim superiority.

Evidence: phase8_final_evaluation/FINAL_EVALUATION_TABLE.md

### N2 [YELLOW] Headline metrics mostly measure ongoing-attack detection; onset forecasting is thin

On the test split 1641/1763 (93.1%) of positives are windows whose CURRENT window is already an attack; only 122 are true onsets (attack begins within 60 s of a benign window), coming from 3 of 7 test days. On the currently-benign subset the World Model scores ROC-AUC 0.834 / PR-AUC 0.332 (prevalence 0.027); at the frozen Phase 8A threshold onset recall is 43.4% with 9.6% FPR. The Transformer is better on the same subset (PR-AUC 0.427). No lead-time analysis exists. 'Early warning' (used in the SIH deck) is at most weakly supported; 'near-term attack-risk forecasting over 6x10 s' is the defensible wording.

Evidence: forecast_vs_detection

Recommended action (not implemented): Reword 'early warning' claims; report the onset-subset numbers as the honest forecasting evidence.

### N3 [YELLOW] Demo shows one canned test-split sample; no live/uploaded input, no PCAP/Zeek

The authoritative page always runs predict_demo_sample(index=0) (Friday-02-03 Bot window, p=0.9999, all six stages COMMAND_AND_CONTROL); a benign case is reachable only via the API ?index= parameter. The engine's input contract is the 157-feature CICFlowMeter-window schema, which the live-capture pipeline does not produce; PCAP alignment was RED in 9H/9I. The page says so explicitly.

Evidence: rendered_pages.authoritative_forecast.text; Phase 9H/9I/9L

### N4 [YELLOW] Output is generic attack-risk, 6 x 10 s horizon, not infiltration-specific

The PS asks for 'infiltration probability over the next K windows'. Delivered: one generic whole-horizon score (K fixed at 6, i.e. 60 s) plus per-step MITRE stage labels; no infiltration-specific probability and no per-step risk (9N RED). Also: validation/test prevalence shift 41.8% vs 28.5% means fixed thresholds do not transfer (8A).

Evidence: Phase 9J/9N; phase8_final_evaluation

## O. What could break a clean demo run

### O1 [RED] Django cannot start in this environment (reportlab declared but not installed)

`python manage.py check` exits 1: ModuleNotFoundError: No module named 'reportlab'. server/views.py imports reportlab at URL-load time, so runserver, every page and the API fail. reportlab>=4.0 is in requirements.txt but not installed here; plotly, streamlit and PyPDF2 are also absent (not needed at start-up). Phase 9L's Django smoke tests pass only when the package is present or shimmed. Test run as-is today (9K+9L+9N): 116 tests -> FAILED (failures=3, errors=1).

Evidence: django_startup; environment; _test_run_as_is.log

Recommended action (not implemented): pip install -r requirements.txt on the demo machine and re-run manage.py check (no code change).

### O2 [RED] The authoritative AI core is not in version control and the checkpoints are gitignored

git HEAD 4e54c7533f ('final') contains NONE of: experiments/inference_engine.py, phase6b/7b/7c code, world_model_dataset.py, phase4_baseline.py, templates/dashboard/authoritative_forecast.html; 50 working-tree entries are uncommitted. .gitignore excludes *.pt, so best_stage_head.pt and Run-1 best_model.pt are ignored (scaler.joblib is merely untracked). The engine also reads data/windows from OUTSIDE the repo (../data/windows, 70.9 MB, not tracked). A clone/zip of the repository would have no engine, no checkpoint, no data. The provenance field code_commit_hash (4e54c7533fc3) shown on the page points to a commit that does not contain the audited code.

Evidence: environment.authoritative_files_tracked_in_git; environment.authoritative_artifacts_gitignored; environment.windows_dir_inside_repo = False

Recommended action (not implemented): Commit the core; ship checkpoint + scaler (un-ignore or release artifact) and a data/feature-schema bundle; re-tag so provenance points at the committed state.

### O3 [YELLOW] First request pays a ~19 s model load inside the HTTP request

Fresh process: engine load 19.1 s (reads all 10 CSVs + checkpoint), first prediction with explanations 0.25 s, warm 34 ms. The first browser request after runserver (and after each autoreload) will hang ~20 s.

Evidence: cold_start

Recommended action (not implemented): Warm the engine at start-up and rehearse.

### O4 [YELLOW] Fresh setup needs migrate + a staff account; page is staff-only

db.sqlite3 is absent (gitignored). The authoritative page requires is_staff (non-staff -> redirect 302, anonymous -> 302); the API accepts any authenticated user (anonymous 403, out-of-range/bad index 400/400).

Evidence: environment.db_sqlite3_present; rendered_pages

Recommended action (not implemented): Create the demo superuser during rehearsal.

### O5 [YELLOW] Judge-facing page contains build-phase jargon and a local absolute path

The page text includes 'established in Phase 9K', 'explicitly out of scope for this phase', 'Phase 9K/9L' and shows the checkpoint path as C:\AKSHAY\Akshay\...\best_stage_head.pt.

Evidence: rendered_pages.authoritative_forecast.text

Recommended action (not implemented): Polish copy; show the relative checkpoint path.

### O6 [YELLOW] Landing-page/deck framing overpromises relative to evidence

Hero copy 'See the next move before it lands' and deck slides ('Early warning before an attack reaches critical stages', 'SOC Early Warning Dashboard', 'Multi-Step Attack Forecasting', inputs 'PCAP / Zeek / CTI', baselines 'RF', 'Docker' deployment (no Dockerfile exists), 'Temporal GNN' as inspiration) describe intended scope, not what the audited core delivers (see N2/N3/N4). The deck is dated 2026-09-10 and is an idea-stage document.

Evidence: claims_scan.presentation_hits

Recommended action (not implemented): Align wording with claims_evidence_matrix.csv before submission.

## P. What MUST be fixed before submission

Consolidated from the RED findings above, in suggested order:

1. **O1 [RED]** Django cannot start in this environment (reportlab declared but not installed) -- pip install -r requirements.txt on the demo machine and re-run manage.py check (no code change).
2. **O2 [RED]** The authoritative AI core is not in version control and the checkpoints are gitignored -- Commit the core; ship checkpoint + scaler (un-ignore or release artifact) and a data/feature-schema bundle; re-tag so provenance points at the committed state.
3. **I3 [RED]** Attributions saturate: the featured demo sample's explanation panel shows ten 0.0000 values -- Attribute the pre-sigmoid logit (or normalise/scale importances) and re-validate against Phase 7C; at minimum display relative importance instead of absolute 4-decimal values.
4. **C2 [RED]** Post-login landing page still shows stale legacy status -- Point the landing page's forecast panel at the authoritative service or replace the stale hero/status copy; remove the five-window wording.
5. **C1 [RED]** README documents the LEGACY pipeline as the forecasting path -- Rewrite README around the authoritative path (inputs, checkpoint, scaler, data dependency, how to run, limitations); demote or delete the legacy instructions.
6. **I2 [RED]** SHAP is claimed in the SIH deck but never executes -- Correct the deck wording to 'gradient x input feature attribution' (or make real SHAP work and re-validate 7C); do not ship the current claim.

Not blockers but should be done in the same pass: D2 (pin versions), E2 (persist schema), O3 (warm start), N2/O6 (reword 'early warning'), M1 (relabel 'probability').

**P1 [YELLOW] Test-suite state and pre-existing failures.** Earlier phases record experiments/tests.py at 149/150 with one pre-existing, unrelated failure (Phase9FCrossHostGraphTests.test_build_comparison_table_ratios, a signature mismatch in frozen Phase 9F test code). It is unrelated to the AI core and was deliberately not modified.

## Test suites in the current environment

`tests.test_phase9k_integration tests.test_phase9l_django_integration tests.test_phase9n_per_step_risk` run as-is (no shim): **116 tests, FAILED (failures=3, errors=1)** in 1435.365 s.
- ERROR: test_no_forbidden_claims_rendered (tests.test_phase9l_django_integration.Step9DjangoSmokeTestCase.test_no_forbidden_claims_rendered)
- FAIL: test_all_smoke_steps_ok (tests.test_phase9l_django_integration.Step9DjangoSmokeTestCase.test_all_smoke_steps_ok)
- FAIL: test_api_view_responds (tests.test_phase9l_django_integration.Step9DjangoSmokeTestCase.test_api_view_responds)
- FAIL: test_dashboard_view_resolves_and_renders (tests.test_phase9l_django_integration.Step9DjangoSmokeTestCase.test_dashboard_view_resolves_and_renders)
- root cause: {'step': 'django_system_check', 'ok': False, 'detail': "No module named 'reportlab'"}

112 of 116 pass. All 4 non-passing tests are in `Step9DjangoSmokeTestCase`: **True**, and their assertion messages show the single cause `No module named 'reportlab'`. That is an environment failure (O1), not a model defect: the engine/contract/leakage/determinism tests in 9K and 9N and the non-smoke 9L tests pass, and the 9L suite passed when it was written (2026-09-15) with reportlab shimmed. Pre-existing unrelated failure recorded in P1.

## Frozen-artifact integrity of this audit

- Checkpoint/scaler hashes equal the Phase 9K manifest and were re-hashed at report time: **True**.
- Files created/modified since the audit began are only Phase 10A files (plus the runtime side effect below): **True** (11 files).
- Audit side effect, disclosed: `server.log` -- gitignored Django request log (settings.LOGGING -> BASE_DIR/server.log); the audit's rendered-page probe made three deliberate negative-path API requests (bad index, out-of-range index, anonymous) which Django logged as WARNING lines. Not a source, model, dataset or frozen artifact; left in place, not edited.
- No training, no model/Django/frontend/dataset change; the 10 tracked-modified files in `git status` are earlier-phase edits (mtimes precede this audit).

## Not covered by this audit

Other machines/OS, an environment with tensorflow installed (silent SHAP switch), CPU-only latency, Django security/DEBUG settings and the non-AI apps (logs/alerts/users/server), the legacy pipeline's internals beyond reachability, and any re-run of frozen training. These are listed so the gate is read as 'AI core as it exists and runs here', not as a statement about untested environments.

STOP. No implementation follows this audit.
