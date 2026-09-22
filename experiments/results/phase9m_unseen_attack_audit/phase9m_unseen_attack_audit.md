# Phase 9M: Unseen-Attack Generalization Audit

Status: **YELLOW**

The audit and leave-one-family-out experiments are methodologically defensible (verified walk-forward split, whole-day held-out-family exclusion, independently-fit scaler, validation-only threshold selection, zero train/validation leakage -- see leakage_checks.json per family), and produced honest, non-overclaimed, family-dependent evidence (Bot: CLAIM B partial generalization; Infiltration: CLAIM C no generalization). YELLOW rather than GREEN because CSE-CIC-IDS2018 confines each attack family to specific capture day(s), so every unseen-family result here is structurally confounded with an unseen-day/background-traffic shift that this dataset cannot separate out, and only 2 of 6 real attack families were even SUITABLE to test.

## 1. Executive Summary

This phase asked whether the frozen Phase 3.5 CIC-IDS2018 forecasting dataset and split can support a defensible leave-one-attack-family-out generalization experiment, and if so, ran the minimum such experiment. Of 6 real attack families, only Bot and Infiltration were SUITABLE (DoS and Web Attack were structurally excluded -- too little train signal / too few test rows; Brute Force and DDoS were MARGINAL and not run, since each groups multiple distinct raw labels under one family name). Leave-one-family-out Temporal Transformer models (Phase 5 architecture, retrained fresh, new scaler, new checkpoint, all confined to this phase's own result directory) were trained excluding each family's entire source-day file(s) from train+validation, then evaluated on the UNMODIFIED frozen test split, stratified into unseen-family-positive and seen-family-positive subsets against a shared negative pool. Result: **Bot shows partial, limited generalization** (unseen ROC-AUC 0.678 vs seen 0.771 -- above chance but a real gap); **Infiltration shows no generalization** (unseen ROC-AUC 0.475, at chance, vs seen 0.879). Every candidate family in this dataset is confined to specific capture day(s), so unseen-family results are structurally confounded with unseen-day/background-traffic shift -- this dataset cannot fully separate the two.

## 2. Frozen Dataset Inventory

Read-only inspection of `data/windows/*.csv` (10 files, `forecast_sample_eligible` rows only). Raw attack labels observed: 14. Unmapped labels: none. One-family-per-source-day invariant holds: **True**.

| raw_label | family | train | validation | test | total |
|---|---|---:|---:|---:|---:|
| Bot | Bot | 1042 | 346 | 635 | 2023 |
| Brute Force -Web | Web Attack | 1 | 229 | 15 | 245 |
| Brute Force -XSS | Web Attack | 125 | 0 | 0 | 125 |
| DDoS attack-HOIC | DDoS | 135 | 0 | 0 | 135 |
| DDoS attack-LOIC-UDP | DDoS | 0 | 0 | 136 | 136 |
| DDoS attacks-LOIC-HTTP | DDoS | 56 | 343 | 24 | 423 |
| DoS attacks-GoldenEye | DoS | 0 | 89 | 0 | 89 |
| DoS attacks-Hulk | DoS | 21 | 0 | 0 | 21 |
| DoS attacks-SlowHTTPTest | DoS | 0 | 276 | 0 | 276 |
| DoS attacks-Slowloris | DoS | 0 | 65 | 176 | 241 |
| FTP-BruteForce | Brute Force | 0 | 226 | 347 | 573 |
| Infiltration | Infiltration | 930 | 474 | 313 | 1717 |
| SQL Injection | Web Attack | 51 | 4 | 0 | 55 |
| SSH-Bruteforce | Brute Force | 546 | 0 | 0 | 546 |

## 3. Attack-Family Definitions

Standard, published CSE-CIC-IDS2018 category grouping (verified against every raw label actually observed in the data, not assumed):

- **Bot**: raw labels `['Bot']`; source day(s): ['Friday-02-03-2018_TrafficForML_CICFlowMeter.csv']
- **DoS**: raw labels `['DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris']`; source day(s): ['Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv']
- **Web Attack**: raw labels `['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection']`; source day(s): ['Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv']
- **DDoS**: raw labels `['DDoS attack-HOIC', 'DDoS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP']`; source day(s): ['Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv']
- **Infiltration**: raw labels `['Infiltration']`; source day(s): ['Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']
- **Brute Force**: raw labels `['FTP-BruteForce', 'SSH-Bruteforce']`; source day(s): ['Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv']

## 4. Train/Validation/Test Family Distribution

family | raw labels | source day(s) | train | val | test | total
|---|---|---|---:|---:|---:|---:|
| Bot | Bot | Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | 1042 | 346 | 635 | 2023 |
| Brute Force | FTP-BruteForce, SSH-Bruteforce | Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv | 546 | 226 | 347 | 1119 |
| DDoS | DDoS attack-HOIC, DDoS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP | Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv, Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv | 191 | 343 | 160 | 694 |
| DoS | DoS attacks-GoldenEye, DoS attacks-Hulk, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris | Friday-16-02-2018_TrafficForML_CICFlowMeter.csv, Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv | 21 | 430 | 176 | 627 |
| Infiltration | Infiltration | Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv, Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv | 930 | 474 | 313 | 1717 |
| Web Attack | Brute Force -Web, Brute Force -XSS, SQL Injection | Friday-23-02-2018_TrafficForML_CICFlowMeter.csv, Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv | 177 | 233 | 15 | 425 |

Structurally flagged (too small / absent from a split / unsuitable for holdout):
- **DDoS**: spans multiple source days (['Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv']) -- holdout removes >1 day
- **DoS**: spans multiple source days (['Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv']) -- holdout removes >1 day
- **Infiltration**: spans multiple source days (['Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']) -- holdout removes >1 day
- **Web Attack**: spans multiple source days (['Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv']) -- holdout removes >1 day

## 5. Candidate Held-Out Family Audit

| candidate | train removal (rows / % train) | test rows | semantic distinctness | leakage risk | suitability |
|---|---|---:|---|---|---|
| Bot | 1042 / 3.554% | 635 | high (botnet C2/zombie behavior, distinct network signature) | low -- single day, cleanly removable | **SUITABLE** |
| Brute Force | 546 / 1.863% | 347 | high (credential brute-forcing, distinct from volumetric/flood attacks) | low -- single day, cleanly removable | **MARGINAL** |
| DDoS | 191 / 0.652% | 160 | medium (grouped tools: HOIC/LOIC-UDP/LOIC-HTTP have different volumetric signatures) | low-medium -- 2 days removable as a block, but multi-day removal shifts more of the benign-traffic distribution at once | **MARGINAL** |
| DoS | 21 / 0.072% | 176 | medium (grouped tools: Slow* are low-and-slow, Hulk/GoldenEye are higher-rate) | low-medium -- 2 days removable as a block, but multi-day removal shifts more of the benign-traffic distribution at once | **UNSUITABLE** |
| Infiltration | 930 / 3.172% | 313 | high (stealthy, low-and-slow; often near-benign-looking by design) | low-medium -- 2 days removable as a block, but multi-day removal shifts more of the benign-traffic distribution at once | **SUITABLE** |
| Web Attack | 177 / 0.604% | 15 | low-medium (grouped: brute-force, XSS injection, SQLi are mechanistically different) | low-medium -- 2 days removable as a block, but multi-day removal shifts more of the benign-traffic distribution at once | **UNSUITABLE** |

- **Bot** (SUITABLE): single raw label, single source day, adequate train removal and test size
- **Brute Force** (MARGINAL): family groups 2 distinct raw labels (different tools) under one family name
- **DDoS** (MARGINAL): family groups 3 distinct raw labels across 2 different source days -- removing it changes multiple days' benign-traffic context at once, not just one attack signature
- **DoS** (UNSUITABLE): train removal impact negligible (21 eligible rows < 50)
- **Infiltration** (SUITABLE): single raw label, single source day, adequate train removal and test size
- **Web Attack** (UNSUITABLE): too few test-split rows (15 < 30) for a defensible evaluation

world_model/world_model.py's world_model/zero_shot_results.json (Track B, NON-AUTHORITATIVE, already flagged by Phase 9J for an undocumented/suspicious evaluation methodology) reports a preliminary leave-Infiltration-out result (test ROC-AUC ~0.37-0.43, recall <1.5%) using an ad-hoc, unaudited day split. This is cited here ONLY as external context motivating extra scrutiny of Infiltration, NOT as evidence -- Phase 9M does not reuse Track B code, data splits, or results.

## 6. Leakage Analysis

- Walk-forward split ordering holds (train block strictly precedes validation strictly precedes test, by `window_start`, within every source-day file): **True**
- Cross-split (source_file, window_start) leakage pairs in the frozen split: **0**
- Duplicate X sequences exist in the frozen split: True (pre-existing, all-zero idle-window rows -- not row-identity leakage; see per-experiment leakage_checks.json)
- Target is the unmodified binary `future_attack_within_horizon` flag, never converted to multiclass: **True**
- Feature columns exclude every meta/label column: **True**
- Reusing the existing Phase 4/5 scaler would leak the held-out family's statistics: **True** -- this is why Phase 9M fits a brand-new scaler per experiment, on the reduced train set only.

**Bot** leakage_checks.json: all_checks_pass=**True** (held-out rows in train/val: 0; scaler differs from original: True; nonzero train/test duplicate sequences: 0 (all-zero idle-window duplicates, benign: 320))
**Infiltration** leakage_checks.json: all_checks_pass=**True** (held-out rows in train/val: 0; scaler differs from original: True; nonzero train/test duplicate sequences: 0 (all-zero idle-window duplicates, benign: 320))

## 7. Experimental Protocol

- Primary family: **Bot**; secondary: **['Infiltration']**
- Not run (MARGINAL): ['Brute Force', 'DDoS']; (UNSUITABLE): ['DoS', 'Web Attack']
- Train exclusion rule: whole held-out-family source-day file(s) removed from train+validation (both attack and benign rows); test split never filtered
- Preprocessing: new StandardScaler fit per-experiment on the reduced train set only
- Threshold selection: chosen on the reduced validation set only (excludes held-out family), never on test
- Seed: 42

## 8. Model Configuration

- PRIMARY: TemporalTransformer (Phase 5 architecture, reused)
- CONTROL: LogisticRegression (Phase 4 convention, reused)
- Excluded by design: ['LSTM variants', 'World Model', 'GNN', 'any new architecture']

## 9. Training Details

| family | best_epoch | best_val_pr_auc | duration_s | parameters | device |
|---|---:|---:|---:|---:|---|
| Bot | 5 | 0.7195 | 18.82 | 286081 | cuda:0 |
| Infiltration | 1 | 0.8035 | 10.53 | 286081 | cuda:0 |

## 10. Results

| family | model | subset | n | pos | recall | precision | f1 | pr_auc | roc_auc | fpr | threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bot | Transformer | unseen | 5069 | 637 | 0.834 | 0.162 | 0.272 | 0.176 | 0.678 | 0.619 | 0.001 |
| Bot | Transformer | seen | 5558 | 1126 | 0.795 | 0.246 | 0.376 | 0.666 | 0.771 | 0.619 | 0.001 |
| Bot | LogReg control | unseen | 5069 | 637 | 0.407 | 0.172 | 0.242 | 0.155 | 0.647 | 0.281 | 0.529 |
| Bot | LogReg control | seen | 5558 | 1126 | 0.718 | 0.394 | 0.509 | 0.742 | 0.771 | 0.281 | 0.529 |
| Infiltration | Transformer | unseen | 4744 | 312 | 0.179 | 0.037 | 0.061 | 0.088 | 0.475 | 0.332 | 0.003 |
| Infiltration | Transformer | seen | 5883 | 1451 | 0.873 | 0.463 | 0.605 | 0.870 | 0.879 | 0.332 | 0.003 |
| Infiltration | LogReg control | unseen | 4744 | 312 | 0.356 | 0.084 | 0.136 | 0.171 | 0.622 | 0.272 | 0.335 |
| Infiltration | LogReg control | seen | 5883 | 1451 | 0.861 | 0.509 | 0.639 | 0.843 | 0.852 | 0.272 | 0.335 |

## 11. Seen-vs-Unseen Comparison

**Bot**: unseen ROC-AUC 0.678 vs seen ROC-AUC 0.771 (gap 0.093); control-model unseen ROC-AUC 0.647.
**Infiltration**: unseen ROC-AUC 0.475 vs seen ROC-AUC 0.879 (gap 0.404); control-model unseen ROC-AUC 0.622.

## 12. Statistical Uncertainty

**Bot**: unseen-family recall bootstrap 95% CI: {'mean': 0.8341538461538461, 'ci_lower_2.5pct': 0.8053375196232339, 'ci_upper_97.5pct': 0.8618524332810047, 'n_boot_valid': 1000}; seen-family PR-AUC bootstrap 95% CI: {'mean': 0.6665113080537528, 'ci_lower_2.5pct': 0.6363197190295933, 'ci_upper_97.5pct': 0.694961336607966, 'n_boot_valid': 1000}
**Infiltration**: unseen-family recall bootstrap 95% CI: {'mean': 0.1788621794871795, 'ci_lower_2.5pct': 0.13782051282051283, 'ci_upper_97.5pct': 0.22115384615384615, 'n_boot_valid': 1000}; seen-family PR-AUC bootstrap 95% CI: {'mean': 0.8696319918818416, 'ci_lower_2.5pct': 0.8545132423012104, 'ci_upper_97.5pct': 0.8840776895175081, 'n_boot_valid': 1000}

## 13. Failure Analysis

Infiltration's unseen ROC-AUC (0.475) is at chance despite a strong seen-family ROC-AUC (0.879) for the SAME architecture and training recipe -- the model is not failing to learn a discriminative signal in general, it is failing to TRANSFER it to Infiltration's held-out day. This is consistent with (not proof of, since methodology differs entirely) the pre-existing, unaudited Track B zero-shot result for Infiltration (Section 5's external-context note), which also found near-chance unseen performance. Bot's unseen ROC-AUC (0.678) is well above chance but the F1-selected operating point for the Transformer has a pathologically low threshold (0.00067) driven by the validation set's probability distribution, producing simultaneously high recall (0.834) AND high FPR (0.619) -- a threshold artifact, not evidence of strong practical detection capability at a usable operating point.

## 14. Scientific Interpretation

**Bot**: **CLAIM B** -- Unseen-family ROC-AUC is 0.678 -- meaningfully above chance (0.5), showing the model's ranking carries real, non-random signal for this held-out family. But PR-AUC (0.176) is far below the seen-family PR-AUC (0.666), and at the validation-selected operating point recall is 0.834 with FPR 0.619 (a high FPR at this threshold, so the raw recall number alone overstates usefulness). This is limited/partial generalization, not reliable detection.
**Infiltration**: **CLAIM C** -- Unseen-family ROC-AUC is 0.475 -- at or near chance (0.5), versus a much higher seen-family ROC-AUC (0.879). The ranking the model learned does not transfer to this held-out family; this experiment does not demonstrate unseen-family generalization.

Distinguishing "unseen attack family" from "traffic that looks statistically similar to known attacks": Bot's partial transfer (ROC-AUC 0.678) may reflect either (a) genuine attack-generic signal (unusual connection counts/durations/byte patterns common to many attack types) or (b) coincidental similarity between Bot's C2 traffic and some seen family's traffic shape -- this experiment cannot distinguish these two explanations; only a controlled ablation of engineered-feature semantics could. Infiltration's chance-level unseen result suggests its traffic pattern (deliberately designed to look benign) is NOT statistically close to any seen attack family, which is an intuitive, dataset-consistent explanation.

## 15. SIH Requirement Mapping

PS text requirement: "generalize to unseen attack patterns." This phase provides the FIRST audited, leakage-checked, direct evidence on this requirement (superseding Phase 9J's citation of Track B's unaudited zero-shot result as preliminary-only). Finding: generalization is **family-dependent and partial at best** in this dataset/model combination -- not demonstrated broadly, not absent either. The PS requirement is NOT satisfied by the current production model (Phase 6B/7B, Phase 9K's authoritative checkpoint) for arbitrary unseen families; it is partially supported for at least one tested family (Bot).

## 16. Limitations

- **Family/day confound**: every family in this dataset occupies specific capture day(s); unseen-family evaluation is inseparable from unseen-day/background-traffic-shift evaluation here.
- Only 2 of 6 real attack families were SUITABLE to test (DoS train signal negligible; Web Attack test set too small).
- Brute Force and DDoS were MARGINAL (multi-raw-label families) and were not run in this minimal experiment -- a natural next step, not performed here.
- The Transformer's F1-selected threshold can be a poor practical operating point (see Bot); ROC-AUC/PR-AUC are more reliable in this report than the single-threshold recall/precision numbers.
- This is one seed, one architecture family (Transformer + LogReg control); no multi-seed variance estimate of the TRAINING process itself (only bootstrap CI over the fixed trained model's test predictions).
- New checkpoints/scalers were created for this phase ONLY (under this phase's result directory); they are NOT used by the Django authoritative inference path and do not change production behavior.

## 17. Claim-Safety Statement

None of the following are claimed anywhere in this report or its artifacts: "generalizes to all unseen attacks", "zero-shot attack detection", "novel attack detection", "unknown attack detection", "detects attacks never seen before", "works on arbitrary future attack families". Every claim is scoped to exactly the tested family and protocol (Bot: CLAIM B partial generalization; Infiltration: CLAIM C no generalization).

## 18. Final Verdict

**YELLOW**

The audit and leave-one-family-out experiments are methodologically defensible (verified walk-forward split, whole-day held-out-family exclusion, independently-fit scaler, validation-only threshold selection, zero train/validation leakage -- see leakage_checks.json per family), and produced honest, non-overclaimed, family-dependent evidence (Bot: CLAIM B partial generalization; Infiltration: CLAIM C no generalization). YELLOW rather than GREEN because CSE-CIC-IDS2018 confines each attack family to specific capture day(s), so every unseen-family result here is structurally confounded with an unseen-day/background-traffic shift that this dataset cannot separate out, and only 2 of 6 real attack families were even SUITABLE to test.

STOP AFTER PHASE 9M. Do NOT start Phase 9N.
