# Phase 7B: MITRE Stage Head (frozen backbone)

## Architecture

`X [B,6,157] -> frozen Phase 6B Run-1 backbone -> z_future [B,6,128] -> (existing AttackForecastHead -> attack risk, unchanged) + (new MitreStageHead -> [B,6,6] stage logits)`.

`VectorWorldModelWithStageHead` (experiments/phase7b_mitre_stage_head.py) subclasses `VectorWorldModel` (experiments/phase6b_vector_world_model.py, not modified) and reuses its `state_encoder`, `temporal_context`, `transition`, `state_decoder`, `attack_head` submodules unchanged. `forward()` returns `(y_hat, attack_logit, stage_logits, z_future)`.

## Target construction

`experiments/phase7b_stage_targets.py` independently re-reads `data/windows/*.csv` (does not import or modify `world_model_dataset.py`). Per-step targets come from each future row's own `current_attack_types`; the aggregate headline target comes from `future_attack_types`. Both route through the corrected, Phase-7A-tested `MitreMapper.from_label_set()`.

## Leakage prevention

`forward()` takes only `x` (observed history). Stage/state/attack targets never appear inside `forward()` -- verified at runtime (see tests.py: test_backbone_outputs_require_no_grad, test_forward_signature_has_no_label_arguments) that `y_hat`/`attack_logit`/`z_future` have `requires_grad=False` when the backbone is frozen, and only `stage_logits` carries a gradient.

## Class coverage limitation

Only 6 of 14 MitreStage values occur in this dataset: BENIGN, INITIAL_ACCESS, CREDENTIAL_ACCESS, LATERAL_MOVEMENT, COMMAND_AND_CONTROL, IMPACT. The following are NOT represented in CIC-IDS2018 training data and are NOT evaluated: RECONNAISSANCE, EXECUTION, PERSISTENCE, PRIVILEGE_ESCALATION, DEFENSE_EVASION, DISCOVERY, COLLECTION, EXFILTRATION.

## Frozen-backbone protocol

Pre-flight reproduction status: **PASS**. Post-training attack-metric identity to pre-flight: **True**. Backbone byte-identical before/after training: **True**.

## Metrics

Test stage accuracy: 0.8395, macro-F1: 0.6094.
Progression consistency (test): 0.9784.
Expected calibration error (test): 0.1032.

See metrics.json for full per-class precision/recall/support (including the explicit INITIAL_ACCESS zero-test-coverage note) and the confusion matrix.

