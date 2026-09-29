# Phase 6B Vector World Model

This experiment reuses the unchanged Phase 3.5 canonical dataset through `world_model_dataset.py` (itself reused, unmodified, by this run) and the unchanged Phase 4 `future_attack_within_horizon` target. It performs a genuine learned latent rollout: history is encoded and contextualized, then a residual transition function is applied recursively six times to produce predicted latents for t+1...t+6, which are decoded into predicted states and pooled into a single attack probability. The attack head never receives the ground-truth future states Y in its forward pass.

## Architecture

- Input: `6 x 157`, Output rollout: `6 x 157`
- d_model=128, heads=4, layers=2, ff_dim=256, dropout=0.2
- Parameters: `347806`
- Device: `cuda:0`
- Mixed precision: `True`
- Best epoch: `1`
- Training duration: `26.184` seconds
- Selected validation threshold: `0.043854560703`

## Comparison (test set, matched target)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression (Phase 4, test) | 0.584603 | 0.805445 | 0.677481 | 0.845412 | 0.849503 | 0.227662 | 1420 | 3423 | 1009 | 343 |
| lstm (Phase 4, test) | 0.430169 | 0.894498 | 0.580954 | 0.853651 | 0.885276 | 0.471345 | 1577 | 2343 | 2089 | 186 |
| temporal_transformer (Phase 5, test) | 0.617992 | 0.806580 | 0.699803 | 0.863815 | 0.876298 | 0.198330 | 1422 | 3553 | 879 | 341 |
| vector_world_model (Phase 6B, test) | 0.418710 | 0.891095 | 0.569719 | 0.868170 | 0.877507 | 0.492103 | 1571 | 2251 | 2181 | 192 |

## Per-horizon state prediction error (test, standardized feature scale)

| Horizon | MSE | MAE |
|---|---:|---:|
| t+1 | 2132.476318 | 1.449314 |
| t+2 | 2131.105469 | 1.455564 |
| t+3 | 2131.321289 | 1.466707 |
| t+4 | 2131.999512 | 1.479216 |
| t+5 | 2129.841064 | 1.490270 |
| t+6 | 2127.759277 | 1.501541 |

## Per-horizon attack/risk performance (test, at the validation-selected threshold)

| future_attack_step | meaning | count | mean predicted probability | predicted-positive rate |
|---|---|---:|---:|---:|
| 0 | no attack in horizon (negative) | 4432 | 0.225812 | 0.492103 |
| 1 | earliest attack at t+1 | 1643 | 0.806745 | 0.883141 |
| 2 | earliest attack at t+2 | 47 | 0.843689 | 1.000000 |
| 3 | earliest attack at t+3 | 29 | 0.770658 | 1.000000 |
| 4 | earliest attack at t+4 | 28 | 0.761490 | 1.000000 |
| 5 | earliest attack at t+5 | 11 | 0.450484 | 1.000000 |
| 6 | earliest attack at t+6 | 5 | 0.538770 | 1.000000 |

The threshold was selected using validation predictions only; the test set was used once for final evaluation. State-error is reported on the standardized feature scale used by the training loss (the scaler is fit on train history states only). `future_attack_step=0` samples have no attack anywhere in the horizon (their predicted-positive rate is a false-positive rate); `future_attack_step=k` samples have their earliest in-horizon attack at t+k.

