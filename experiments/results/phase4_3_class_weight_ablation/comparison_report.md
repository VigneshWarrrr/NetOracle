# Phase 4.3 Class-Weight Ablation

Only `pos_weight` changed from the Phase 4 weighted LSTM. No raw data, temporal data, split, architecture, scaler methodology, optimizer, learning rate, batch size, or existing artifact was modified.

## Comparison

| Model | Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Threshold | TP | TN | FP | FN | Best epoch | Duration (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_lstm | validation | 0.614870 | 0.922068 | 0.737768 | 0.805096 | 0.833505 | 0.415487 | 0.00076760322554 | 2390 | 2106 | 1497 | 202 | 9 | not recorded |
| weighted_lstm | test | 0.430169 | 0.894498 | 0.580954 | 0.853651 | 0.885276 | 0.471345 | 0.00076760322554 | 1577 | 2343 | 2089 | 186 | 9 | not recorded |
| unweighted_lstm | validation | 0.630005 | 0.910494 | 0.744714 | 0.800445 | 0.833157 | 0.384679 | 0.000555350503419 | 2360 | 2217 | 1386 | 232 | 9 | 18.699699700002384 |
| unweighted_lstm | test | 0.424242 | 0.857629 | 0.567674 | 0.857640 | 0.885502 | 0.462996 | 0.000555350503419 | 1512 | 2380 | 2052 | 251 | 9 | 18.699699700002384 |

## Diagnostic interpretation

- Removing class weighting changed the test false-positive count from 2089 to 2052 at each model's validation-selected threshold.
- Test recall changed from 0.894498 to 0.857629; test F1 changed from 0.580954 to 0.567674.
- Test PR-AUC changed from 0.853651 to 0.857640; test ROC-AUC changed from 0.885276 to 0.885502.
- At fixed threshold 0.10, the unweighted model has test FPR 0.107626, recall 0.753829, and F1 0.744746; the weighted model has FPR 0.219765, recall 0.762904, and F1 0.658991. At fixed threshold 0.50, the unweighted model has FPR 0.050767, recall 0.742484, and F1 0.794055. These are diagnostic fixed-threshold comparisons, not test-selected operating points.
- The unweighted model's selected validation threshold is 0.0005553505, lower than the weighted model's 0.0007676032. Therefore removing class weighting does not by itself solve the low-threshold problem.
- Probability scores become less aggressive for benign samples: on test, benign mean/median/P75 move from 0.136698/0.000399/0.071871 weighted to 0.059213/0.000302/0.008057 unweighted. Positive test means remain similar (0.749474 weighted versus 0.743456 unweighted).
- No single metric alone determines the better practical forecasting model.

## Explicit answers

1. **Does removing class weighting reduce false positives?** Slightly at validation-selected thresholds: 2,089 to 2,052, a reduction of 37 false positives and FPR from 0.471345 to 0.462996. At fixed thresholds, the reduction is much clearer, especially at 0.10 and above.
2. **How does recall change?** Test recall falls from 0.894498 to 0.857629, a decrease of 0.036869.
3. **How does F1 change?** Test F1 falls from 0.580954 to 0.567674, a decrease of 0.013280 at the respective validation-selected thresholds. At fixed threshold 0.10, unweighted F1 is higher (`0.744746` versus weighted `0.658991`).
4. **Does PR-AUC change?** It increases slightly from 0.853651 to 0.857640 on test, while validation PR-AUC decreases from 0.805096 to 0.800445.
5. **Does ROC-AUC change?** It increases slightly on test from 0.885276 to 0.885502, while validation ROC-AUC decreases from 0.833505 to 0.833157.
6. **Does the LSTM retain useful ranking ability?** Yes. Both weighted and unweighted models retain strong test PR-AUC and ROC-AUC; the unweighted model is marginally better on both test ranking metrics.
7. **Does the probability distribution become less aggressive?** Yes for benign samples. The unweighted test benign mean drops by 0.077485 and its P75 drops by 0.063815. The selected threshold remains extremely low, so the operating point is still aggressive.
8. **Is class weighting a major cause of the original high FPR?** It is a contributor, not the sole or dominant explanation. Removing it reduces benign scores and fixed-threshold FPR, but the validation-selected threshold remains near zero and test FPR remains 0.462996. Threshold/score-scale behavior and temporal distribution shift remain major factors.

## Conclusion

The class-weight ablation shows a **combination**: class weighting amplifies benign scores and false positives at comparable fixed thresholds, but it does not fully explain the original high FPR. The unweighted model has slightly better ranking metrics and a less aggressive benign score distribution, yet its validation-selected threshold still produces a high test FPR and lower recall/F1 than the weighted model. The practical choice therefore depends on the required recall/FPR operating point rather than a single aggregate metric.
