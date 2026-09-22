# Score Distribution Diagnostic

Statistics are calculated from inference using the existing models; probabilities were not calibrated or modified.

## Validation

| Model | Class | Count | Mean | Median | Min | Max | P25 | P75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | positives | 2592 | 0.6729725 | 0.83271566 | 2.8373806e-42 | 1 | 0.40651514 | 0.999996 |
| logistic | negatives | 3603 | 0.31527693 | 0.22843769 | 1.0949192e-54 | 1 | 0.020335972 | 0.54612203 |
| lstm | positives | 2592 | 0.49723765 | 0.35353729 | 1.2261297e-06 | 0.99999464 | 0.011023773 | 0.99992561 |
| lstm | negatives | 3603 | 0.071929082 | 0.00020998598 | 1.4573026e-06 | 0.99998963 | 1.0082779e-05 | 0.026800105 |

## Test

| Model | Class | Count | Mean | Median | Min | Max | P25 | P75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | positives | 1763 | 0.80621412 | 0.99640739 | 1.9710046e-33 | 1 | 0.89041186 | 1 |
| logistic | negatives | 4432 | 0.27994648 | 0.13573142 | 5.864556e-120 | 1 | 0.0087505174 | 0.5257253 |
| lstm | positives | 1763 | 0.74947351 | 0.9999094 | 4.7711005e-06 | 0.99999475 | 0.38860157 | 0.99996877 |
| lstm | negatives | 4432 | 0.13669786 | 0.0003986118 | 9.3815947e-07 | 0.99998331 | 1.0657745e-05 | 0.071871452 |
