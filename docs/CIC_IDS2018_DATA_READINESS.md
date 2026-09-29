# CIC-IDS2018 Data Readiness

This Phase 2 report is read-only. Each raw CSV was scanned independently with bounded-memory CSV iteration. No source data was merged, rewritten, sorted, imputed, deleted, moved, or repaired. No model or temporal training window was created.

## Dataset overview

The ten files contain 16,233,002 flow rows across approximately 6.89 GB of raw CSV. Exact header counts: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv: 80, Friday-16-02-2018_TrafficForML_CICFlowMeter.csv: 80, Friday-23-02-2018_TrafficForML_CICFlowMeter.csv: 80, Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv: 84, Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv: 80, Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv: 80, Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv: 80, Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv: 80, Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv: 80, Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv: 80.

## Schema findings

Nine files share the 80-column common schema. `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` has 84 columns and adds `Flow ID`, `Src IP`, `Src Port`, and `Dst IP`. Relative to the 84-column union, the other nine files are missing exactly those four columns. No other header differences were found.

## Label findings

- `Benign` -> `Benign`: 13,484,708 (83.06971194%), files: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv; Friday-16-02-2018_TrafficForML_CICFlowMeter.csv; Friday-23-02-2018_TrafficForML_CICFlowMeter.csv; Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv; Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv; Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv; Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv; Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv.
- `DDOS attack-HOIC` -> `DDoS attack-HOIC`: 686,012 (4.22603287%), files: Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv.
- `DDoS attacks-LOIC-HTTP` -> `DDoS attacks-LOIC-HTTP`: 576,191 (3.54950366%), files: Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv.
- `DoS attacks-Hulk` -> `DoS attacks-Hulk`: 461,912 (2.84551188%), files: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv.
- `Bot` -> `Bot`: 286,191 (1.76301956%), files: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv.
- `FTP-BruteForce` -> `FTP-BruteForce`: 193,360 (1.19115368%), files: Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv.
- `SSH-Bruteforce` -> `SSH-Bruteforce`: 187,589 (1.15560264%), files: Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv.
- `Infilteration` -> `Infiltration`: 161,934 (0.9975604%), files: Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv; Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv.
- `DoS attacks-SlowHTTPTest` -> `DoS attacks-SlowHTTPTest`: 139,890 (0.86176297%), files: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv.
- `DoS attacks-GoldenEye` -> `DoS attacks-GoldenEye`: 41,508 (0.25570132%), files: Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv.
- `DoS attacks-Slowloris` -> `DoS attacks-Slowloris`: 10,990 (0.06770159%), files: Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv.
- `DDOS attack-LOIC-UDP` -> `DDoS attack-LOIC-UDP`: 1,730 (0.0106573%), files: Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv.
- `Brute Force -Web` -> `Brute Force -Web`: 611 (0.00376394%), files: Friday-23-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv.
- `Brute Force -XSS` -> `Brute Force -XSS`: 230 (0.00141687%), files: Friday-23-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv.
- `SQL Injection` -> `SQL Injection`: 87 (0.00053595%), files: Friday-23-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv.
- `Label` -> `Label`: 59 (0.00036346%), files: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv; Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv; Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv.

The repeated header rows appear in the label report as raw label `Label`; they are not attack classes and are excluded from the attack timeline.

## Attack timeline

- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` / `Bot`: 286,191 flows, `2018-03-02 01:00:00` to `2018-03-02 12:59:59` (43199.0 seconds).
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` / `DoS attacks-Hulk`: 461,912 flows, `2018-02-16 01:45:27` to `2018-02-16 01:48:44` (197.0 seconds).
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` / `DoS attacks-SlowHTTPTest`: 139,890 flows, `2018-02-16 10:12:14` to `2018-02-16 10:58:08` (2754.0 seconds).
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` / `Brute Force -Web`: 362 flows, `2018-02-23 09:17:04` to `2018-02-23 11:02:58` (6354.0 seconds).
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` / `Brute Force -XSS`: 151 flows, `2018-02-23 01:01:04` to `2018-02-23 02:10:05` (4141.0 seconds).
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` / `SQL Injection`: 53 flows, `2018-02-23 03:05:22` to `2018-02-23 10:53:21` (28079.0 seconds).
- `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` / `DDoS attacks-LOIC-HTTP`: 576,191 flows, `2018-02-20 01:14:17` to `2018-02-20 11:16:48` (36151.0 seconds).
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` / `Infiltration`: 93,063 flows, `2018-03-01 02:00:00` to `2018-03-01 10:54:59` (32099.0 seconds).
- `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` / `DoS attacks-GoldenEye`: 41,508 flows, `2018-02-15 09:27:42` to `2018-02-15 10:02:59` (2117.0 seconds).
- `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` / `DoS attacks-Slowloris`: 10,990 flows, `2018-02-15 11:00:12` to `2018-02-15 11:42:01` (2509.0 seconds).
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` / `Brute Force -Web`: 249 flows, `2018-02-22 10:13:44` to `2018-02-22 11:23:11` (4167.0 seconds).
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` / `Brute Force -XSS`: 79 flows, `2018-02-22 01:51:39` to `2018-02-22 02:28:40` (2221.0 seconds).
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` / `SQL Injection`: 34 flows, `2018-02-22 01:52:33` to `2018-02-22 11:09:45` (33432.0 seconds).
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` / `FTP-BruteForce`: 193,360 flows, `2018-02-14 10:33:26` to `2018-02-14 12:10:31` (5825.0 seconds).
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` / `SSH-Bruteforce`: 187,589 flows, `2018-02-14 02:01:21` to `2018-02-14 03:32:30` (5469.0 seconds).
- `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` / `DDoS attack-HOIC`: 686,012 flows, `2018-02-21 02:11:08` to `2018-02-21 02:33:29` (1341.0 seconds).
- `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` / `DDoS attack-LOIC-UDP`: 1,730 flows, `2018-02-21 10:08:51` to `2018-02-21 10:43:16` (2065.0 seconds).
- `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` / `Infiltration`: 68,871 flows, `2018-02-28 01:42:00` to `2018-02-28 12:04:59` (37379.0 seconds).

## Timestamp quality

There are 59 invalid timestamps across all files. The requested year-1970 investigation found 14 affected rows: 9 in Thursday-22-02 and 5 in Wednesday-14-02.
These are malformed source values, not a parser-only issue: the raw strings explicitly contain year 1970. They are preserved verbatim in the audit process and are not repaired.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246435: raw `10/01/1970 03:04:26`, parsed value `1970-01-10 03:04:26`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246436: raw `11/01/1970 12:05:36`, parsed value `1970-01-11 12:05:36`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246437: raw `11/01/1970 05:12:30`, parsed value `1970-01-11 05:12:30`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246438: raw `11/01/1970 03:51:32`, parsed value `1970-01-11 03:51:32`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246439: raw `12/01/1970 06:40:49`, parsed value `1970-01-12 06:40:49`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246440: raw `12/01/1970 01:09:53`, parsed value `1970-01-12 01:09:53`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246441: raw `12/01/1970 09:18:52`, parsed value `1970-01-12 09:18:52`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 246717: raw `12/01/1970 09:30:03`, parsed value `1970-01-12 09:30:03`, classified as malformed source timestamp.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` row 248316: raw `12/01/1970 09:30:26`, parsed value `1970-01-12 09:30:26`, classified as malformed source timestamp.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` row 410958: raw `05/01/1970 03:01:17`, parsed value `1970-01-05 03:01:17`, classified as malformed source timestamp.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` row 410959: raw `08/01/1970 07:32:33`, parsed value `1970-01-08 07:32:33`, classified as malformed source timestamp.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` row 410960: raw `12/01/1970 07:17:56`, parsed value `1970-01-12 07:17:56`, classified as malformed source timestamp.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` row 410961: raw `12/01/1970 09:15:10`, parsed value `1970-01-12 09:15:10`, classified as malformed source timestamp.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` row 412186: raw `12/01/1970 09:44:12`, parsed value `1970-01-12 09:44:12`, classified as malformed source timestamp.

## Numeric quality

Across the numeric features, the scan found 64,323 invalid numeric strings, 131,799 positive infinity values, and 0 negative infinity values. Blank-cell counts and finite ranges are recorded per feature in `data/audit/numeric_feature_quality.csv`.

- `ACK Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Active Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `114000000.0`.
- `Active Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `114000000.0`.
- `Active Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `114000000.0`.
- `Active Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `75232413.85230836`.
- `Bwd Blk Rate Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Bwd Byts/b Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Bwd Header Len`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `2462372.0`.
- `Bwd IAT Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `120000000.0`.
- `Bwd IAT Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `120000000.0`.
- `Bwd IAT Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `120000000.0`.
- `Bwd IAT Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `84837189.5109486`.
- `Bwd IAT Tot`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `120000000.0`.
- `Bwd PSH Flags`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Bwd Pkt Len Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `65160.0`.
- `Bwd Pkt Len Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `33879.28358`.
- `Bwd Pkt Len Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1460.0`.
- `Bwd Pkt Len Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `22448.413526582273`.
- `Bwd Pkts/b Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Bwd Pkts/s`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `2000000.0`.
- `Bwd Seg Size Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `33879.28358`.
- `Bwd URG Flags`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `CWE Flag Count`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Down/Up Ratio`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `311.0`.
- `Dst Port`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `65535.0`.
- `ECE Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `FIN Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Flow Byts/s`: missing 0, invalid 59,780, +inf 36,039, -inf 0, finite range `0.0` to `1806642857.14286`.
- `Flow Duration`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-919011000000.0` to `120000000.0`.
- `Flow IAT Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-828220000000.0` to `979781000000.0`.
- `Flow IAT Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-828220000000.0` to `120000000.0`.
- `Flow IAT Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-947405000000.0` to `120000000.0`.
- `Flow IAT Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `474354474600.909`.
- `Flow Pkts/s`: missing 0, invalid 59, +inf 95,760, -inf 0, finite range `-0.0088953248` to `6000000.0`.
- `Fwd Act Data Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `309628.0`.
- `Fwd Blk Rate Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Fwd Byts/b Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Fwd Header Len`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `2477032.0`.
- `Fwd IAT Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-828220000000.0` to `979781000000.0`.
- `Fwd IAT Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-828220000000.0` to `120000000.0`.
- `Fwd IAT Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-947405000000.0` to `120000000.0`.
- `Fwd IAT Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `474354474600.909`.
- `Fwd IAT Tot`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-919011000000.0` to `120000000.0`.
- `Fwd PSH Flags`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Fwd Pkt Len Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `64440.0`.
- `Fwd Pkt Len Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `16529.3138401559`.
- `Fwd Pkt Len Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1460.0`.
- `Fwd Pkt Len Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `18401.5827717299`.
- `Fwd Pkts/b Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `0.0`.
- `Fwd Pkts/s`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `6000000.0`.
- `Fwd Seg Size Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `16529.3138401559`.
- `Fwd Seg Size Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `56.0`.
- `Fwd URG Flags`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Idle Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `979781000000.0`.
- `Idle Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `395571421052.631`.
- `Idle Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `239934000000.0`.
- `Idle Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `262247866338.599`.
- `Init Bwd Win Byts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-1.0` to `65535.0`.
- `Init Fwd Win Byts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `-1.0` to `65535.0`.
- `PSH Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Pkt Len Max`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `65160.0`.
- `Pkt Len Mean`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `17344.98473`.
- `Pkt Len Min`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1460.0`.
- `Pkt Len Std`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `22788.28621`.
- `Pkt Len Var`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `519000000.0`.
- `Pkt Size Avg`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `17478.40769`.
- `Protocol`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `17.0`.
- `RST Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `SYN Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.
- `Src Port`: missing 0, invalid 0, +inf 0, -inf 0, finite range `0.0` to `65535.0`.
- `Subflow Bwd Byts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `156360426.0`.
- `Subflow Bwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `123118.0`.
- `Subflow Fwd Byts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `144391846.0`.
- `Subflow Fwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `1.0` to `309629.0`.
- `Tot Bwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `123118.0`.
- `Tot Fwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `1.0` to `309629.0`.
- `TotLen Bwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `156360426.0`.
- `TotLen Fwd Pkts`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `144391846.0`.
- `URG Flag Cnt`: missing 0, invalid 59, +inf 0, -inf 0, finite range `0.0` to `1.0`.

## Temporal ordering

- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-03-02 01:00:00` to `2018-03-02 12:59:59`; out-of-order 292,106 (27.85742555%); invalid timestamps 0.
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-16 01:00:32` to `2018-02-16 12:58:24`; out-of-order 268,941 (25.64823689%); invalid timestamps 1.
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-23 01:00:00` to `2018-02-23 12:59:59`; out-of-order 378,363 (36.08354195%); invalid timestamps 0.
- `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-20 01:00:00` to `2018-02-20 12:59:59`; out-of-order 2,977,911 (37.46389998%); invalid timestamps 0.
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-03-01 01:00:00` to `2018-03-01 12:59:59`; out-of-order 37,824 (11.42287656%); invalid timestamps 25.
- `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-15 01:00:00` to `2018-02-15 12:59:59`; out-of-order 382,967 (36.52261402%); invalid timestamps 0.
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `1970-01-10 03:04:26` to `2018-02-22 12:59:59`; out-of-order 380,319 (36.27008082%); invalid timestamps 0.
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `1970-01-05 03:01:17` to `2018-02-14 12:59:59`; out-of-order 289,765 (27.63417018%); invalid timestamps 0.
- `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-21 01:55:46` to `2018-02-21 10:43:21`; out-of-order 4,056 (0.38681067%); invalid timestamps 0.
- `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv`: NO; range `2018-02-28 01:00:00` to `2018-02-28 12:59:59`; out-of-order 237,535 (38.74301913%); invalid timestamps 33.

Ordering is assessed in original row order using valid parsed timestamps only; rows with invalid timestamps are reported separately and do not get silently repositioned.

## Feature policy

- `SAFE_TRAFFIC_FEATURE`: CICFlowMeter-derived traffic measurements, protocol, ports, packet counts, byte counts, durations, rates, flags, sizes, windows, and inter-arrival statistics, subject to numeric-quality checks.
- `POTENTIAL_IDENTIFIER`: `Flow ID`, `Src IP`, and `Dst IP`; these can identify flows or hosts and may enable memorization or environment-specific shortcuts.
- `POTENTIAL_LEAKAGE`: `Label`; it is the supervised target and must not be used as an input feature. `Timestamp` is also leakage-prone if used beyond an explicitly justified temporal split or as a proxy for collection conditions.
- `NEEDS_INVESTIGATION`: `Timestamp`, port semantics, and all identifier-like fields require an explicit split and generalization policy before modeling. `Src Port` is present only in the Tuesday schema and needs schema handling.

## Packet-level gap

The current processed CSVs provide flow-level, CICFlowMeter-derived features. SIH 26153 also requires packet-level features. Packet-level integration is not performed in this phase and remains a separate data-engineering requirement.

## Recommended preprocessing policy

1. Preserve raw files as immutable inputs and keep per-file provenance.
2. Resolve the four-column schema discrepancy explicitly; do not infer absent identifiers as real values.
3. Define a documented policy for repeated header rows and malformed 1970 timestamps before any downstream feature generation.
4. Track blank, invalid, and infinite numeric values separately; choose any later handling only after domain review, without changing raw inputs.
5. Use time-aware, file-aware validation that prevents identifier and collection-time leakage.
6. Address the packet-level requirement before claiming complete SIH 26153 readiness.

## Final decision

NOT READY — ISSUE(S) REQUIRE RESOLUTION
