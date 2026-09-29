# Phase 9B: CSE-CIC-IDS2018 Raw PCAP Object Discovery

Bucket: `s3://cse-cic-ids2018/` (region `ca-central-1`). Access method: unauthenticated HTTPS GET to the public ListObjectsV2 REST endpoint (no AWS CLI, no boto3, no credentials).
Bucket accessible: **True**

## 1. Top-level prefixes

- `Original Network Traffic and Log data/`
- `Processed Traffic Data for ML Algorithms/`

## 2. Raw traffic day folders (under 'Original Network Traffic and Log data/')

- `Original Network Traffic and Log data/Friday-02-03-2018/`
- `Original Network Traffic and Log data/Friday-16-02-2018/`
- `Original Network Traffic and Log data/Friday-23-02-2018/`
- `Original Network Traffic and Log data/Thursday-01-03-2018/`
- `Original Network Traffic and Log data/Thursday-15-02-2018/`
- `Original Network Traffic and Log data/Thursday-22-02-2018/`
- `Original Network Traffic and Log data/Tuesday-20-02-2018/`
- `Original Network Traffic and Log data/Wednesday-14-02-2018/` <- target
- `Original Network Traffic and Log data/Wednesday-21-02-2018/`
- `Original Network Traffic and Log data/Wednesday-28-02-2018/`

## 3. Candidate objects for Wednesday-14-02-2018

| Key | Size | Last modified | Storage class | Classification | Under 100MB? |
|---|---:|---|---|---|---|
| `Original Network Traffic and Log data/Wednesday-14-02-2018/logs.zip` | 133.68 MB | 2018-10-10T16:44:20.000Z | STANDARD | log_archive | False |
| `Original Network Traffic and Log data/Wednesday-14-02-2018/pcap.zip` | 37.17 GB | 2018-10-11T12:22:03.000Z | STANDARD | raw_pcap_archive | False |

## 4. Correspondence to Wednesday-14-02-2018

Confidence: **HIGH, by naming convention only**

The raw-traffic prefix contains a folder named exactly 'Wednesday-14-02-2018/', matching the day-of-week and date of our existing 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', under the same S3 bucket/organization (CIC) that produced the processed CSVs.

Verified by content inspection: **False**. No packet or log byte has been read. This is folder-name correspondence only, not a verified flow-level or timestamp-level match.

## 5. Download status

Download performed: **False**

All candidate objects at this prefix meet or exceed the 100 MB tiny-download threshold; per instructions, this script stops after metadata discovery and does not fetch any object body.

## Corroboration: processed CSV prefix

Files under 'Processed Traffic Data for ML Algorithms/' (for cross-checking against our existing 10 local CSVs, listing only):

- `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`
- `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv`
- `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv`
- `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`
- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`
- `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv`
- `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv`
- `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`
- `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv`
- `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv`
