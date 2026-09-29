# CIC-IDS-2018 Audit

The audit script at `audits/audit_cic_ids2018.py` inspects the ten requested files under `data/raw/CSE-CIC-IDS2018` independently. It reads rows with Python's CSV reader in bounded chunks and writes:

- `reports/cic_ids2018_audit.json`: machine-readable per-file results.
- `reports/cic_ids2018_audit.md`: human-readable summary and findings.

Run it from `NetOracle`:

```powershell
python audits/audit_cic_ids2018.py
```

For a smaller memory window:

```powershell
python audits/audit_cic_ids2018.py --chunk-size 10000
```

The audit checks file presence and size, header/schema differences, row counts, row-width anomalies, repeated header rows, replacement characters, blank cells, timestamp parsing and ranges, label frequencies, and parse/range statistics for representative numeric fields. It does not combine, clean, rewrite, normalize, or delete source data. It does not import `train.py`, alter `train.py`, create model artifacts, or train a model.

## Interpretation boundaries

The audit is descriptive. A numeric parse failure or unusual range is recorded for investigation; it is not automatically repaired or treated as a reason to discard a row. Label names are reported as supplied by each source file. The existing ingestion and feature-extraction behavior should be reviewed separately before any modeling decision is made.