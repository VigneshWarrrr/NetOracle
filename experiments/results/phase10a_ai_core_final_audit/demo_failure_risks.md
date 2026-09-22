# Phase 10A: What could break a clean demo run

Audit date 2026-09-20. Nothing was fixed by this phase; each row lists the risk, how it was established, and the minimal mitigation. RED = must fix before submission, YELLOW = rehearse / disclose.

| ID | Class | Risk | Likelihood | Impact | Mitigation (not implemented) | Finding |
|---|---|---|---|---|---|---|
| D01 | RED | `manage.py runserver`/`check` fails: `ModuleNotFoundError: reportlab` (server app imports it at URL-load time) | Certain on this machine as configured | Whole site down | `pip install -r requirements.txt`; re-run `manage.py check` | O1 |
| D02 | RED | Clean clone/zip has no engine code, no `.pt` checkpoints (gitignored), no `data/windows` (outside repo) -> engine cannot start | Certain if submitted via git/zip | Authoritative page/API return 503/error | Commit core, ship checkpoint+scaler+schema/data bundle | O2 |
| D03 | RED | Explanation panel for the default demo sample shows ten `0.0000` importances | Certain (default sample) | Looks broken on the headline attack case | Fix attribution target/format (needs code change) | I3 |
| D04 | RED | Landing page after login says 'Model offline / Waiting for trained checkpoint / five observation windows' | Certain | Judge concludes model is not running | Fix landing copy/wiring | C2 |
| D05 | RED | Deck/pitch says 'SHAP values'; implemented method is Gradient x Input; SHAP path silently fails | Certain if deck reused | Direct factual contradiction if asked | Correct wording | I2 |
| D06 | YELLOW | First request after start-up blocks ~19 s while the engine loads (and after every autoreload) | Certain on first hit | Looks hung / browser timeout | Warm engine before demo; use `--noreload` | O3 |
| D07 | YELLOW | No `db.sqlite3` (gitignored): need `migrate` + `createsuperuser`; page is staff-only | Certain on fresh setup | Login fails | Prepare demo account | O4 |
| D08 | YELLOW | Different scikit-learn/torch on the demo machine (unpinned requirements): scaler unpickle warning (pickled with 1.9.1) or failure on older releases | Possible | Warnings / rare load error | Pin versions; rehearse on target | D2 |
| D09 | YELLOW | If tensorflow+shap are present on the demo machine the attribution method may silently become SHAP GradientExplainer (untested) | Possible | Different attributions than the audited ones | Keep environment identical or remove silent fallback | I4 |
| D10 | YELLOW | Default sample is always the Bot attack window (p=0.9999, six x COMMAND_AND_CONTROL); benign/other cases only via API `?index=` | Certain | Judge sees a single canned case; questions about live data | Prepare 2-3 API indices (e.g. 1000 benign, 100 attack) and state the input contract | N3 |
| D11 | YELLOW | Page shows 'Phase 9K/9L' jargon and `C:\AKSHAY\...` checkpoint path | Certain | Unpolished / leaks local path | Copy edit | O5 |
| D12 | YELLOW | Two simultaneous first requests on the threaded dev server load two engines (no lock) | Unlikely with one presenter | ~40 s stall, 2x GPU memory | Warm start-up | G2 |
| D13 | YELLOW | Landing-page/deck questions about early warning, per-step risk, unseen attacks, infiltration probability | Likely from judges | Claims exceed evidence | Use claims_evidence_matrix.csv wording | N2/N4/L2 |
| D14 | YELLOW | No GPU on demo machine: works (CPU verified equal within 2.2e-4, same stages); CPU latency not measured | Possible | Slower explanations | Time it on target | G1 |
| D15 | YELLOW | Legacy /dashboard/forecast/ or capture_traffic started by mistake -> heuristic 'forecast' with 'Live model output' badge | Unlikely | Non-authoritative numbers shown | Do not run capture during demo | C3 |

## Things that were checked and did NOT fail

- Engine load, prediction and explanation on GPU; all 6195 test samples score without error.
- Repeat/fresh-instance determinism; CPU fallback gives identical stages (100% of 300).
- 4-thread concurrent requests on a loaded engine: 0 exceptions, 0 mismatches.
- Auth behaviour: anonymous page -> 302, non-staff page -> 302, anonymous API -> 403, bad index -> 400, out-of-range -> 400, valid -> 200.
- Pages `/dashboard/`, `/dashboard/forecast/`, `/dashboard/forecast/authoritative/` render HTTP 200 once reportlab is available (verified with a throw-away shim in a temp directory, not in the repo).

## Suggested rehearsal (not executed by this audit)

1. Fresh venv -> `pip install -r requirements.txt` -> `python manage.py migrate` -> `createsuperuser` -> `manage.py check`.
2. `runserver --noreload`; hit `/api/authoritative-predict/?index=0` once to warm the engine; then open the authoritative page.
3. Rehearse API indices 0 (attack, saturated), 100 (attack), 1000 (benign, p~0.009), and answer the limitation questions using `claims_evidence_matrix.csv`.

