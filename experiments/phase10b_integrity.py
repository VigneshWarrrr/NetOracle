"""Phase 10B: AI-core integrity fingerprint (capture before / after, then compare).

Read-only. Proves that submission remediation did not change the frozen AI core:
  * SHA-256 of the authoritative checkpoint / scaler / Run-1 checkpoint / window CSVs
  * SHA-256 of every frozen source file the authoritative path imports
  * a tree hash of every experiments/results/phase* directory (phase10b_* excluded)
  * a digest of the engine's outputs over ALL test samples (attack-risk score, six
    stage names, stage confidences) and of Gradient x Input attributions on 50
    strided samples

Usage:
    python experiments/phase10b_integrity.py capture before
    python experiments/phase10b_integrity.py capture after
    python experiments/phase10b_integrity.py compare
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
WINDOWS_DIR = REPO_ROOT.parent / "data/windows"
RES = EXPERIMENTS_DIR / "results"
OUT = RES / "phase10b_submission_remediation"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

CKPT = RES / "phase7b_mitre_stage_head/model/best_stage_head.pt"
SCALER = RES / "phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib"
RUN1 = RES / "phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt"

FROZEN_SOURCES = [
    "experiments/inference_engine.py",
    "experiments/phase4_baseline.py",
    "experiments/phase6b_vector_world_model.py",
    "experiments/phase7b_mitre_stage_head.py",
    "experiments/phase7b_stage_targets.py",
    "experiments/phase7c_explainability.py",
    "experiments/world_model_dataset.py",
    "explainability/shap_explainer.py",
    "forecasting/mitre_mapping.py",
    "world_model/inference_service.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tree_hash(root: Path) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(sha(p).encode())
    return {"files": len(files), "tree_sha256": h.hexdigest()}


def capture(label: str) -> dict:
    import torch
    import sklearn
    from inference_engine import NetOracleInferenceEngine

    fp = {"label": label, "captured_unix": time.time()}
    fp["authoritative_artifacts"] = {
        "checkpoint_best_stage_head.pt": sha(CKPT), "scaler.joblib": sha(SCALER), "run1_best_model.pt": sha(RUN1),
    }
    fp["window_csv_sha256"] = {p.name: sha(p) for p in sorted(WINDOWS_DIR.glob("*.csv"))}
    fp["frozen_source_sha256"] = {rel: sha(REPO_ROOT / rel) for rel in FROZEN_SOURCES}
    fp["result_dir_tree_hashes"] = {d.name: tree_hash(d) for d in sorted(RES.iterdir()) if d.is_dir() and not d.name.startswith("phase10b")}
    fp["environment"] = {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.cuda.is_available()}

    e = NetOracleInferenceEngine()
    n = e.test_sample_count()
    probs = np.zeros(n, dtype=np.float64)
    conf = np.zeros((n, 6), dtype=np.float64)
    stages = []
    for i in range(n):
        s = e.get_test_sample(i)
        r = e.predict(s["x_raw"], s["source_file"], s["window_start"], include_explanations=False)
        probs[i] = r["whole_horizon_attack_probability"]["value"]
        conf[i] = r["mitre_stage_trajectory"]["per_step_confidence"]
        stages.append(r["mitre_stage_trajectory"]["per_step_stage"])
    frozen = {}
    with (RES / "phase7b_mitre_stage_head/predictions_test.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frozen[(row["source_file"], row["window_start"])] = row
    diffs, mism = [], 0
    for i in range(n):
        s = e.get_test_sample(i)
        row = frozen[(s["source_file"], s["window_start"])]
        diffs.append(abs(probs[i] - float(row["attack_probability"])))
        mism += sum(a != row[f"mitre_stage_step_{k+1}"] for k, a in enumerate(stages[i]))
    fp["engine_outputs_all_test"] = {
        "n": n,
        "prob_digest_sha256": hashlib.sha256(probs.tobytes()).hexdigest(),
        "stage_digest_sha256": hashlib.sha256("|".join(",".join(s) for s in stages).encode()).hexdigest(),
        "confidence_digest_sha256": hashlib.sha256(conf.tobytes()).hexdigest(),
        "max_abs_prob_diff_vs_frozen_phase7b_predictions": float(max(diffs)),
        "stage_cells_mismatching_frozen_phase7b": int(mism),
    }
    idx = list(range(0, n, max(1, n // 50)))[:50]
    att, methods = hashlib.sha256(), set()
    for i in idx:
        s = e.get_test_sample(i)
        r = e.predict(s["x_raw"], s["source_file"], s["window_start"], top_k=10)
        ex = r["explanations"]
        att.update(np.asarray(ex["temporal_evidence"]["attack_risk_attribution"], dtype=np.float64).tobytes())
        att.update(np.asarray(ex["temporal_evidence"]["stage_attribution"], dtype=np.float64).tobytes())
        methods.add(ex["future_attack_risk_prediction"]["explanation_method"])
        methods.add(ex["mitre_stage_prediction"]["explanation_method"])
    fp["explanations_50_strided"] = {"n": len(idx), "attribution_digest_sha256": att.hexdigest(), "methods": sorted(methods)}
    s0 = e.get_test_sample(0)
    r0 = e.predict(s0["x_raw"], s0["source_file"], s0["window_start"], top_k=5)
    fp["sample_0"] = {
        "source_file": s0["source_file"], "window_start": s0["window_start"],
        "attack_risk_score": r0["whole_horizon_attack_probability"]["value"],
        "stages": r0["mitre_stage_trajectory"]["per_step_stage"],
        "top5_attack_features": [(f["feature"], f["importance"]) for f in r0["explanations"]["current_state_evidence"]["top_attack_risk_features"]],
        "explanation_method": r0["explanations"]["future_attack_risk_prediction"]["explanation_method"],
    }
    fp["engine_provenance_hashes"] = {k: r0["provenance"][k] for k in ("checkpoint_sha256", "scaler_sha256", "feature_schema_sha256", "feature_count")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"integrity_{label}.json").write_text(json.dumps(fp, indent=2, default=str), encoding="utf-8")
    return fp


def compare() -> dict:
    b = json.loads((OUT / "integrity_before.json").read_text(encoding="utf-8"))
    a = json.loads((OUT / "integrity_after.json").read_text(encoding="utf-8"))
    checks = {}
    for key in ("authoritative_artifacts", "window_csv_sha256", "frozen_source_sha256", "result_dir_tree_hashes", "environment",
                "engine_outputs_all_test", "explanations_50_strided", "sample_0", "engine_provenance_hashes"):
        checks[key] = (b[key] == a[key])
    # independent references (not produced by this phase)
    ev10a = json.loads((RES / "phase10a_ai_core_final_audit/evidence.json").read_text(encoding="utf-8"))
    ref = ev10a["full_test_reproduction"]
    c9k = json.loads((RES / "phase9k_integration/inference_contract.json").read_text(encoding="utf-8"))
    hashes10a = ev10a["artifact_integrity"]["current_sha256"]
    checks["matches_phase10a_max_abs_prob_diff_exactly"] = a["engine_outputs_all_test"]["max_abs_prob_diff_vs_frozen_phase7b_predictions"] == ref["max_abs_prob_diff_vs_phase7b"]
    checks["matches_phase10a_stage_mismatch_count"] = a["engine_outputs_all_test"]["stage_cells_mismatching_frozen_phase7b"] == round((1 - ref["stage_step_exact_match_rate"]) * ref["stage_steps_compared"])
    checks["sample0_score_matches_phase9k_contract_within_1e-6"] = abs(a["sample_0"]["attack_risk_score"] - c9k["whole_horizon_attack_probability"]["value"]) <= 1e-6
    checks["sample0_stages_match_phase9k_contract_exactly"] = a["sample_0"]["stages"] == c9k["mitre_stage_trajectory"]["per_step_stage"]
    checks["checkpoint_hash_matches_phase10a"] = a["authoritative_artifacts"]["checkpoint_best_stage_head.pt"] == hashes10a["phase7b_stage_head_checkpoint"]
    checks["scaler_hash_matches_phase10a"] = a["authoritative_artifacts"]["scaler.joblib"] == hashes10a["phase6b_run1_scaler"]
    checks["explanation_method_still_gradient_x_input"] = a["explanations_50_strided"]["methods"] == ["Gradient × Input"]
    diffs = {}
    for k in ("frozen_source_sha256", "result_dir_tree_hashes", "window_csv_sha256"):
        d = [n for n in set(b[k]) | set(a[k]) if b[k].get(n) != a[k].get(n)]
        if d:
            diffs[k] = sorted(d)
    res = {"all_identical": all(checks.values()), "checks": checks, "differences": diffs}
    (OUT / "integrity_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        r = capture(sys.argv[2])
        print(json.dumps({"captured": sys.argv[2], "checkpoint": r["authoritative_artifacts"]["checkpoint_best_stage_head.pt"][:16],
                          "prob_digest": r["engine_outputs_all_test"]["prob_digest_sha256"][:16], "sample0": r["sample_0"]["attack_risk_score"]}, indent=1))
    else:
        r = compare()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["all_identical"] else 1)
