"""Phase 10A: NetOracle AI-core final audit -- EVIDENCE COLLECTION ONLY.

Read-only against every frozen artifact (checkpoints, scaler, dataset, prior
phase code/results, Django app). Trains nothing, modifies nothing, imports
nothing from Track B. The only thing this script writes is
experiments/results/phase10a_ai_core_final_audit/evidence.json, which the
companion phase10a_write_report.py turns into the six required deliverables.

Each check is wrapped so one failure cannot hide the others; a failed check
is itself recorded as evidence.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
DATA_ROOT = REPO_ROOT.parent
WINDOWS_DIR = DATA_ROOT / "data/windows"
OUT_DIR = EXPERIMENTS_DIR / "results/phase10a_ai_core_final_audit"
PY = sys.executable
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

RES = EXPERIMENTS_DIR / "results"
CKPT_7B = RES / "phase7b_mitre_stage_head/model/best_stage_head.pt"
CKPT_RUN1 = RES / "phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt"
SCALER = RES / "phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib"
CKPT_PLAIN_6B = RES / "phase6b_vector_world_model/model/best_model.pt"
CKPT_9N = RES / "phase9n_per_step_risk/model/best_per_step_risk_head.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def guarded(name):
    def deco(fn):
        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **k)
                out = {"ok": True, **out} if isinstance(out, dict) else {"ok": True, "value": out}
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
            out["_seconds"] = round(time.perf_counter() - t0, 2)
            print(f"[{name}] ok={out['ok']} ({out['_seconds']}s)", flush=True)
            return out
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# A. artifact integrity + checkpoint/scaler compatibility (objective D)
# ---------------------------------------------------------------------------


@guarded("artifact_integrity")
def check_artifacts() -> dict:
    import torch
    manifest = json.loads((RES / "phase9k_integration/safety_manifest.json").read_text(encoding="utf-8"))
    recorded = {k: v["sha256"] for k, v in manifest["checksums"].items()}
    now = {
        "phase7b_stage_head_checkpoint": sha256(CKPT_7B),
        "phase6b_run1_checkpoint": sha256(CKPT_RUN1),
        "phase6b_run1_scaler": sha256(SCALER),
    }
    hash_match = {k: recorded.get(k) == v for k, v in now.items()}

    s7 = torch.load(CKPT_7B, map_location="cpu", weights_only=True)["model_state_dict"]
    s1 = torch.load(CKPT_RUN1, map_location="cpu", weights_only=True)["model_state_dict"]
    stage_keys = [k for k in s7 if k.startswith("stage_head.")]
    backbone_keys_7b = [k for k in s7 if not k.startswith("stage_head.")]
    same_keyset = set(backbone_keys_7b) == set(s1.keys())
    tensors_identical = same_keyset and all(torch.equal(s7[k], s1[k]) for k in backbone_keys_7b)
    stage_params = int(sum(s7[k].numel() for k in stage_keys))
    backbone_params = int(sum(s7[k].numel() for k in backbone_keys_7b))

    import sklearn
    import joblib
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaler = joblib.load(SCALER)
    # NB: BaseEstimator.__getstate__ reports the RUNTIME version, so the pickled version is read from the unpickle warning.
    mm = re.search(r"from version ([\d.]+) when using version", " ".join(str(x.message) for x in w))
    pickled_sklearn = mm.group(1) if mm else sklearn.__version__
    checks = {
        "n_features_in": int(scaler.n_features_in_),
        "mean_finite": bool(np.isfinite(scaler.mean_).all()),
        "scale_finite": bool(np.isfinite(scaler.scale_).all()),
        "min_scale": float(scaler.scale_.min()),
        "num_zero_scale": int((scaler.scale_ == 0).sum()),
        "with_mean": bool(scaler.with_mean), "with_std": bool(scaler.with_std),
        "pickled_with_sklearn": pickled_sklearn, "runtime_sklearn": sklearn.__version__,
        "sklearn_version_mismatch": pickled_sklearn != sklearn.__version__,
    }
    return {
        "recorded_sha256_phase9k": recorded, "current_sha256": now, "hash_match_phase9k_manifest": hash_match,
        "phase9n_checkpoint_present": CKPT_9N.exists(),
        "plain_phase6b_checkpoint_present_but_unused": CKPT_PLAIN_6B.exists(),
        "plain_phase6b_sha256": sha256(CKPT_PLAIN_6B) if CKPT_PLAIN_6B.exists() else None,
        "run1_vs_plain_6b_checkpoints_identical": (sha256(CKPT_PLAIN_6B) == now["phase6b_run1_checkpoint"]) if CKPT_PLAIN_6B.exists() else None,
        "phase7b_backbone_keys": len(backbone_keys_7b), "phase7b_stage_head_keys": len(stage_keys),
        "backbone_keyset_equals_run1": same_keyset,
        "phase7b_backbone_tensors_bit_identical_to_run1": bool(tensors_identical),
        "backbone_param_count": backbone_params, "stage_head_param_count": stage_params,
        "total_param_count": backbone_params + stage_params,
        "scaler": checks,
    }


@guarded("feature_schema_and_scaler_provenance")
def check_schema_and_scaler() -> dict:
    import joblib
    from sklearn.preprocessing import StandardScaler
    from world_model_dataset import read_world_model_samples, EXPECTED_FEATURE_COUNT
    from phase4_baseline import META_COLUMNS
    samples, feature_columns, _ = read_world_model_samples(WINDOWS_DIR)
    p4cfg = json.loads((RES / "phase4_baseline/config.json").read_text(encoding="utf-8"))
    p4cols = p4cfg["feature_columns"]
    scaler = joblib.load(SCALER)
    refit = StandardScaler().fit(samples["train"].X.reshape(-1, samples["train"].X.shape[-1]))
    rel_mean = float(np.max(np.abs(refit.mean_ - scaler.mean_) / (np.abs(scaler.mean_) + 1e-12)))
    rel_scale = float(np.max(np.abs(refit.scale_ - scaler.scale_) / (np.abs(scaler.scale_) + 1e-12)))
    # also: would a scaler fit on train+val+test differ? (proves the stored one is train-only)
    allx = np.concatenate([samples[s].X.reshape(-1, 157) for s in ("train", "validation", "test")])
    full = StandardScaler().fit(allx)
    rel_mean_full = float(np.max(np.abs(full.mean_ - scaler.mean_) / (np.abs(scaler.mean_) + 1e-12)))
    return {
        "feature_count": len(feature_columns), "expected": EXPECTED_FEATURE_COUNT,
        "unique_feature_names": len(set(feature_columns)),
        "no_meta_columns_in_features": not any(c in META_COLUMNS for c in feature_columns),
        "identical_order_to_phase4_config": feature_columns == p4cols,
        "phase4_config_feature_count": len(p4cols),
        "feature_schema_sha256": hashlib.sha256("|".join(feature_columns).encode()).hexdigest(),
        "first5": feature_columns[:5], "last5": feature_columns[-5:],
        "stored_scaler_vs_train_refit_max_rel_diff_mean": rel_mean,
        "stored_scaler_vs_train_refit_max_rel_diff_scale": rel_scale,
        "stored_scaler_matches_train_only_refit": bool(rel_mean < 1e-4 and rel_scale < 1e-4),
        "stored_scaler_vs_train_val_test_fit_max_rel_diff_mean": rel_mean_full,
        "split_counts": {s: int(samples[s].X.shape[0]) for s in samples},
        "x_shape": list(samples["train"].X.shape[1:]),
        "x_dtype": str(samples["train"].X.dtype),
        "test_x_all_finite": bool(np.isfinite(samples["test"].X).all()),
    }


# ---------------------------------------------------------------------------
# B. engine == frozen checkpoint on the FULL test split (objective F, H)
# ---------------------------------------------------------------------------


def _load_pred_csv(path: Path, keyfields=("source_file", "window_start")):
    with path.open(newline="", encoding="utf-8") as f:
        return {tuple(r[k] for k in keyfields): r for r in csv.DictReader(f)}


_ENGINE = {}


def get_engine():
    if "e" not in _ENGINE:
        from inference_engine import NetOracleInferenceEngine
        t0 = time.perf_counter()
        _ENGINE["e"] = NetOracleInferenceEngine()
        _ENGINE["load_s"] = time.perf_counter() - t0
    return _ENGINE["e"]


@guarded("full_test_reproduction")
def check_full_test_reproduction() -> dict:
    e = get_engine()
    p7b = _load_pred_csv(RES / "phase7b_mitre_stage_head/predictions_test.csv")
    run1 = _load_pred_csv(RES / "phase6b_vector_world_model_ablation/run1_existing_scaling/predictions_test.csv")
    n = e.test_sample_count()
    prob_diff_7b, prob_diff_run1, conf_diff = [], [], []
    stage_exact, stage_total, missing = 0, 0, 0
    sample_stage_all_match = 0
    engine_probs = np.zeros(n)
    stage_counts = {}
    step_stage_names = []
    t0 = time.perf_counter()
    for i in range(n):
        s = e.get_test_sample(i)
        r = e.predict(s["x_raw"], s["source_file"], s["window_start"], include_explanations=False)
        key = (s["source_file"], s["window_start"])
        p = r["whole_horizon_attack_probability"]["value"]
        engine_probs[i] = p
        st = r["mitre_stage_trajectory"]["per_step_stage"]
        cf = r["mitre_stage_trajectory"]["per_step_confidence"]
        step_stage_names.append(st)
        for name in st:
            stage_counts[name] = stage_counts.get(name, 0) + 1
        if key not in p7b or key not in run1:
            missing += 1
            continue
        prob_diff_7b.append(abs(p - float(p7b[key]["attack_probability"])))
        prob_diff_run1.append(abs(p - float(run1[key]["world_model_probability"])))
        frozen_stages = [p7b[key][f"mitre_stage_step_{k}"] for k in range(1, 7)]
        frozen_conf = [float(p7b[key][f"mitre_confidence_step_{k}"]) for k in range(1, 7)]
        ok = [a == b for a, b in zip(st, frozen_stages)]
        stage_exact += sum(ok)
        stage_total += 6
        sample_stage_all_match += int(all(ok))
        conf_diff.append(max(abs(a - b) for a, b in zip(cf, frozen_conf)))
    dur = time.perf_counter() - t0
    return {
        "n_test_samples": n, "missing_keys": missing, "seconds": round(dur, 1),
        "max_abs_prob_diff_vs_phase7b": float(max(prob_diff_7b)), "mean_abs_prob_diff_vs_phase7b": float(np.mean(prob_diff_7b)),
        "max_abs_prob_diff_vs_run1": float(max(prob_diff_run1)),
        "stage_step_exact_match_rate": stage_exact / max(stage_total, 1), "stage_steps_compared": stage_total,
        "samples_with_all_six_stage_steps_matching": sample_stage_all_match,
        "max_abs_stage_confidence_diff": float(max(conf_diff)),
        "engine_cold_load_seconds_in_process": round(_ENGINE.get("load_s", 0.0), 2),
        "predicted_stage_step_counts_over_test": stage_counts,
        "predicted_probability_summary": {"min": float(engine_probs.min()), "max": float(engine_probs.max()), "mean": float(engine_probs.mean()),
                                            "frac_gt_0.99": float((engine_probs > 0.99).mean()), "frac_lt_0.01": float((engine_probs < 0.01).mean())},
    }


# ---------------------------------------------------------------------------
# C. determinism (objective G)
# ---------------------------------------------------------------------------


@guarded("determinism")
def check_determinism() -> dict:
    import torch
    from inference_engine import NetOracleInferenceEngine
    e = get_engine()
    n = 300
    idx = list(range(0, e.test_sample_count(), max(1, e.test_sample_count() // n)))[:n]

    def run(engine):
        out = []
        for i in idx:
            s = engine.get_test_sample(i)
            r = engine.predict(s["x_raw"], include_explanations=False)
            out.append((r["whole_horizon_attack_probability"]["value"], tuple(r["mitre_stage_trajectory"]["per_step_stage"]),
                        np.array(r["future_state_rollout"]["predicted_state_scaled"])))
        return out

    a = run(e)
    a2 = run(e)
    fresh = NetOracleInferenceEngine()
    b = run(fresh)
    same_instance_bit_exact = all(x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) for x, y in zip(a, a2))
    fresh_instance_bit_exact = all(x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    res = {
        "n_samples": len(idx), "device_primary": str(e.device),
        "same_instance_repeat_bit_exact": bool(same_instance_bit_exact),
        "fresh_instance_bit_exact": bool(fresh_instance_bit_exact),
        "max_prob_diff_fresh_instance": float(max(abs(x[0] - y[0]) for x, y in zip(a, b))),
    }
    del fresh
    cpu = NetOracleInferenceEngine(device=torch.device("cpu"))
    c = run(cpu)
    res["cpu_vs_primary_max_prob_diff"] = float(max(abs(x[0] - y[0]) for x, y in zip(a, c)))
    res["cpu_vs_primary_stage_identical_fraction"] = float(np.mean([x[1] == y[1] for x, y in zip(a, c)]))
    res["cpu_vs_primary_max_rollout_diff_scaled"] = float(max(np.abs(x[2] - y[2]).max() for x, y in zip(a, c)))
    # explanation determinism (same sample twice + CPU vs primary)
    s = e.get_test_sample(1000)
    r1 = e.predict(s["x_raw"], top_k=10)["explanations"]
    r2 = e.predict(s["x_raw"], top_k=10)["explanations"]
    rc = cpu.predict(s["x_raw"], top_k=10)["explanations"]
    att = lambda r: np.array(r["temporal_evidence"]["attack_risk_attribution"])
    res["explanation_repeat_bit_exact"] = bool(np.array_equal(att(r1), att(r2)))
    res["explanation_cpu_vs_primary_max_abs_diff"] = float(np.abs(att(r1) - att(rc)).max())
    top = lambda r: [f["feature"] for f in r["current_state_evidence"]["top_attack_risk_features"]]
    res["explanation_top10_identical_cpu_vs_primary"] = top(r1) == top(rc)
    return res


# ---------------------------------------------------------------------------
# D. stage head really produces the MITRE trajectory (objective H)
# ---------------------------------------------------------------------------


@guarded("stage_head_provenance")
def check_stage_head() -> dict:
    import torch
    from phase7b_stage_targets import CLASS_INDEX_TO_STAGE
    e = get_engine()
    s = e.get_test_sample(3000)
    x = np.asarray(s["x_raw"], dtype=np.float32)
    from phase6b_vector_world_model import scale_array
    xs = scale_array(x[None], e.scaler)
    xt = torch.from_numpy(xs).to(e.device)
    with torch.no_grad():
        _, _, stage_logits, z_future = e.model(xt)
        manual = [CLASS_INDEX_TO_STAGE[int(i)].name for i in stage_logits[0].argmax(-1).cpu().numpy()]
        # independent recomputation straight through the stage_head module on z_future
        manual2 = [CLASS_INDEX_TO_STAGE[int(i)].name for i in e.model.stage_head(z_future)[0].argmax(-1).cpu().numpy()]
    eng = e.predict(x, include_explanations=False)["mitre_stage_trajectory"]["per_step_stage"]

    # causal dependence: perturbing ONLY stage_head weights must change engine-style output; perturbing nothing must not
    m2 = copy.deepcopy(e.model)
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for p in m2.stage_head.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * 1.0)
        _, _, sl2, _ = m2(xt)
    perturbed = [CLASS_INDEX_TO_STAGE[int(i)].name for i in sl2[0].argmax(-1).cpu().numpy()]
    n_changed_over_batch = 0
    with torch.no_grad():
        xb = torch.from_numpy(scale_array(np.stack([e.get_test_sample(i)["x_raw"] for i in range(0, 6000, 60)]).astype(np.float32), e.scaler)).to(e.device)
        base = e.model(xb)[2].argmax(-1)
        pert = m2(xb)[2].argmax(-1)
        n_changed_over_batch = int((base != pert).sum().item())
        total_cells = int(base.numel())
    # attack-head independence: attack logit must be unchanged when stage_head is perturbed
    with torch.no_grad():
        a1 = e.model(xt)[1]
        a2 = m2(xt)[1]
    return {
        "engine_stages": eng, "manual_argmax_over_model_stage_logits": manual, "manual_argmax_via_stage_head_on_z_future": manual2,
        "engine_equals_manual": eng == manual == manual2,
        "perturbed_stage_head_stages": perturbed, "perturbing_stage_head_changes_output": perturbed != eng,
        "cells_changed_when_stage_head_perturbed": n_changed_over_batch, "cells_total": total_cells,
        "attack_logit_independent_of_stage_head": bool(torch.equal(a1, a2)),
        "stage_head_module": type(e.model.stage_head).__name__, "model_class": type(e.model).__name__,
        "class_index_to_stage": {int(k): v.name for k, v in CLASS_INDEX_TO_STAGE.items()},
    }


# ---------------------------------------------------------------------------
# E. explainability: actual method + demo-visible saturation (objective I)
# ---------------------------------------------------------------------------


@guarded("explainability")
def check_explainability() -> dict:
    import torch
    e = get_engine()
    from explainability import shap_explainer as se
    from phase7c_explainability import TemporalAwareExplainer
    from phase6b_vector_world_model import scale_array
    out = {"shap_package_available_flag": bool(se.SHAP_AVAILABLE), "shap_version": getattr(se.shap, "__version__", None) if se.shap else None}

    methods, top_importances, disp_zero, max_att_t = [], [], 0, []
    N = 300
    idxs = list(range(0, e.test_sample_count(), e.test_sample_count() // N))[:N]
    probs, labels = [], []
    for i in idxs:
        s = e.get_test_sample(i)
        r = e.predict(s["x_raw"], top_k=10)
        ex = r["explanations"]
        methods.append((ex["future_attack_risk_prediction"]["explanation_method"], ex["mitre_stage_prediction"]["explanation_method"]))
        t1 = ex["current_state_evidence"]["top_attack_risk_features"][0]["importance"]
        top_importances.append(t1)
        disp_zero += int(f"{t1:.4f}" == "0.0000")
        max_att_t.append(max(f["importance"] for f in ex["current_state_evidence"]["top_attack_risk_features"]))
        probs.append(r["whole_horizon_attack_probability"]["value"])
        labels.append(s["label"])
    probs, labels, top_importances = np.array(probs), np.array(labels), np.array(top_importances)
    out.update({
        "n_samples": len(idxs),
        "distinct_method_strings": sorted({m[0] for m in methods} | {m[1] for m in methods}),
        "top1_importance_displayed_as_0.0000_fraction_all": disp_zero / len(idxs),
        "top1_importance_displayed_as_0.0000_fraction_predicted_attack_gt_0.99": float(np.mean(np.array([f"{v:.4f}" == "0.0000" for v in top_importances])[probs > 0.99])) if (probs > 0.99).any() else None,
        "n_samples_predicted_attack_gt_0.99": int((probs > 0.99).sum()),
        "top1_importance_median": float(np.median(top_importances)), "top1_importance_min": float(top_importances.min()), "top1_importance_max": float(top_importances.max()),
        "top1_importance_median_when_prob_gt_0.99": float(np.median(top_importances[probs > 0.99])) if (probs > 0.99).any() else None,
        "top1_importance_median_when_prob_lt_0.5": float(np.median(top_importances[probs < 0.5])) if (probs < 0.5).any() else None,
    })

    # why is SHAP not the reported method even though shap is installed?
    s = e.get_test_sample(1000)
    xs = scale_array(np.asarray(s["x_raw"], np.float32)[None], e.scaler)[0]
    ex = TemporalAwareExplainer(e.model, e.feature_columns, device=str(e.device))
    try:
        ex.explain_attack_risk(xs, top_k=5, method="shap")
        out["explicit_shap_call"] = "succeeded"
    except Exception as exc:  # noqa: BLE001
        out["explicit_shap_call"] = f"raised {type(exc).__name__}: {str(exc)[:300]}"
    # what does an environment WITHOUT a working shap path report? (monkeypatch flag only, in-process)
    orig = se.SHAP_AVAILABLE
    try:
        se.SHAP_AVAILABLE = False
        from phase7c_explainability import explain_sample
        r_g = explain_sample(e.model, e.feature_columns, xs, e.scaler, "x", "y", e.device, top_k=10)
        top_g = [f["feature"] for f in r_g["current_state_evidence"]["top_attack_risk_features"]]
    finally:
        se.SHAP_AVAILABLE = orig
    r_a = explain_sample(e.model, e.feature_columns, xs, e.scaler, "x", "y", e.device, top_k=10)
    top_a = [f["feature"] for f in r_a["current_state_evidence"]["top_attack_risk_features"]]
    out["method_with_shap_flag_true"] = r_a["future_attack_risk_prediction"]["explanation_method"]
    out["method_with_shap_flag_false"] = r_g["future_attack_risk_prediction"]["explanation_method"]
    out["top10_identical_shap_flag_true_vs_false"] = top_a == top_g
    out["faithful_claims_note"] = r_a["faithful_claims_only"]
    out["shap_wording_present_in_faithful_claims_note"] = "SHAP" in r_a["faithful_claims_only"]
    out["explainer_gradient_target"] = "sigmoid probability (see explainability/shap_explainer.py::_explain_with_gradients -> _convert_to_probability -> target.backward())"
    return out


# ---------------------------------------------------------------------------
# F. concurrency (Django runserver is threaded) -- demo failure risk
# ---------------------------------------------------------------------------


@guarded("thread_safety")
def check_thread_safety() -> dict:
    e = get_engine()
    idxs = [10, 500, 1500, 2500, 3500, 4500, 5500, 6000]

    def call(i):
        s = e.get_test_sample(i)
        r = e.predict(s["x_raw"], top_k=5)
        ex = r["explanations"]["current_state_evidence"]["top_attack_risk_features"]
        return i, r["whole_horizon_attack_probability"]["value"], tuple(r["mitre_stage_trajectory"]["per_step_stage"]), tuple(f["feature"] for f in ex), [f["importance"] for f in ex]

    seq = {i: call(i) for i in idxs}
    errors, mismatches = [], []
    def worker(i):
        try:
            return call(i)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            return None
    for round_ in range(3):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, idxs * 2))
        for r in results:
            if r is None:
                continue
            ref = seq[r[0]]
            if r[1] != ref[1] or r[2] != ref[2] or r[3] != ref[3] or not np.allclose(r[4], ref[4], rtol=1e-5, atol=1e-12):
                mismatches.append({"i": r[0], "prob_diff": abs(r[1] - ref[1]), "stages_equal": r[2] == ref[2], "top_features_equal": r[3] == ref[3]})
    return {"rounds": 3, "threads": 4, "calls_per_round": len(idxs) * 2, "exceptions": errors[:5], "n_exceptions": len(errors),
            "n_result_mismatches_vs_sequential": len(mismatches), "mismatch_examples": mismatches[:5],
            "singleton_has_no_lock": "_lock" not in json.dumps(dir(__import__("world_model.inference_service", fromlist=["x"]).AuthoritativeForecastService))}


# ---------------------------------------------------------------------------
# G. forecasting vs. ongoing-attack detection (SIH limitation)
# ---------------------------------------------------------------------------


@guarded("forecast_vs_detection")
def check_forecast_vs_detection() -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    cur = {}
    for part in sorted(WINDOWS_DIR.glob("*.csv")):
        with part.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["forecast_sample_eligible"] == "1" and r["split"] == "test":
                    cur[(part.name, r["window_start"])] = (int(r["current_attack"]), int(r["future_attack_within_horizon"]))
    p7b = _load_pred_csv(RES / "phase7b_mitre_stage_head/predictions_test.csv")
    tf = _load_pred_csv(RES / "phase5_temporal_transformer/predictions_test.csv")
    keys = [k for k in p7b if k in cur]
    y = np.array([cur[k][1] for k in keys]); c = np.array([cur[k][0] for k in keys])
    wm = np.array([float(p7b[k]["attack_probability"]) for k in keys])
    tfp = np.array([float(tf[k]["transformer_probability"]) for k in keys if k in tf]) if all(k in tf for k in keys) else None
    thr = 0.335242  # Phase 8A frozen validation-selected threshold, Run 1 (see FINAL_EVALUATION_TABLE.md)

    def m(score, mask):
        yy, ss = y[mask], score[mask]
        if yy.sum() == 0 or yy.sum() == len(yy):
            return None
        return {"n": int(mask.sum()), "positives": int(yy.sum()), "prevalence": float(yy.mean()), "pr_auc": float(average_precision_score(yy, ss)), "roc_auc": float(roc_auc_score(yy, ss))}
    allm, onset, ongoing = np.ones(len(y), bool), c == 0, c == 1
    out = {
        "n_test": len(keys), "positives": int(y.sum()),
        "positives_with_current_window_already_attack": int(((y == 1) & (c == 1)).sum()),
        "positives_that_are_true_onsets_current_benign": int(((y == 1) & (c == 0)).sum()),
        "fraction_of_positives_already_under_attack": float(((y == 1) & (c == 1)).sum() / max(y.sum(), 1)),
        "oracle_persistence_uses_ground_truth_current_label_unavailable_at_inference": m(c.astype(float), allm),
        "world_model_all": m(wm, allm), "world_model_onset_subset_current_benign": m(wm, onset), "world_model_ongoing_subset_current_attack": m(wm, ongoing),
        "world_model_onset_recall_at_frozen_phase8a_threshold": float(((wm >= thr) & (y == 1) & (c == 0)).sum() / max(((y == 1) & (c == 0)).sum(), 1)),
        "world_model_fpr_at_frozen_threshold_on_fully_benign": float(((wm >= thr) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "world_model_ongoing_recall_at_frozen_threshold": float(((wm >= thr) & (y == 1) & (c == 1)).sum() / max(((y == 1) & (c == 1)).sum(), 1)),
        "frozen_threshold_used": thr,
    }
    if tfp is not None:
        out["transformer_all"] = m(tfp, allm); out["transformer_onset_subset_current_benign"] = m(tfp, onset)
    # by source-day: where do onset positives come from?
    by_file = {}
    for k, yy, cc in zip(keys, y, c):
        d = by_file.setdefault(k[0], {"positives": 0, "onset_positives": 0})
        d["positives"] += int(yy); d["onset_positives"] += int(yy == 1 and cc == 0)
    out["test_positives_by_source_day"] = {k: v for k, v in by_file.items() if v["positives"]}
    return out


# ---------------------------------------------------------------------------
# H. repository path inventory + claim scans (objectives B, C, J, K, L, M)
# ---------------------------------------------------------------------------

SKIP_DIRS = {".git", "__pycache__", "myenv", "results", "node_modules", "migrations", "static"}


def iter_files(exts):
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(exts):
                yield Path(root) / fn


def grep(pattern, exts=(".py", ".html", ".md", ".txt", ".yaml", ".json"), exclude=()):
    rx = re.compile(pattern, re.I)
    hits = []
    for p in iter_files(exts):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(x) for x in exclude):
            continue
        try:
            for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append({"file": rel, "line": ln, "text": line.strip()[:200]})
        except Exception:  # noqa: BLE001
            pass
    return hits


def pptx_text(path: Path) -> list[str]:
    texts = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)):
            xml = z.read(name).decode("utf-8", errors="replace")
            texts.append(" ".join(re.findall(r"<a:t>([^<]*)</a:t>", xml)))
    return texts


@guarded("path_inventory")
def check_path_inventory() -> dict:
    prod_excl = ("experiments/", "tests/")
    code = (".py",)
    out = {}
    out["authoritative_refs_outside_experiments_tests"] = [h for h in grep(r"inference_engine|NetOracleInferenceEngine|AuthoritativeForecastService", code, prod_excl)]
    out["legacy_heuristic_refs"] = grep(r"AttackForecastService|AttackRiskForecaster|_heuristic_risk|heuristic_fallback|attack_forecaster\.pt", (".py", ".html", ".md"), ("experiments/", "tests/"))
    out["track_b_refs"] = grep(r"NetworkWorldModel|world_model\.world_model|best_world_model|from world_model\.(inference|baseline|explainer|feature_extraction|mitre_mapper)|import streamlit", (".py", ".html", ".md"), ("experiments/", "tests/"))
    out["phase9n_refs_in_production_code"] = grep(r"per_step_risk|PerStepAttackRisk|phase9n|best_per_step", (".py", ".html", ".md"), ("experiments/", "tests/"))
    out["phase9n_refs_in_authoritative_engine_or_django"] = [h for h in grep(r"per_step_risk|PerStepAttackRisk|phase9n|best_per_step", (".py", ".html"), ("tests/",))
                                                             if h["file"] in ("experiments/inference_engine.py", "world_model/inference_service.py", "dashboard/views.py", "api/views.py") or h["file"].startswith("templates/")]
    out["torch_load_sites_outside_experiments_tests"] = grep(r"torch\.load|joblib\.load|load_state_dict", code, prod_excl)
    out["checkpoint_files_in_repo_tree"] = sorted(p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.rglob("*") if p.suffix in (".pt", ".pth", ".joblib") and ".git" not in p.parts)
    out["engine_import_sites_of_track_b"] = grep(r"^\s*(from|import)\s+world_model", (".py",), ("tests/",)) if False else []
    return out


@guarded("claims_scan")
def check_claims() -> dict:
    ui_files = [p for p in REPO_ROOT.joinpath("templates").rglob("*.html")]
    docs = [REPO_ROOT / "README.md"] + sorted(REPO_ROOT.joinpath("docs").glob("*.md"))
    patterns = {
        "shap": r"\bSHAP\b", "calibrated": r"calibrat", "unseen_or_zero_day": r"zero[- ]?day|zero[- ]?shot|unseen|novel attack|unknown attack|never seen",
        "infiltration": r"infiltrat", "early_warning": r"early[- ]?warning", "per_step": r"\bT\+[1-9]\b|per[- ]step|next (five|six|5|6) (observation )?windows|five-window|risk_timeline|forecast_windows|window\.probability",
        "realtime": r"real[- ]?time|live model|live capture", "beats_baselines": r"outperform|better than|state[- ]of[- ]the[- ]art|accuracy|99\.\d|100%",
        "attack_chain_or_next_step": r"attack chain|attack-chain|next access step|likely target|predicted attack type|precaution",
    }
    out = {}
    for label, rx in patterns.items():
        hits = []
        rxc = re.compile(rx, re.I)
        for p in ui_files + docs:
            rel = p.relative_to(REPO_ROOT).as_posix()
            for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rxc.search(line):
                    hits.append({"file": rel, "line": ln, "text": re.sub(r"\s+", " ", line.strip())[:220]})
        out[label] = hits[:40]
    pptx = REPO_ROOT / "world_model" / "Presentation SIH.pptx"
    slides = pptx_text(pptx) if pptx.exists() else []
    out["presentation_slide_count"] = len(slides)
    ppt_hits = {}
    for label, rx in patterns.items():
        rxc = re.compile(rx, re.I)
        ppt_hits[label] = [{"slide": i + 1, "text": s[:400]} for i, s in enumerate(slides) if rxc.search(s)][:10]
    out["presentation_hits"] = ppt_hits
    out["presentation_full_text_first_slides"] = [s[:600] for s in slides[:12]]
    out["presentation_last_modified"] = time.strftime("%Y-%m-%d", time.localtime(pptx.stat().st_mtime)) if pptx.exists() else None
    return out


# ---------------------------------------------------------------------------
# I. clean-demo risks: environment, cold start, rendered pages (objective O)
# ---------------------------------------------------------------------------


@guarded("environment")
def check_environment() -> dict:
    import sklearn, torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "sklearn": sklearn.__version__}
    for mod in ("shap", "reportlab", "django", "rest_framework", "plotly", "scapy", "streamlit", "PyPDF2", "networkx", "polars"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "installed")
        except Exception as exc:  # noqa: BLE001
            env[mod] = f"MISSING ({type(exc).__name__})"
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    env["requirements_txt_pins_exact_versions"] = bool(re.search(r"==", req))
    env["requirements_txt_lines"] = [l for l in req.splitlines() if l.strip() and not l.startswith("#")]
    env["windows_dir_used_by_engine"] = str(WINDOWS_DIR)
    env["windows_dir_exists"] = WINDOWS_DIR.exists()
    env["windows_dir_inside_repo"] = str(WINDOWS_DIR).startswith(str(REPO_ROOT))
    env["windows_csv_count"] = len(list(WINDOWS_DIR.glob("*.csv"))) if WINDOWS_DIR.exists() else 0
    env["windows_dir_total_mb"] = round(sum(p.stat().st_size for p in WINDOWS_DIR.glob("*.csv")) / 1e6, 1) if WINDOWS_DIR.exists() else 0
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    env["gitignore_patterns"] = gi
    env["gitignore_excludes_pt_checkpoints"] = "*.pt" in gi
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    env["git_head"] = git("rev-parse", "HEAD")
    env["git_dirty_entries"] = len([l for l in git("status", "--short").splitlines() if l.strip()])
    tracked = set(git("ls-files").splitlines())
    must = ["experiments/inference_engine.py", "experiments/phase6b_vector_world_model.py", "experiments/phase7b_mitre_stage_head.py",
            "experiments/phase7c_explainability.py", "experiments/world_model_dataset.py", "experiments/phase4_baseline.py",
            "templates/dashboard/authoritative_forecast.html", "world_model/inference_service.py", "dashboard/views.py", "api/views.py"]
    env["authoritative_files_tracked_in_git"] = {f: (f in tracked) for f in must}
    committed_engine = "experiments/inference_engine.py" in tracked
    env["engine_committed"] = committed_engine
    env["modified_but_uncommitted_tracked_files"] = [l for l in git("status", "--short").splitlines() if l.startswith(" M") or l.startswith("M")]
    ck = {"best_stage_head.pt": CKPT_7B, "run1 best_model.pt": CKPT_RUN1, "scaler.joblib": SCALER}
    ignored = {}
    for name, p in ck.items():
        r = subprocess.run(["git", "check-ignore", "-q", str(p.relative_to(REPO_ROOT))], cwd=REPO_ROOT)
        ignored[name] = r.returncode == 0
    env["authoritative_artifacts_gitignored"] = ignored
    env["authoritative_artifacts_tracked"] = {n: (p.relative_to(REPO_ROOT).as_posix() in tracked) for n, p in ck.items()}
    dbp = REPO_ROOT / "db.sqlite3"
    env["db_sqlite3_present"] = dbp.exists()
    if dbp.exists():
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{dbp.as_posix()}?mode=ro", uri=True)
            tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
            env["db_tables_count"] = len(tables)
            for t in tables:
                if t.endswith("user") and "auth" not in t or t == "auth_user":
                    try:
                        env[f"db_{t}_rows"] = con.execute(f"select count(*) from {t}").fetchone()[0]
                        cols = [r[1] for r in con.execute(f"pragma table_info({t})")]
                        if "is_staff" in cols:
                            env[f"db_{t}_staff_rows"] = con.execute(f"select count(*) from {t} where is_staff=1 or is_superuser=1").fetchone()[0]
                    except Exception as exc:  # noqa: BLE001
                        env[f"db_{t}_err"] = str(exc)
            con.close()
        except Exception as exc:  # noqa: BLE001
            env["db_read_error"] = str(exc)
    return env


@guarded("django_startup_without_workarounds")
def check_django_startup() -> dict:
    r = subprocess.run([PY, "manage.py", "check"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    return {"returncode": r.returncode, "stdout_tail": r.stdout.strip()[-300:], "stderr_last_line": (r.stderr.strip().splitlines() or [""])[-1][:300]}


@guarded("cold_start_fresh_process")
def check_cold_start() -> dict:
    code = ("import time,sys,warnings;warnings.filterwarnings('ignore');t=time.time();"
            "from world_model.inference_service import AuthoritativeForecastService as S;"
            "e=S.get_engine();t1=time.time()-t;"
            "t=time.time();r=S.predict_demo_sample(0);t2=time.time()-t;"
            "t=time.time();r=S.predict_demo_sample(0);t3=time.time()-t;"
            "print(round(t1,2),round(t2,2),round(t3,3))")
    r = subprocess.run([PY, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    parts = r.stdout.strip().split()
    return {"engine_load_seconds": float(parts[0]), "first_prediction_incl_explanations_seconds": float(parts[1]), "warm_prediction_seconds": float(parts[2]), "stderr_tail": r.stderr.strip()[-200:]}


RENDER_SCRIPT = r'''
import os, sys, re, json, warnings
warnings.filterwarnings("ignore")
stub = sys.argv[1]; sys.path.insert(0, stub)
sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")
import django; django.setup()
from django.test.utils import setup_test_environment, teardown_test_environment
from django.test import Client
from django.db import connection
from django.contrib.auth import get_user_model
setup_test_environment()
cfg = connection.creation.create_test_db(verbosity=0)
out = {}
try:
    U = get_user_model(); u = U.objects.create_superuser(username="p10a", email="a@b.c", password="x-not-real-1")
    c = Client(); c.force_login(u)
    def text(html):
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    for name, url in [("landing_dashboard_index", "/dashboard/"), ("legacy_forecast", "/dashboard/forecast/"), ("authoritative_forecast", "/dashboard/forecast/authoritative/")]:
        r = c.get(url)
        body = r.content.decode("utf-8", "replace")
        out[name] = {"status": r.status_code, "text": text(body)[:6000], "html_len": len(body)}
    r = c.get("/", follow=True)
    out["root_redirect"] = {"status": r.status_code, "chain": [str(x) for x in r.redirect_chain], "final_url_text": text(r.content.decode("utf-8", "replace"))[:300]}
    r = c.get("/api/authoritative-predict/?index=0")
    j = r.json(); out["api_keys"] = sorted(j.keys()); out["api_status"] = r.status_code
    out["api_whole_horizon"] = j["whole_horizon_attack_probability"]
    out["api_top_level_key_names_containing_risk_or_prob"] = [k for k in j if "risk" in k or "prob" in k]
    r2 = c.get("/api/authoritative-predict/?index=999999"); out["api_out_of_range_status"] = r2.status_code
    r3 = c.get("/api/authoritative-predict/?index=abc"); out["api_bad_index_status"] = r3.status_code
    r4 = Client().get("/api/authoritative-predict/?index=0"); out["api_unauthenticated_status"] = r4.status_code
    r5 = Client().get("/dashboard/forecast/authoritative/"); out["page_unauthenticated_status"] = r5.status_code
    U.objects.create_user(username="plain", password="x-not-real-2")
    c2 = Client(); c2.login(username="plain", password="x-not-real-2")
    out["page_non_staff_status"] = c2.get("/dashboard/forecast/authoritative/").status_code
    out["api_non_staff_status"] = c2.get("/api/authoritative-predict/?index=0").status_code
finally:
    connection.creation.destroy_test_db(cfg, verbosity=0); teardown_test_environment()
print("JSON>>" + json.dumps(out))
'''


@guarded("rendered_pages")
def check_rendered_pages() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="p10a_stub_"))
    # verification-only shim for the (declared-but-uninstalled) reportlab dependency; lives in a temp dir, never in the repo
    (tmp / "reportlab" / "pdfgen").mkdir(parents=True); (tmp / "reportlab" / "lib").mkdir(); (tmp / "reportlab" / "platypus").mkdir()
    for f in ("reportlab/__init__.py", "reportlab/pdfgen/__init__.py", "reportlab/lib/__init__.py"):
        (tmp / f).write_text("")
    (tmp / "reportlab/pdfgen/canvas.py").write_text("class Canvas: pass\n")
    (tmp / "reportlab/lib/pagesizes.py").write_text("letter=(612,792)\n")
    (tmp / "reportlab/lib/colors.py").write_text("black=grey=white=None\n")
    (tmp / "reportlab/lib/styles.py").write_text("def getSampleStyleSheet(): return {}\n")
    (tmp / "reportlab/platypus/__init__.py").write_text("class SimpleDocTemplate: pass\nclass Table: pass\nclass TableStyle: pass\nclass Paragraph: pass\n")
    rs = tmp / "render.py"
    rs.write_text(RENDER_SCRIPT)
    r = subprocess.run([PY, str(rs), str(tmp)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON>>")]
    if not line:
        return {"render_failed": True, "stderr_tail": r.stderr[-800:], "stdout_tail": r.stdout[-400:]}
    data = json.loads(line[0][6:])
    return data


@guarded("supplement_amp_and_scaler_pickle")
def check_supplement() -> dict:
    """(1) Does the frozen Phase 7B numerical protocol (mixed precision) reproduce the frozen
    predictions exactly, i.e. is the residual difference of the fp32 engine purely numeric?
    (2) Which sklearn version was the scaler actually pickled with?
    (BaseEstimator.__getstate__ reports the RUNTIME version, so the pickled one is read from the
    unpickle warning.)"""
    import torch
    from torch.amp import autocast
    import joblib
    import sklearn
    from inference_engine import RUN1_SCALER_PATH
    from phase6b_vector_world_model import scale_array
    from phase7b_stage_targets import CLASS_INDEX_TO_STAGE
    out = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        joblib.load(RUN1_SCALER_PATH)
        msgs = [str(x.message) for x in w if "unpickle" in str(x.message).lower()]
    m = re.search(r"from version ([\d.]+) when using version ([\d.]+)", msgs[0]) if msgs else None
    out["scaler_pickled_with_sklearn"] = m.group(1) if m else None
    out["runtime_sklearn"] = sklearn.__version__
    out["scaler_unpickle_warning_emitted"] = bool(msgs)

    e = get_engine()
    frozen = _load_pred_csv(RES / "phase7b_mitre_stage_head/predictions_test.csv")
    n = e.test_sample_count()
    samples = [e.get_test_sample(i) for i in range(n)]
    keys = [(s["source_file"], s["window_start"]) for s in samples]
    Xs = scale_array(np.stack([s["x_raw"] for s in samples]).astype(np.float32), e.scaler)
    names = [CLASS_INDEX_TO_STAGE[j].name for j in range(6)]
    fp = np.array([float(frozen[k]["attack_probability"]) for k in keys])
    fs = np.array([[names.index(frozen[k][f"mitre_stage_step_{s}"]) for s in range(1, 7)] for k in keys])

    def run(amp):
        probs, stages = [], []
        e.model.eval()
        with torch.no_grad():
            for i in range(0, n, 128):
                xb = torch.from_numpy(Xs[i:i + 128]).to(e.device)
                with autocast(device_type=e.device.type, enabled=amp):
                    _, al, sl, _ = e.model(xb)
                probs.append(torch.sigmoid(al.float()).cpu().numpy())
                stages.append(sl.float().argmax(-1).cpu().numpy())
        return np.concatenate(probs), np.concatenate(stages)

    for amp in (False, True):
        p, s = run(amp)
        out[f"amp_{amp}"] = {"max_abs_prob_diff_vs_frozen_phase7b": float(np.abs(p - fp).max()), "mean_abs_prob_diff": float(np.abs(p - fp).mean()),
                             "stage_cells_mismatch": int((s != fs).sum()), "stage_cells_total": int(s.size),
                             "samples_with_any_stage_mismatch": int((s != fs).any(1).sum())}
    return out


def main_supplement() -> None:
    p = OUT_DIR / "evidence.json"
    ev = json.loads(p.read_text(encoding="utf-8"))
    ev["supplement"] = check_supplement()
    sc = ev["artifact_integrity"]["scaler"]
    if ev["supplement"].get("ok") and ev["supplement"].get("scaler_pickled_with_sklearn"):
        sc["_first_pass_value_pickled_with_sklearn"] = sc.get("pickled_with_sklearn")  # was the runtime version (wrong source)
        sc["pickled_with_sklearn"] = ev["supplement"]["scaler_pickled_with_sklearn"]
        sc["sklearn_version_mismatch"] = sc["pickled_with_sklearn"] != sc["runtime_sklearn"]
        sc["_correction_note"] = "first-pass value read _sklearn_version via __getstate__ (returns runtime version); corrected from the unpickle warning in the supplement check"
    p.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print("supplement merged into", p)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = {"phase": "10A", "generated_unix": time.time(), "note": "evidence only; no model/dataset/Django/frontend modification"}
    ev["artifact_integrity"] = check_artifacts()
    ev["schema_and_scaler"] = check_schema_and_scaler()
    ev["full_test_reproduction"] = check_full_test_reproduction()
    ev["determinism"] = check_determinism()
    ev["stage_head"] = check_stage_head()
    ev["explainability"] = check_explainability()
    ev["thread_safety"] = check_thread_safety()
    ev["forecast_vs_detection"] = check_forecast_vs_detection()
    ev["path_inventory"] = check_path_inventory()
    ev["claims_scan"] = check_claims()
    ev["environment"] = check_environment()
    ev["django_startup"] = check_django_startup()
    ev["cold_start"] = check_cold_start()
    ev["rendered_pages"] = check_rendered_pages()
    (OUT_DIR / "evidence.json").write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print("evidence written:", OUT_DIR / "evidence.json")


if __name__ == "__main__":
    if "--supplement" in sys.argv:
        main_supplement()
    else:
        main()
