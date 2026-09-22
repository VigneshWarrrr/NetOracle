"""Phase 10B: build experiments/results/phase10b_submission_remediation/authoritative_artifacts.json.

Records, for every authoritative artifact: path, SHA-256, size, role, whether Git tracks it
(from `git ls-files`, i.e. what Git actually proves) and, if it is not in Git, how to obtain it.
Also records the exact source import closure of the authoritative inference path, measured by
importing it, so "which source files must be version-controlled" is answered by evidence.

Read-only with respect to every model artifact. Run AFTER `git add` so the tracked flags are
meaningful; re-run after the commit to confirm they still hold against HEAD.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
WINDOWS_DIR = REPO_ROOT.parent / "data/windows"
OUT = EXPERIMENTS_DIR / "results/phase10b_submission_remediation"
sys.path.insert(0, str(REPO_ROOT))

CKPT = "experiments/results/phase7b_mitre_stage_head/model/best_stage_head.pt"
SCALER = "experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/scaler.joblib"
RUN1 = "experiments/results/phase6b_vector_world_model_ablation/run1_existing_scaling/model/best_model.pt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout


def main() -> dict:
    mode = "head" if "--head" in sys.argv else "index"
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD").splitlines()) if mode == "head" else set(git("ls-files").splitlines())

    phase9k = json.loads((EXPERIMENTS_DIR / "results/phase9k_integration/safety_manifest.json").read_text(encoding="utf-8"))["checksums"]
    expected = {CKPT: phase9k["phase7b_stage_head_checkpoint"]["sha256"], SCALER: phase9k["phase6b_run1_scaler"]["sha256"], RUN1: phase9k["phase6b_run1_checkpoint"]["sha256"]}

    artifacts = []
    for aid, rel, role in (
        ("checkpoint", CKPT, "AUTHORITATIVE checkpoint: frozen Phase 6B Run-1 backbone + trained Phase 7B MITRE stage head (VectorWorldModelWithStageHead, 356,452 parameters)"),
        ("scaler", SCALER, "AUTHORITATIVE scaler: train-only StandardScaler, 157 features, never refit"),
        ("run1_backbone_reference", RUN1, "Reference: Phase 6B Run-1 backbone; its 39 tensors are bit-identical to the backbone inside the authoritative checkpoint (used by verification, not by inference)"),
    ):
        p = REPO_ROOT / rel
        artifacts.append({
            "id": aid, "path": rel, "sha256": sha(p), "size_bytes": p.stat().st_size, "role": role,
            "matches_phase9k_manifest": sha(p) == expected[rel],
            "tracked_by_git": rel in tracked,
            "storage": "A: committed to Git (plain blob, no LFS)" if rel in tracked else "NOT in Git",
            "retrieval": None if rel in tracked else "not applicable",
        })
    windows = []
    for p in sorted(WINDOWS_DIR.glob("*.csv")):
        windows.append({
            "id": "window:" + p.name, "path": "../data/windows/" + p.name, "sha256": sha(p), "size_bytes": p.stat().st_size,
            "role": "Phase 3.5 window file: source of the 157-feature names/order and of the replay samples read by the engine at start-up",
            "tracked_by_git": False,
            "storage": "C: external to Git (outside the repository tree; the sibling directory <workspace>/data/windows)",
            "retrieval": "Obtain from the submission bundle and place at <workspace>/data/windows/. No immutable public URL exists for these derived files. "
                         "They can be rebuilt with scripts/build_temporal_dataset.py from the public CSE-CIC-IDS2018 CSVs, but a rebuild is NOT verified to be byte-identical: compare against this SHA-256.",
        })

    # exact source closure of the authoritative path, measured by importing it
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    import world_model.inference_service as svc  # noqa: F401
    svc.AuthoritativeForecastService  # noqa: B018
    import inference_engine  # noqa: F401
    closure = sorted({Path(m.__file__).resolve().relative_to(REPO_ROOT).as_posix() for m in list(sys.modules.values())
                      if getattr(m, "__file__", None) and Path(m.__file__).is_absolute()  # some third-party modules report a relative __file__
                      and str(Path(m.__file__).resolve()).startswith(str(REPO_ROOT)) and "site-packages" not in str(m.__file__)})
    wiring = ["dashboard/views.py", "dashboard/urls.py", "api/views.py", "api/urls.py", "templates/dashboard/authoritative_forecast.html", "templates/dashboard/index.html"]
    sources = [{"path": rel, "sha256": sha(REPO_ROOT / rel), "tracked_by_git": rel in tracked, "why": "imported by the authoritative inference path (measured)" if rel in closure else "Django wiring / template for the authoritative page"}
               for rel in sorted(set(closure) | set(wiring))]

    other_ckpt = []
    for p in sorted((EXPERIMENTS_DIR / "results").rglob("*")):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if p.is_file() and p.suffix in (".pt", ".pth") and rel not in (CKPT, RUN1):
            other_ckpt.append({"path": rel, "sha256": sha(p), "size_bytes": p.stat().st_size, "tracked_by_git": rel in tracked})

    feature_schema = json.loads((OUT / "integrity_before.json").read_text(encoding="utf-8"))["engine_provenance_hashes"]
    manifest = {
        "phase": "10B", "checked_against": "HEAD" if mode == "head" else "git index",
        "artifact_storage_summary": {
            "checkpoint": "A (committed to Git)", "scaler": "A (committed to Git)", "run1_backbone_reference": "A (committed to Git)",
            "window_dataset": "C (external, SHA-256 per file + layout/retrieval instructions; no immutable public URL)",
            "git_lfs": "not used (artifacts are 1.4 MB and 4 KB)",
        },
        "required_workspace_layout": {"<workspace>/NetOracle/": "this repository", "<workspace>/data/windows/": "ten Phase 3.5 CSV files (external)"},
        "feature_schema": {"count": feature_schema["feature_count"], "sha256": feature_schema["feature_schema_sha256"],
                           "source": "header order of data/windows/*.csv after removing the META columns (identical to feature_columns in experiments/results/phase4_baseline/config.json); the schema is not stored inside the checkpoint"},
        "expected_input_shape": [6, 157],
        "artifacts": artifacts, "window_files": windows, "authoritative_source_closure_and_wiring": sources,
        "other_local_checkpoints_not_needed_for_inference": {"note": "ignored by .gitignore (*.pt) and not tracked; required only to re-run other phases' tests/experiments", "files": other_ckpt},
    }
    (OUT / "authoritative_artifacts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bad = [a["path"] for a in artifacts if not a["matches_phase9k_manifest"]] + [s["path"] for s in sources if not s["tracked_by_git"]] + [a["path"] for a in artifacts if not a["tracked_by_git"]]
    print(json.dumps({"checked_against": manifest["checked_against"], "artifacts": [(a["id"], a["sha256"][:12], a["tracked_by_git"]) for a in artifacts],
                      "source_files": len(sources), "sources_not_tracked": [s["path"] for s in sources if not s["tracked_by_git"]],
                      "problems": bad}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
