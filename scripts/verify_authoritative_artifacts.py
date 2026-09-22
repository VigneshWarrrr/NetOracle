"""Verify the NetOracle authoritative artifacts against the recorded manifest.

    python scripts/verify_authoritative_artifacts.py            # checkpoint, scaler, backbone reference, window CSVs
    python scripts/verify_authoritative_artifacts.py --no-data  # skip the external window CSVs

Reads experiments/results/phase10b_submission_remediation/authoritative_artifacts.json, recomputes each
SHA-256 and size, and exits 0 only if everything matches. Read-only; standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "experiments/results/phase10b_submission_remediation/authoritative_artifacts.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL manifest not found: {MANIFEST}")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = list(manifest["artifacts"])
    if "--no-data" not in sys.argv:
        entries += manifest["window_files"]
    failures = 0
    for entry in entries:
        path = (REPO_ROOT / entry["path"]).resolve()
        if not path.is_file():
            print(f"MISSING  {entry['path']}")
            if entry.get("retrieval"):
                print(f"         retrieval: {entry['retrieval']}")
            failures += 1
            continue
        ok = path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"]
        print(f"{'OK      ' if ok else 'MISMATCH'} {entry['path']}  sha256={entry['sha256'][:16]}...")
        failures += 0 if ok else 1
    print("\nALL AUTHORITATIVE ARTIFACTS VERIFIED" if not failures else f"\n{failures} PROBLEM(S) FOUND")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
