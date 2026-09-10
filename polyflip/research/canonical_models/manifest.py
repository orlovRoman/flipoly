"""Step 40: reproducible result (git + manifest + latest.json for Windows)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                                       text=True).strip()
    except Exception:
        return "unknown"


def write_run_manifest(repo_root: Path, run_id: str, payload: dict) -> Path:
    art = repo_root / "artifacts" / "research" / "canonical_models" / run_id
    art.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload["git_sha"] = git_sha(repo_root)
    (art / "manifest.json").write_text(json.dumps(payload, indent=2))
    # heavy local files: index with hashes + storage paths (not the bytes)
    files = sorted(art.glob("**/*"))
    index = []
    for f in files:
        if f.is_file() and f.suffix not in (".json", ".md"):
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            index.append({"path": str(f.relative_to(repo_root)), "sha256": h,
                          "bytes": f.stat().st_size})
    (art / "file_index.json").write_text(json.dumps(index, indent=2))
    latest = {"run_id": run_id, "manifest": str((art / "manifest.json").as_posix()),
              "updated_at": payload["created_at"]}
    (repo_root / "artifacts" / "research" / "canonical_models" / "latest.json").write_text(
        json.dumps(latest, indent=2))
    return art / "manifest.json"
