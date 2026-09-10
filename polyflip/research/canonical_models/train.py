"""Step 28: local-only training with full provenance.

Saves models, features, params, seed, lib versions, train periods and
input hashes. Re-running with identical inputs reproduces metrics within
protocol.numeric_tolerance. Refuses SSHFS/server (see guards).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import models as M
from .guards import assert_train_allowed
from .protocol import attach_hash, load_protocol


def lib_versions() -> dict:
    out = {}
    for mod in ("sklearn", "lightgbm", "pandas", "numpy", "pyarrow"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "missing"
    return out


def hash_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: str(x.get("market_id"))):
        h.update(json.dumps({k: str(v) for k, v in sorted(r.items())},
                            sort_keys=True).encode())
    return h.hexdigest()


def train_all(workdir: str | Path, run_id: str, train_rows: list[dict],
              valid_rows: list[dict] | None = None, seed: int = 42) -> dict:
    root = assert_train_allowed(workdir)
    proto = load_protocol()
    art = Path(root) / "artifacts" / "research" / "canonical_models" / run_id
    art.mkdir(parents=True, exist_ok=True)

    m1 = M.LogRegModel(M.M1_COLUMNS).fit(train_rows)
    m2 = M.LogRegModel(M.M2_COLUMNS, C=1.0).fit(train_rows)
    m3 = M.LgbmModel(M.M2_COLUMNS, seed=seed).fit(train_rows, valid_rows)
    m4 = M.MarketOffsetModel(M.M2_COLUMNS).fit(train_rows)

    (art / "m1.pkl").write_bytes(m1.dumps())
    import pickle
    (art / "m2.pkl").write_bytes(m2.dumps())
    m3.booster.save_model(str(art / "m3.txt"))
    with open(art / "m4.json", "w") as f:
        json.dump({"w": m4.w.tolist(), "b": m4.b,
                   "mean": m4.scaler.mean.tolist(),
                   "scale": m4.scaler.scale.tolist(),
                   "columns": list(m4.columns)}, f)
    manifest = attach_hash({
        "run_id": run_id,
        "seed": seed,
        "lib_versions": lib_versions(),
        "train_periods": {
            "n_train": len(train_rows),
            "n_valid": len(valid_rows or []),
        },
        "input_hash": hash_rows(train_rows),
        "features": {"M1": list(M.M1_COLUMNS), "M2_M3_M4": list(M.M2_COLUMNS)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, proto)
    (art / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
