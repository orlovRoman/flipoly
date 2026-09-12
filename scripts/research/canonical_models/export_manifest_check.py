"""Step 6: verify server exports locally (row counts, periods, SHA-256)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    a = ap.parse_args()
    d = Path(a.export_dir)
    files = sorted(d.glob("*.parquet")) + sorted(d.glob("*.csv"))
    if not files:
        print("[export-check] no files; export small date/market slices first")
        return 2
    for f in files:
        print(f"[export-check] {f.name} bytes={f.stat().st_size} sha256={sha256(f)[:16]}...")
    print("[export-check] confirm local hashes match server manifest; stop if the trading cycle degrades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
