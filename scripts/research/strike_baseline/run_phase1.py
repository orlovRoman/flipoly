"""Run-level driver for strike-baseline Phase 1 (steps 1-17).

Produces artifacts/research/strike_baseline/<run_id>/ with:
  run_manifest.json  — commit hash, protocol hash, export hash, list of artifacts
  ledger.parquet     — one row per (market, entry_variant) causal decision data
  coverage.csv       — status counts per entry grid / asset (P1-04, P1-06)
  strict/full splits  — availability accounting

Heavy compute stays local; data export + 1m klines happen first (see scripts).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from polyflip.research.strike_baseline.dataset import STATUS_OK, CandleStore, build_ledger
from polyflip.research.strike_baseline.market_rules import build_window

def _parse_end_time_est(val) -> datetime:
    """Convert end_time_est string or Timestamp to a timezone-aware datetime (UTC)."""
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    ts = pd.to_datetime(val, utc=True)
    return ts.to_pydatetime()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write text with LF newlines on every platform (CRLF breaks sha pinning).

    Windows text-mode translation (\\n -> \\r\\n) would make manifest hashes
    platform-dependent; git also normalizes to LF. Bytes on disk are LF always.
    """
    path.write_bytes(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def write_manifest(path: Path, payload: dict) -> dict:
    """Write a manifest with a verifiable self-hash.

    manifest_file_sha256 = sha256 of the canonical JSON bytes of the payload
    WITHOUT the self-hash key. Verification: load JSON, pop the key, re-dump
    with (indent=2, sort_keys=True) + LF, compare sha.
    """
    canonical = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    payload = dict(payload)
    payload["manifest_file_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    final = json.dumps(payload, indent=2, sort_keys=True).replace("\r\n", "\n")
    write_text_lf(path, final + "\n")
    return payload


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def load_export(export_dir: Path) -> dict[str, pd.DataFrame]:
    raw = export_dir / "raw"
    out: dict[str, pd.DataFrame] = {}

    markets = pd.read_csv(raw / "markets.csv.gz", compression="gzip")
    out["markets"] = markets

    snaps = pd.read_csv(
        raw / "snapshots.csv.gz", compression="gzip",
        dtype={"best_bid": "float64", "best_ask": "float64"},
    )
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True, format="ISO8601")
    snaps = snaps[["market_id", "recorded_at", "best_bid", "best_ask"]]
    out["snapshots"] = snaps

    out["candles_15m"] = pd.DataFrame()
    return out


def load_1m(export_dir: Path, candles_15m: pd.DataFrame | None = None) -> CandleStore:
    """Build 1m candle store from the Binance fixture fetched locally."""
    fx = export_dir / "fixture_1m"
    frames = []
    for sym_csv in sorted(fx.glob("*.csv.gz")):
        df = pd.read_csv(sym_csv, compression="gzip")
        # Binance CSV has millisecond epoch ints; convert explicitly.
        df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        df["symbol"] = sym_csv.stem.replace(".csv", "")
        frames.append(df[["symbol", "open_time", "close_time", "open", "high", "low", "close"]])
    all_fx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "open_time", "close_time", "open", "high", "low", "close"]
    )
    return CandleStore(all_fx)


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("usage: run_phase1.py <worktree_root> [data_export_dir]", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    export_dir = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else root / "artifacts" / "research" / "strike_baseline" / "data_export" / "_latest"

    run_id = f"sb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = root / "artifacts" / "research" / "strike_baseline" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit(root)
    protocol_file = root / "research" / "strike_baseline" / "protocol.yaml"
    prot_sha = sha256_file(protocol_file)
    export_manifest_file = export_dir / "export_manifest.json"
    exp_sha = sha256_file(export_manifest_file) if export_manifest_file.exists() else None

    print(f"[run] {run_id}  commit={commit[:12]}  protocol={prot_sha[:12]}")

    data = load_export(export_dir)
    markets = data["markets"]
    snapshots = data["snapshots"]

    # Windows
    windows = {}
    for _, m in markets.iterrows():
        windows[m["market_id"]] = build_window(
            m["market_id"], m["asset"], m["question"], _parse_end_time_est(m["end_time_est"]),
        )

    # 1m candles
    store = load_1m(export_dir)

    print(f"[run] markets={len(markets)} snapshots={len(snapshots)}")

    ledger = build_ledger(markets, windows, snapshots, store)
    print(f"[run] ledger rows={len(ledger)}  ok={int((ledger.status == STATUS_OK).sum())}")

    ledger_path = out_dir / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    coverage = (
        ledger.groupby(["asset", "entry_min_before_close", "status"])
        .size()
        .rename("n")
        .reset_index()
    )
    coverage_path = out_dir / "coverage.csv"
    coverage_csv = coverage.to_csv(index=False, lineterminator="\n")
    write_text_lf(coverage_path, coverage_csv)

    run_manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch_protocol": {"commit": commit, "protocol_sha256": prot_sha, "protocol_file": str(protocol_file)},
        "data_export": {
            "dir": str(export_dir),
            "export_manifest_sha256": exp_sha,
            "binance_manifest_sha256": sha256_file(export_dir / "fixture_1m" / "binance_1m_manifest.json"),
        },
        "artifacts": {},
    }
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            run_manifest["artifacts"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    man_path = out_dir / "run_manifest.json"
    write_manifest(man_path, run_manifest)

    print()
    print(f"RUN_COMPLETE dir={out_dir}")
    print(f"  manifest={man_path}")


if __name__ == "__main__":
    main()