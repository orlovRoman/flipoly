"""Quality report for the oracle-basis data foundation.

Reads export/build outputs and writes data_quality.json: stream coverage and
arrival gaps, market-boundary checks (15-minute windows, last-minute spot
coverage), reconstruction availability, and horizon exclusion accounting.
Read-only; fails loudly on malformed inputs instead of imputing them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

MARKET_WINDOW_MS = 15 * 60 * 1000


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write oracle-basis data_quality.json."
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Directory with observations/markets/manifest JSON.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory with dataset/boundary_audit JSON.",
    )
    parser.add_argument("--out", required=True, help="Output data_quality.json path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    observations = read_json(os.path.join(args.export_dir, "observations.json"))
    markets = read_json(os.path.join(args.export_dir, "markets.json"))
    manifest = read_json(os.path.join(args.export_dir, "manifest.json"))
    rows = read_json(os.path.join(args.dataset_dir, "dataset.json"))
    boundary = read_json(os.path.join(args.dataset_dir, "boundary_audit.json"))

    loader = manifest.get("loader_summary", {})
    streams = loader.get("streams", {})
    stream_table = {}
    for key, stats in streams.items():
        stream_table[key] = {
            "count": stats.get("count", 0),
            "gaps_over_2s": stats.get("gaps_over_threshold", 0),
            "max_gap_ms": stats.get("max_gap_ms", 0),
            "out_of_order": stats.get("out_of_order", 0),
        }

    obs_by_stream: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        obs_by_stream.setdefault(f"{obs['source']}|{obs['symbol']}", []).append(obs)

    boundary_checks = []
    for market in markets:
        key = f"{market['spot_source']}|{market['spot_symbol']}"
        end = int(market["end_ms"])
        start = market.get("start_ms")
        duration_ok = start is not None and end - int(start) == MARKET_WINDOW_MS
        last_minute = sum(
            1
            for o in obs_by_stream.get(key, [])
            if end - 60_000 <= int(o["observed_at_ms"]) <= end
        )
        boundary_checks.append(
            {
                "market_id": market["market_id"],
                "window_is_15m": bool(duration_ok),
                "spot_obs_last_minute": last_minute,
                "outcome_known": market.get("outcome_up") is not None,
                "official_stream": market.get("official_symbol"),
                "strike_known": market.get("strike_e18") is not None,
            }
        )

    recon = [r for r in rows if r.get("reconstruction_available")]
    exclusions: Counter[str] = Counter()
    spot_status: Counter[str] = Counter()
    for row in rows:
        spot_status[str(row.get("spot_status"))] += 1
        for reason in row.get("exclusion_reasons", []) or []:
            exclusions[str(reason)] += 1
    gap_buckets: Counter[str] = Counter()
    for row in recon:
        gap = row.get("reconstruction_gap_abs_bps")
        gap_buckets["<3bps" if gap < 3 else "3-10bps" if gap <= 10 else ">10bps"] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "streams": stream_table,
        "duplicates": loader.get("duplicates", 0),
        "conflicts": loader.get("conflicts", 0),
        "rejected": loader.get("rejected_count", 0),
        "markets": len(markets),
        "markets_window_15m": sum(1 for b in boundary_checks if b["window_is_15m"]),
        "markets_outcome_known": sum(1 for b in boundary_checks if b["outcome_known"]),
        "markets_strike_known": sum(1 for b in boundary_checks if b["strike_known"]),
        "boundary_checks": boundary_checks,
        "horizon_rows": len(rows),
        "spot_status": dict(spot_status),
        "reconstruction_rows": len(recon),
        "reconstruction_share": (len(recon) / len(rows)) if rows else 0.0,
        "gap_buckets": dict(gap_buckets),
        "boundary_flip_errors": sum(
            1 for b in boundary if (b.get("flip_error") or {}).get("flip_error") is True
        ),
        "exclusions": dict(exclusions),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_json(args.out, report)
    print(f"markets={len(markets)} rows={len(rows)} reconstruction={len(recon)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
