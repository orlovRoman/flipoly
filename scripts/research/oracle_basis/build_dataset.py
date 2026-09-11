"""Build the oracle-basis dataset from normalized export outputs.

Reads export_data.py outputs (observations.json, markets.json, manifest.json),
computes boundary proxy/official TWAP diagnostics plus per-horizon as-of rows,
and writes dataset.json, boundary_audit.json, and a build manifest. Historical
boundary reconstruction is retrospective; per-horizon rows use only receipts at
or before each checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Project root on sys.path (repo convention for standalone scripts).
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from polyflip.collector.rtds_collector import RTDSObservation, RTDSError
from polyflip.research.oracle_basis.basis_dataset import HORIZONS_SEC, assemble_row
from polyflip.research.oracle_basis.proxy_twap import (
    InsufficientCoverageError,
    flip_error,
    gap_bps,
    twap_e18,
)
from polyflip.research.oracle_basis.rtds_loader import (
    observations_for_stream,
    select_checkpoint_observation,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def coerce_observation(item: dict[str, Any], index: int) -> RTDSObservation:
    try:
        price = int(item["price_e18"])
        observed = int(item["observed_at_ms"])
        received = int(item["received_at_ms"])
        seq = item.get("seq")
        obs = RTDSObservation(
            source=str(item["source"]).strip().upper(),
            asset=str(item["asset"]).strip().upper(),
            symbol=str(item["symbol"]).strip().upper(),
            currency=str(item["currency"]).strip().upper(),
            price_e18=price,
            observed_at_ms=observed,
            received_at_ms=received,
            seq=int(seq) if seq is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RTDSError(f"observations[{index}] is not normalized: {exc}") from exc
    if price <= 0 or received < observed:
        raise RTDSError(f"observations[{index}] violates price/causality")
    return obs


def window_points(
    stream: list[RTDSObservation], window_end_ms: int, window_ms: int, asof_ms: int
) -> list[tuple[int, int]]:
    # All state known at window end and received by asof; twap_e18 holds the
    # latest pre-window value constant from the window start.
    return [
        (o.observed_at_ms, o.price_e18)
        for o in stream
        if o.observed_at_ms <= window_end_ms and o.received_at_ms <= asof_ms
    ]


def window_twap(
    stream: list[RTDSObservation], window_end_ms: int, window_ms: int, asof_ms: int
) -> int | None:
    try:
        return twap_e18(
            window_points(stream, window_end_ms, window_ms, asof_ms),
            window_end_ms - window_ms,
            window_end_ms,
        )
    except InsufficientCoverageError:
        return None


def parse_horizons(text: str) -> tuple[int, ...]:
    horizons = tuple(
        sorted({int(part) for part in text.split(",") if part.strip()}, reverse=True)
    )
    if not horizons or any(h <= 0 for h in horizons):
        raise RTDSError("horizons must be positive seconds")
    return horizons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build oracle-basis dataset rows.")
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Directory with observations/markets/manifest JSON.",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for dataset JSON."
    )
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS_SEC))
    parser.add_argument("--max-age-ms", type=int, default=15000)
    parser.add_argument("--proxy-window-ms", type=int, default=60000)
    parser.add_argument("--official-window-ms", type=int, default=60000)
    parser.add_argument("--official-window-30-ms", type=int, default=30000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    horizons = parse_horizons(args.horizons)
    observations = [
        coerce_observation(item, i)
        for i, item in enumerate(
            read_json(os.path.join(args.export_dir, "observations.json"))
        )
    ]
    markets = read_json(os.path.join(args.export_dir, "markets.json"))
    manifest = read_json(os.path.join(args.export_dir, "manifest.json"))

    rows: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    for market in markets:
        spot = observations_for_stream(
            observations, market["spot_source"], market["spot_symbol"]
        )
        official = (
            observations_for_stream(
                observations, market["official_source"], market["official_symbol"]
            )
            if market.get("official_symbol")
            else []
        )
        end = int(market["end_ms"])
        start = market.get("start_ms")
        proxy_open = (
            window_twap(spot, start, args.proxy_window_ms, end)
            if start is not None
            else None
        )
        proxy_close = window_twap(spot, end, args.proxy_window_ms, end)
        official_open = (
            window_twap(official, start, args.official_window_ms, end)
            if start is not None and official
            else None
        )
        official_close = (
            window_twap(official, end, args.official_window_ms, end)
            if official
            else None
        )
        gap_open = (
            gap_bps(proxy_open, official_open) if proxy_open and official_open else None
        )
        gap_close = (
            gap_bps(proxy_close, official_close)
            if proxy_close and official_close
            else None
        )
        flip = None
        if proxy_open and proxy_close and market.get("outcome_up") is not None:
            flip = flip_error(proxy_open, proxy_close, bool(market["outcome_up"]))
        boundary.append(
            {
                "market_id": market["market_id"],
                "asset": market["asset"],
                "proxy_open_e18": proxy_open,
                "proxy_close_e18": proxy_close,
                "official_open_e18": official_open,
                "official_close_e18": official_close,
                "gap_open_bps": gap_open,
                "gap_close_bps": gap_close,
                "gap_change_bps": (
                    (gap_close - gap_open)
                    if gap_open is not None and gap_close is not None
                    else None
                ),
                "retrospective": True,
                "flip_error": flip,
            }
        )
        for horizon in horizons:
            checkpoint = end - horizon * 1000
            spot_obs, spot_status, spot_age = select_checkpoint_observation(
                spot, checkpoint, int(args.max_age_ms)
            )
            official_60 = (
                window_twap(
                    official, checkpoint, int(args.official_window_ms), checkpoint
                )
                if official
                else None
            )
            official_30 = (
                window_twap(
                    official, checkpoint, int(args.official_window_30_ms), checkpoint
                )
                if official
                else None
            )
            quote = (market.get("book") or {}).get(str(horizon), {})
            row = assemble_row(
                market_id=market["market_id"],
                asset=market["asset"],
                horizon_sec=horizon,
                checkpoint_ms=checkpoint,
                spot_status=spot_status,
                spot_age_ms=spot_age,
                proxy_open_e18=proxy_open,
                proxy_close_e18=proxy_close,
                official_twap_e18=official_60,
                official_status="OK" if official_60 is not None else "MISSING",
                strike_e18=market.get("strike_e18"),
                up_on_equal=market.get("up_on_equal"),
                book_mid_up=quote.get("mid_up"),
                ask_up=quote.get("ask_up"),
                ask_down=quote.get("ask_down"),
                outcome_up=market.get("outcome_up"),
            )
            row["extensions"] = {
                "official_twap_30_e18": official_30,
                "spot_price_e18": spot_obs.price_e18 if spot_obs else None,
                "retrospective_boundary": True,
            }
            rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    dataset_path = os.path.join(args.out_dir, "dataset.json")
    boundary_path = os.path.join(args.out_dir, "boundary_audit.json")
    manifest_path = os.path.join(args.out_dir, "build_manifest.json")
    write_json(dataset_path, rows)
    write_json(boundary_path, boundary)
    write_json(
        manifest_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "export_manifest": manifest,
            "horizons_sec": list(horizons),
            "max_age_ms": int(args.max_age_ms),
            "rows": len(rows),
            "markets": len(markets),
            "dataset_sha256": sha256_file(dataset_path),
            "boundary_sha256": sha256_file(boundary_path),
        },
    )
    print(f"markets={len(markets)} rows={len(rows)}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
