"""Export and normalize RTDS observations plus market boundaries.

Research-only I/O wrapper. Reads raw RTDS observation JSONL and a market-boundary
JSON file, validates them with the frozen helpers, and writes normalized JSON plus
a manifest with SHA-256 hashes. No network, no database, no trading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

# Project root on sys.path (repo convention for standalone scripts).
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from polyflip.collector.rtds_collector import (
    RTDSError,
    parse_price_e18,
    parse_timestamp_ms,
)
from polyflip.research.oracle_basis.rtds_loader import load_observations


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RTDSError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def normalize_markets(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise RTDSError("markets file must be a non-empty JSON list")
    markets = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RTDSError(f"markets[{index}] must be an object")
        try:
            market_id = str(item["market_id"])
            asset = str(item["asset"]).strip().upper()
            end_ms = parse_timestamp_ms(item["end_ms"])
            spot_symbol = str(item["spot_symbol"]).strip().upper()
            spot_source = str(item.get("spot_source", "BINANCE")).strip().upper()
        except KeyError as exc:
            raise RTDSError(f"markets[{index}] missing field: {exc}") from exc
        if not market_id or not asset or not spot_symbol:
            raise RTDSError(f"markets[{index}] has empty market_id/asset/spot_symbol")
        start_ms = item.get("start_ms")
        start = parse_timestamp_ms(start_ms) if start_ms is not None else None
        if start is not None and start >= end_ms:
            raise RTDSError(f"markets[{index}] start_ms must precede end_ms")
        outcome = item.get("outcome_up")
        if outcome is not None and not isinstance(outcome, bool):
            raise RTDSError(f"markets[{index}] outcome_up must be bool or null")
        official_symbol = item.get("official_symbol")
        official_source = (
            str(item.get("official_source", "CHAINLINK_OFFICIAL")).strip().upper()
        )
        if official_symbol is not None:
            official_symbol = str(official_symbol).strip().upper()
            if not official_symbol:
                raise RTDSError(f"markets[{index}] official_symbol is empty")
        strike = item.get("strike_e18")
        if strike is not None and (
            isinstance(strike, bool) or not isinstance(strike, int) or strike <= 0
        ):
            raise RTDSError(
                f"markets[{index}] strike_e18 must be a positive int or null"
            )
        up_on_equal = item.get("up_on_equal")
        if up_on_equal is not None and not isinstance(up_on_equal, bool):
            raise RTDSError(f"markets[{index}] up_on_equal must be bool or null")
        book = item.get("book", {})
        if not isinstance(book, dict):
            raise RTDSError(f"markets[{index}] book must be an object or omitted")
        # Validate price strings early so export fails fast on malformed books.
        for horizon, quote in book.items():
            if not isinstance(quote, dict) or "mid_up" not in quote:
                raise RTDSError(f"markets[{index}] book[{horizon}] needs mid_up")
        markets.append(
            {
                "market_id": market_id,
                "asset": asset,
                "start_ms": start,
                "end_ms": end_ms,
                "outcome_up": outcome,
                "spot_source": spot_source,
                "spot_symbol": spot_symbol,
                "official_source": official_source,
                "official_symbol": official_symbol,
                "strike_e18": strike,
                "up_on_equal": up_on_equal,
                "book": book,
            }
        )
    ids = [m["market_id"] for m in markets]
    if len(set(ids)) != len(ids):
        raise RTDSError("market_id must be unique")
    # Validate book quotes early so export fails fast on malformed numbers.
    for market in markets:
        for horizon, quote in market["book"].items():
            mid = quote.get("mid_up")
            if (
                not isinstance(mid, (float, int))
                or isinstance(mid, bool)
                or not 0.0 <= float(mid) <= 1.0
            ):
                raise RTDSError(
                    f"book[{horizon}] mid_up must be a probability in [0, 1]"
                )
            for key in ("ask_up", "ask_down"):
                if quote.get(key) is not None:
                    parse_price_e18(str(quote[key]))
    return markets


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize RTDS observations and market boundaries."
    )
    parser.add_argument(
        "--observations", required=True, help="Raw RTDS observation JSONL file."
    )
    parser.add_argument("--markets", required=True, help="Market boundary JSON file.")
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for normalized JSON."
    )
    parser.add_argument(
        "--protocol",
        default=os.path.join("research", "oracle_basis", "protocol.yaml"),
        help="Frozen protocol file hashed into the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_rows = read_jsonl(args.observations)
    with open(args.markets, "r", encoding="utf-8") as handle:
        raw_markets = json.load(handle)
    observations, summary = load_observations(raw_rows)
    markets = normalize_markets(raw_markets)
    os.makedirs(args.out_dir, exist_ok=True)
    obs_path = os.path.join(args.out_dir, "observations.json")
    markets_path = os.path.join(args.out_dir, "markets.json")
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_json(obs_path, [asdict(o) for o in observations])
    write_json(markets_path, markets)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_file": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "observations": args.observations,
            "observations_sha256": sha256_file(args.observations),
            "markets": args.markets,
            "markets_sha256": sha256_file(args.markets),
        },
        "outputs": {
            "observations": obs_path,
            "observations_sha256": sha256_file(obs_path),
            "markets": markets_path,
            "markets_sha256": sha256_file(markets_path),
        },
        "loader_summary": summary,
    }
    write_json(manifest_path, manifest)
    print(
        f"kept={summary['rows_kept']} rejected={summary['rejected_count']} "
        f"duplicates={summary['duplicates']} conflicts={summary['conflicts']}"
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
