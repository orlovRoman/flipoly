"""01_fetch_universe.py

Fetches all active rewards markets from Polymarket CLOB with full pagination until LTE=.
Enriches top reward-yielding candidates via CLOB market info, excludes neg_risk, and saves active universe.
"""

import argparse
import asyncio
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import List, Optional
import httpx

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol
from polyflip.research.lp_rewards.universe import (
    enrich_market_info,
    fetch_all_rewards_markets,
    parse_market_reward_config,
    rank_markets_by_reward_density,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch active rewards markets universe.")
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="Path to root storage directory (overrides protocol default and LP_STORAGE_ROOT).",
    )
    return parser


async def main(storage_root: Optional[str] = None, argv: Optional[List[str]] = None):
    if argv is not None:
        args = build_parser().parse_args(argv)
        if storage_root is None:
            storage_root = args.storage_root

    protocol = load_protocol(storage_root=storage_root)
    print(f"Loaded protocol: {protocol.protocol_id} (SHA-256: {protocol.sha256_hash[:12]}...)")
    print(f"Querying rewards endpoint: {protocol.universe.rewards_endpoint}")

    raw_markets = await fetch_all_rewards_markets(
        terminal_marker=protocol.universe.pagination_terminal_marker,
        page_size=protocol.universe.page_size,
    )
    print(f"Fetched {len(raw_markets)} raw reward records across all pages.")

    # Filter markets that have viable rewards (above dust limit) and sort descending by daily rate
    min_dust = float(protocol.scoring.min_payout_dust_usdc)
    active_pool = [
        m for m in raw_markets
        if float(m.get("total_daily_rate") or m.get("native_daily_rate") or 0) >= min_dust
    ]
    active_pool.sort(key=lambda m: float(m.get("total_daily_rate") or m.get("native_daily_rate") or 0), reverse=True)
    print(f"Markets with viable reward pools (>= ${min_dust:.2f}/day): {len(active_pool)}")

    # Enrich viable candidates (up to 200 top reward-yielding candidates for active/reserve selection)
    candidates = active_pool[:200] if len(active_pool) > 200 else active_pool
    print(f"Enriching {len(candidates)} top reward-yielding candidates via CLOB market info...")

    semaphore = asyncio.Semaphore(15)
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            enrich_market_info(m["condition_id"], client, semaphore)
            for m in candidates
        ]
        enriched_results = await asyncio.gather(*tasks)

    info_by_cid = {cid: info for cid, info in enriched_results if info is not None}
    print(f"Successfully retrieved CLOB metadata for {len(info_by_cid)} markets.")

    valid_configs = []
    excluded_neg_risk = 0
    missing_tokens = 0

    for item in candidates:
        cid = item["condition_id"]
        info = info_by_cid.get(cid)
        if not info:
            continue

        if info.get("neg_risk"):
            excluded_neg_risk += 1
            continue

        cfg = parse_market_reward_config(item, clob_market_info=info, exclude_neg_risk=protocol.universe.exclude_neg_risk)
        if cfg is None:
            missing_tokens += 1
            continue
        valid_configs.append(cfg)

    print(f"Valid markets passing filters: {len(valid_configs)}")
    print(f"Excluded due to neg_risk=true: {excluded_neg_risk}")

    active, reserve = rank_markets_by_reward_density(
        valid_configs,
        top_n_active=protocol.universe.active_markets_count,
        top_n_reserve=protocol.universe.reserve_markets_count,
    )

    print(f"\nTop-{len(active)} Active Markets:")
    for i, m in enumerate(active, 1):
        print(f"  {i}. {m.question[:60]}... | Daily Rate: ${m.rewards_daily_rate:.2f} | OAS: {m.oas}s | Spread: {float(m.rewards_max_spread)*100:.2f}%")

    out_dir = Path(protocol.data_storage.root_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "universe_active.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([m.model_dump(mode="json") for m in active], f, indent=2)

    print(f"\nSaved active universe to {out_file}")


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    asyncio.run(main(storage_root=cli_args.storage_root))
