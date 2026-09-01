#!/usr/bin/env python3
"""Create reproducible 7/14/30-day weighted-policy baseline reports."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.weighted_policy_benchmark import _policy_config, load_from_database
from polyflip.trading.weighted_benchmark import BenchmarkConfig, MarketObservation, benchmark


async def run(args: argparse.Namespace) -> int:
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    windows: dict[str, object] = {}
    for days in (7, 14, 30):
        raw_rows, source = await load_from_database(database_url, days)
        observations = [MarketObservation.from_mapping(row) for row in raw_rows]
        report = benchmark(
            observations,
            config=BenchmarkConfig(
                policy_config=_policy_config(),
                min_net_ev=args.min_net_ev,
                train_min_rows=args.train_min_rows,
                test_size=args.test_size,
                purge_gap=args.purge_gap,
                ridge_lambda=args.ridge_lambda,
                coefficient_bound=args.coefficient_bound,
                bootstrap_iterations=args.bootstrap_iterations,
            ),
        )
        windows[str(days)] = {
            "source": source,
            "raw_rows": len(raw_rows),
            "market_observations": len(observations),
            "report": report.as_dict(),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": windows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--database-url")
    result.add_argument("--output", required=True)
    result.add_argument("--train-min-rows", type=int, default=300)
    result.add_argument("--test-size", type=int, default=100)
    result.add_argument("--purge-gap", type=int, default=0)
    result.add_argument("--ridge-lambda", type=float, default=1.0)
    result.add_argument("--coefficient-bound", type=float, default=5.0)
    result.add_argument("--min-net-ev", type=float, default=0.0)
    result.add_argument("--bootstrap-iterations", type=int, default=1000)
    return result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
