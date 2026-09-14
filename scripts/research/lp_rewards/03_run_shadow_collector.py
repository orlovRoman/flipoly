"""03_run_shadow_collector.py

Main entrypoint for 7-day shadow collection and paper LP simulation.
Collects L2 snapshots to D:\\flipoly-research\\lp-rewards and runs Quoting FSM.
"""

import asyncio
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.collector import MarketDataCollector
from polyflip.research.lp_rewards.models import MarketRewardConfig, QuotingState
from polyflip.research.lp_rewards.protocol import load_protocol
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM
from polyflip.research.lp_rewards.reconciler import BookReconciler
from polyflip.research.lp_rewards.scoring import calculate_sample_scores
from polyflip.research.lp_rewards.watchdog import SystemWatchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shadow_collector")


async def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    watchdog = SystemWatchdog(
        storage_path=str(storage_path),
        warning_threshold_gb=protocol.data_storage.disk_warning_threshold_gb,
        halt_threshold_gb=protocol.data_storage.disk_emergency_halt_gb,
    )
    reconciler = BookReconciler(
        reconciliation_interval_sec=protocol.ws_collector.rest_reconciliation_interval_sec
    )
    collector = MarketDataCollector(
        ws_endpoint=protocol.ws_collector.ws_endpoint,
        storage_path=str(storage_path),
        ping_interval_sec=protocol.ws_collector.ping_interval_sec,
        pong_timeout_sec=protocol.ws_collector.pong_timeout_sec,
        watchdog=watchdog,
        reconciler=reconciler,
    )

    universe_file = storage_path / "universe_active.json"
    if universe_file.exists():
        with open(universe_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        active_markets = [MarketRewardConfig(**m) for m in raw_data]
    else:
        logger.warning(f"{universe_file} not found. Running with mock fallback configuration.")
        active_markets = []

    collector.set_active_markets(active_markets)
    logger.info(f"Initialized shadow collector for {len(active_markets)} markets. Storage on {storage_path}.")

    # Run collector in background task
    collector_task = asyncio.create_task(collector.run())

    try:
        while True:
            await asyncio.sleep(60.0)
            free_gb, status = watchdog.check_disk_space()
            if status == "EMERGENCY_HALT":
                logger.critical("Emergency disk halt triggered. Terminating shadow collector.")
                collector.running = False
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down cleanly...")
        collector.running = False
    finally:
        collector_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
