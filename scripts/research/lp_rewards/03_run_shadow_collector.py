import asyncio
from decimal import Decimal
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List
import httpx

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.capital_allocator import CapitalAllocator
from polyflip.research.lp_rewards.collector import MarketDataCollector
from polyflip.research.lp_rewards.fill_simulation import QueuePositionTracker
from polyflip.research.lp_rewards.ledger import PortfolioLedger
from polyflip.research.lp_rewards.models import MarketPosition, MarketRewardConfig, OrderSide, QuotingState, VirtualOrder
from polyflip.research.lp_rewards.protocol import load_protocol
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM
from polyflip.research.lp_rewards.reconciler import BookReconciler
from polyflip.research.lp_rewards.scoring import (
    calculate_cutoff_midpoint,
    calculate_sample_scores,
    calculate_competitor_and_own_scores,
)
from polyflip.research.lp_rewards.watchdog import SystemWatchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shadow_collector")


async def periodic_l2_and_trade_flush(collector: MarketDataCollector, interval_sec: float = 10.0):
    """Periodically flush RAM orderbook snapshots and trades to Parquet on disk."""
    logger.info("Started periodic L2 snapshots and trades persistence loop.")
    while collector.running:
        try:
            await asyncio.sleep(interval_sec)
            collector.flush_l2_snapshots_to_disk()
            collector.flush_trades_to_disk()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic L2/trade flush: {e}")


async def periodic_book_reconciler(
    collector: MarketDataCollector,
    reconciler: BookReconciler,
    active_markets: List[MarketRewardConfig],
    interval_sec: float = 45.0,
):
    """Periodically query REST /book to verify WS orderbook integrity."""
    logger.info(f"Started periodic BookReconciler loop (every {interval_sec}s).")
    async with httpx.AsyncClient(timeout=15.0) as client:
        while collector.running:
            try:
                await asyncio.sleep(interval_sec)
                for market in active_markets:
                    for token_id in (market.yes_token_id, market.no_token_id):
                        snap = collector.ram_store.get_snapshot(market.condition_id, token_id)
                        rest_res = await reconciler.fetch_rest_book(token_id, client)
                        if rest_res:
                            rest_bid, rest_ask = rest_res
                            reconciler.reconcile_book(token_id, snap.bids, snap.asks, rest_bid, rest_ask)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in periodic book reconciler: {e}")


async def periodic_fsm_and_scoring(
    collector: MarketDataCollector,
    active_markets: List[MarketRewardConfig],
    fsms: Dict[str, MarketQuotingFSM],
    ledger: PortfolioLedger,
    allocator: CapitalAllocator,
    protocol: Any,
    interval_sec: float = 60.0,
):
    """Periodically evaluate minute scoring samples and advance quoting FSM."""
    logger.info(f"Started periodic FSM and Scoring loop (every {interval_sec}s).")
    storage_path = Path(protocol.data_storage.root_path)

    order_trackers = {}
    fsm_ticks = 0
    market_uptime = {m.condition_id: 0 for m in active_markets}
    loop_start_time = time.time()

    while collector.running:
        try:
            await asyncio.sleep(interval_sec)
            now_ns = time.time_ns()
            date_str = time.strftime("%Y-%m-%d", time.gmtime())
            fsm_ticks += 1

            # Snapshot recent trades to process for fill simulation
            recent_trades = list(collector.simulation_trade_buffer)
            collector.simulation_trade_buffer.clear()

            # Clean up closed orders from trackers
            all_open_oids = set()
            for m in active_markets:
                all_open_oids.update(fsms[m.condition_id].open_orders.keys())
            order_trackers = {oid: trk for oid, trk in order_trackers.items() if oid in all_open_oids}

            for market in active_markets:
                cid = market.condition_id
                fsm = fsms[cid]

                yes_snap = collector.ram_store.get_snapshot(cid, market.yes_token_id)
                no_snap = collector.ram_store.get_snapshot(cid, market.no_token_id)

                is_uncertain = (
                    collector.reconciler.uncertain_markets.get(market.yes_token_id, False)
                    or collector.reconciler.uncertain_markets.get(market.no_token_id, False)
                )

                # 1. Simulate fills from incoming trades against open orders
                if fsm.open_orders and recent_trades:
                    mkt_trades = [t for t in recent_trades if t.get("condition_id") == cid or t.get("asset_id") in (market.yes_token_id, market.no_token_id)]
                    for t in mkt_trades:
                        t_price = Decimal(str(t.get("price", "0")))
                        t_size = Decimal(str(t.get("size", "0")))
                        t_side_str = str(t.get("side", "")).upper()
                        t_side = OrderSide.BUY if t_side_str in ("BUY", "BID") else OrderSide.SELL
                        t_time = int(t.get("observed_at_ns", t.get("timestamp_ns", now_ns)))

                        for oid, order in list(fsm.open_orders.items()):
                            if order.asset_id == t.get("asset_id"):
                                if oid not in order_trackers:
                                    snap = yes_snap if order.asset_id == market.yes_token_id else no_snap
                                    levels = snap.bids if order.side == OrderSide.BUY else snap.asks
                                    depth = Decimal("0.0")
                                    for lvl in levels:
                                        if (order.side == OrderSide.BUY and lvl.price > order.price) or \
                                           (order.side == OrderSide.SELL and lvl.price < order.price):
                                            depth += lvl.size
                                        elif lvl.price == order.price:
                                            depth += lvl.size
                                    order_trackers[oid] = QueuePositionTracker(
                                        order=order,
                                        existing_depth_ahead=depth,
                                        min_order_age_sec=market.oas,
                                    )
                                tracker = order_trackers[oid]
                                fill = tracker.process_public_trade(t_price, t_size, t_side, t_time)
                                if fill:
                                    fsm.on_fill(fill)
                                    ledger.record_fill(fill)

                # 2. Quoting FSM management
                midpoint = calculate_cutoff_midpoint(yes_snap.bids, yes_snap.asks, market.rewards_min_size)
                if midpoint is not None and not is_uncertain:
                    market_uptime[cid] += 1
                    if fsm.state == QuotingState.FLAT:
                        new_quotes = fsm.generate_quote_orders(midpoint=midpoint, timestamp_ns=now_ns)
                        positions = {m.condition_id: fsms[m.condition_id].position for m in active_markets}
                        open_orders = {m.condition_id: list(fsms[m.condition_id].open_orders.values()) for m in active_markets}
                        allowed, _ = allocator.can_allocate_orders(cid, positions, open_orders, new_quotes)
                        if not allowed:
                            fsm.reset_orders()

                # 3. Competitive LP Scoring and Reward Share Calculation
                lp_est = calculate_competitor_and_own_scores(
                    condition_id=cid,
                    timestamp_ns=now_ns,
                    public_yes_bids=yes_snap.bids,
                    public_yes_asks=yes_snap.asks,
                    public_no_bids=no_snap.bids,
                    public_no_asks=no_snap.asks,
                    our_orders=list(fsm.open_orders.values()),
                    max_spread=market.rewards_max_spread,
                    min_size=market.rewards_min_size,
                    yes_token_id=market.yes_token_id,
                    no_token_id=market.no_token_id,
                    is_uncertain=is_uncertain,
                )
                if lp_est.status == "VALID" and lp_est.share_expected > Decimal("0.0"):
                    minute_reward = (market.rewards_daily_rate / Decimal("1440.0")) * lp_est.share_expected
                    ledger.record_daily_rewards(date_str, minute_reward)

                # 4. Check FSM timeouts and forced exit
                exited, fill = fsm.check_timeout_and_exit(now_ns, yes_snap if fsm.position.yes_inventory > 0 else no_snap)
                if fill:
                    ledger.record_fill(fill)

            # 5. Periodically export ledger parquet
            sim_ledger_dir = storage_path / "simulated_ledger"
            sim_ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger.export_trades_parquet(sim_ledger_dir / f"{date_str}.parquet")

            # 6. Calculate Executable MTM & Net PnL and export daily evaluation artifact
            positions = {m.condition_id: fsms[m.condition_id].position for m in active_markets}
            current_bids = {}
            taker_fees = {}
            for m in active_markets:
                yes_snap = collector.ram_store.get_snapshot(m.condition_id, m.yes_token_id)
                no_snap = collector.ram_store.get_snapshot(m.condition_id, m.no_token_id)
                current_bids[f"{m.condition_id}_YES"] = yes_snap.bids
                current_bids[f"{m.condition_id}_NO"] = no_snap.bids
                taker_fees[m.condition_id] = m.taker_fee_rate

            exec_mtm = ledger.calculate_executable_mtm(positions, current_bids, taker_fees)
            net_pnl = ledger.calculate_net_pnl(positions, exec_mtm)

            daily_eval_dir = storage_path / "daily_evaluations"
            daily_eval_dir.mkdir(parents=True, exist_ok=True)
            daily_eval_file = daily_eval_dir / f"{date_str}.json"

            market_breakdown = {}
            for m in active_markets:
                m_pos = fsms[m.condition_id].position
                cov_ratio = market_uptime[m.condition_id] / fsm_ticks if fsm_ticks > 0 else 0.0
                market_breakdown[m.condition_id] = {
                    "net_pnl": str(m_pos.realized_trading_pnl),
                    "coverage_ratio": f"{cov_ratio:.4f}",
                }

            uncertain_count = sum(1 for v in collector.reconciler.uncertain_markets.values() if v)
            quote_hours = (time.time() - loop_start_time) / 3600.0

            eval_record = {
                "date": date_str,
                "protocol_id": protocol.protocol_id,
                "protocol_hash": protocol.sha256_hash,
                "net_pnl": str(net_pnl),
                "quote_hours": f"{quote_hours:.4f}",
                "book_uncertain_count": uncertain_count,
                "market_breakdown": market_breakdown,
                "total_trades": len(ledger.trades),
                "executable_mtm": str(exec_mtm),
                "total_rewards_accrued": str(ledger.daily_rewards_accrued.get(date_str, Decimal("0.0"))),
            }
            with open(daily_eval_file, "w", encoding="utf-8") as ef:
                json.dump(eval_record, ef, indent=2)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in FSM and scoring loop: {e}")


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

    # Initialize FSMs, Ledger, Allocator
    fsms = {
        m.condition_id: MarketQuotingFSM(
            config=m,
            hedging_timeout_sec=protocol.quoting_fsm.hedging_timeout_sec,
            max_combined_fill_cost=protocol.quoting_fsm.max_combined_fill_cost,
        )
        for m in active_markets
    }
    ledger = PortfolioLedger(allocated_capital=protocol.capital_allocation.allocated_working_capital)
    allocator = CapitalAllocator(
        allocated_working_capital=protocol.capital_allocation.allocated_working_capital,
        max_unhedged_per_market=protocol.capital_allocation.max_unhedged_per_market,
        max_unhedged_total=protocol.capital_allocation.max_unhedged_total,
    )

    collector_task = asyncio.create_task(collector.run())
    l2_flush_task = asyncio.create_task(periodic_l2_and_trade_flush(collector, interval_sec=10.0))
    reconciler_task = asyncio.create_task(periodic_book_reconciler(collector, reconciler, active_markets, interval_sec=protocol.ws_collector.rest_reconciliation_interval_sec))
    fsm_task = asyncio.create_task(periodic_fsm_and_scoring(collector, active_markets, fsms, ledger, allocator, protocol, interval_sec=60.0))

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
        collector.running = False
        collector_task.cancel()
        l2_flush_task.cancel()
        reconciler_task.cancel()
        fsm_task.cancel()
        # Final flush on exit
        collector.flush_l2_snapshots_to_disk()
        collector.flush_trades_to_disk()


if __name__ == "__main__":
    asyncio.run(main())
