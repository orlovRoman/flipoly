import asyncio
from decimal import Decimal
import importlib
import json
from pathlib import Path
import pytest

collector_03 = importlib.import_module("scripts.research.lp_rewards.03_run_shadow_collector")
periodic_fsm_and_scoring = collector_03.periodic_fsm_and_scoring

from polyflip.research.lp_rewards.capital_allocator import CapitalAllocator
from polyflip.research.lp_rewards.collector import MarketDataCollector, OrderbookRAMStore
from polyflip.research.lp_rewards.ledger import PortfolioLedger
from polyflip.research.lp_rewards.models import (
    MarketPosition,
    MarketRewardConfig,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderSide,
    QuotingState,
    VirtualOrder,
)
from polyflip.research.lp_rewards.protocol import LPProtocol, load_protocol
from polyflip.research.lp_rewards.quoting_fsm import MarketQuotingFSM


@pytest.mark.asyncio
async def test_shadow_collector_quote_hours_only_when_quoting(tmp_path, monkeypatch):
    """Verify market_uptime / quote_hours strictly tracks order presence, not mere midpoint existence."""
    protocol = load_protocol()
    # Point data root to tmp_path
    monkeypatch.setattr(protocol.data_storage, "root_path", str(tmp_path))

    cfg = MarketRewardConfig(
        condition_id="c_test_quote",
        question="Test Question?",
        rewards_daily_rate=Decimal("100.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        oas=Decimal("5.0"),
        taker_fee_rate=Decimal("0.01"),
        yes_token_id="tok_y",
        no_token_id="tok_n",
    )

    collector = MarketDataCollector(storage_path=str(tmp_path))
    collector.running = True

    # Populate RAM orderbook with wide book so calculate_cutoff_midpoint works
    collector.ram_store.apply_snapshot(
        "tok_y",
        bids=[{"price": "0.48", "size": "100.0"}],
        asks=[{"price": "0.52", "size": "100.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )
    collector.ram_store.apply_snapshot(
        "tok_n",
        bids=[{"price": "0.48", "size": "100.0"}],
        asks=[{"price": "0.52", "size": "100.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )

    fsm = MarketQuotingFSM(config=cfg)
    fsms = {cfg.condition_id: fsm}
    ledger = PortfolioLedger(allocated_capital=Decimal("100.0"))

    # Case 1: Allocator allows orders -> fsm has open_orders -> uptime should increment
    allocator = CapitalAllocator(
        allocated_working_capital=Decimal("100.0"),
        max_unhedged_per_market=Decimal("25.0"),
        max_unhedged_total=Decimal("100.0"),
    )

    task = asyncio.create_task(
        periodic_fsm_and_scoring(
            collector=collector,
            active_markets=[cfg],
            fsms=fsms,
            ledger=ledger,
            allocator=allocator,
            protocol=protocol,
            interval_sec=0.01,
        )
    )

    # Let it tick twice
    await asyncio.sleep(0.05)
    collector.running = False
    await task

    daily_eval_dir = tmp_path / "daily_evaluations"
    eval_files = list(daily_eval_dir.glob("*.json"))
    assert len(eval_files) >= 1

    with open(eval_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    # quote_hours should be > 0 because orders were open
    qh = float(data["quote_hours"])
    assert qh > 0.0
    assert float(data["simulated_quote_hours"]) == qh
    assert data["actual_quote_hours"] == "0.0"
    cov = float(data["market_breakdown"][cfg.condition_id]["coverage_ratio"])
    assert cov > 0.0


@pytest.mark.asyncio
async def test_shadow_collector_quote_hours_zero_when_allocation_denied(tmp_path, monkeypatch):
    """Verify that if capital allocator denies placement, no orders are open and quote_hours stays 0.0."""
    protocol = load_protocol()
    monkeypatch.setattr(protocol.data_storage, "root_path", str(tmp_path))

    cfg = MarketRewardConfig(
        condition_id="c_test_denied",
        question="Test Denied Question?",
        rewards_daily_rate=Decimal("100.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        oas=Decimal("5.0"),
        taker_fee_rate=Decimal("0.01"),
        yes_token_id="tok_y",
        no_token_id="tok_n",
    )

    collector = MarketDataCollector(storage_path=str(tmp_path))
    collector.running = True

    collector.ram_store.apply_snapshot(
        "tok_y",
        bids=[{"price": "0.48", "size": "100.0"}],
        asks=[{"price": "0.52", "size": "100.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )
    collector.ram_store.apply_snapshot(
        "tok_n",
        bids=[{"price": "0.48", "size": "100.0"}],
        asks=[{"price": "0.52", "size": "100.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )

    fsm = MarketQuotingFSM(config=cfg)
    fsms = {cfg.condition_id: fsm}
    ledger = PortfolioLedger(allocated_capital=Decimal("100.0"))

    # Deny all allocations (capital limit 0.0)
    allocator = CapitalAllocator(
        allocated_working_capital=Decimal("0.0"),
        max_unhedged_per_market=Decimal("0.0"),
        max_unhedged_total=Decimal("0.0"),
    )

    task = asyncio.create_task(
        periodic_fsm_and_scoring(
            collector=collector,
            active_markets=[cfg],
            fsms=fsms,
            ledger=ledger,
            allocator=allocator,
            protocol=protocol,
            interval_sec=0.01,
        )
    )

    await asyncio.sleep(0.05)
    collector.running = False
    await task

    daily_eval_dir = tmp_path / "daily_evaluations"
    eval_files = list(daily_eval_dir.glob("*.json"))
    assert len(eval_files) >= 1

    with open(eval_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    # quote_hours should be strictly 0.000000 because no orders were ever open
    qh = float(data["quote_hours"])
    assert qh == 0.0
    cov = float(data["market_breakdown"][cfg.condition_id]["coverage_ratio"])
    assert cov == 0.0


@pytest.mark.asyncio
async def test_shadow_collector_midnight_daily_pnl_reset(tmp_path, monkeypatch):
    """Verify that daily evaluations strictly record the Net PnL delta of that day, not cumulative."""
    import time
    from unittest.mock import patch

    protocol = load_protocol()
    monkeypatch.setattr(protocol.data_storage, "root_path", str(tmp_path))

    cfg = MarketRewardConfig(
        condition_id="c_rollover",
        question="Rollover test?",
        rewards_daily_rate=Decimal("100.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_y",
        no_token_id="tok_n",
    )
    collector = MarketDataCollector(storage_path=str(tmp_path))
    collector.running = True
    fsm = MarketQuotingFSM(config=cfg)
    fsms = {cfg.condition_id: fsm}
    ledger = PortfolioLedger(allocated_capital=Decimal("100.0"))
    allocator = CapitalAllocator(
        allocated_working_capital=Decimal("100.0"),
        max_unhedged_per_market=Decimal("25.0"),
        max_unhedged_total=Decimal("100.0"),
    )

    dates = iter(["2026-09-14", "2026-09-14", "2026-09-15", "2026-09-15"])

    def mock_strftime(fmt, *args):
        if "%Y-%m-%d" in fmt:
            try:
                return next(dates)
            except StopIteration:
                return "2026-09-15"
        return time.strftime(fmt, *args)

    # Record 10.0 rewards on Day 1
    ledger.record_daily_rewards("2026-09-14", Decimal("10.0"))

    with patch("time.strftime", side_effect=mock_strftime):
        task = asyncio.create_task(
            periodic_fsm_and_scoring(
                collector=collector,
                active_markets=[cfg],
                fsms=fsms,
                ledger=ledger,
                allocator=allocator,
                protocol=protocol,
                interval_sec=0.01,
            )
        )
        await asyncio.sleep(0.06)
        collector.running = False
        await task

    d1 = json.loads((tmp_path / "daily_evaluations" / "2026-09-14.json").read_text(encoding="utf-8"))
    d2 = json.loads((tmp_path / "daily_evaluations" / "2026-09-15.json").read_text(encoding="utf-8"))

    assert Decimal(d1["net_pnl"]) == Decimal("10.0")
    assert Decimal(d2["net_pnl"]) == Decimal("0.0")
    assert Decimal(d2["cumulative_net_pnl"]) == Decimal("10.0")
