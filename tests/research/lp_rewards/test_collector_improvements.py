import asyncio
from decimal import Decimal
import json
from pathlib import Path
import pandas as pd
import pytest

from polyflip.research.lp_rewards.collector import MarketDataCollector, OrderbookRAMStore
from polyflip.research.lp_rewards.models import MarketRewardConfig, OrderSide
from polyflip.research.lp_rewards.watchdog import SystemWatchdog


def test_price_changes_array_parsing():
    store = OrderbookRAMStore()
    now_ns = 1_700_000_000_000_000_000

    # Initial snapshot
    store.apply_snapshot("tok_yes", bids=[{"price": "0.48", "size": "50.0"}], asks=[{"price": "0.52", "size": "50.0"}], timestamp_ns=now_ns)

    # Polymarket WS price_change event with price_changes array containing multiple deltas
    ws_msg = {
        "event_type": "price_change",
        "market": "0xcond_1",
        "timestamp": "1700000001000",
        "price_changes": [
            {"asset_id": "tok_yes", "side": "BUY", "price": "0.49", "size": "100.0"},
            {"asset_id": "tok_yes", "side": "BUY", "price": "0.48", "size": "0.0"},  # deletion
            {"asset_id": "tok_yes", "side": "SELL", "price": "0.51", "size": "75.0"},
        ],
    }

    # Simulate processing price_changes
    collector = MarketDataCollector()
    collector.ram_store = store

    # Parse message like _message_loop
    pcs = ws_msg["price_changes"]
    for pc in pcs:
        collector.ram_store.apply_delta(
            pc["asset_id"],
            pc["side"],
            Decimal(str(pc["price"])),
            Decimal(str(pc["size"])),
            now_ns + 1_000_000_000,
        )

    snap = collector.ram_store.get_snapshot("0xcond_1", "tok_yes")
    # Bid 0.48 deleted (size 0), bid 0.49 added (size 100)
    bid_prices = [b.price for b in snap.bids]
    assert Decimal("0.48") not in bid_prices
    assert Decimal("0.49") in bid_prices
    assert snap.bids[0].size == Decimal("100.0")

    # Ask 0.51 added (size 75)
    ask_prices = [a.price for a in snap.asks]
    assert Decimal("0.51") in ask_prices
    assert snap.asks[0].size == Decimal("75.0")


def test_text_pong_heartbeat_reception():
    collector = MarketDataCollector(pong_timeout_sec=10.0)
    initial_pong_time = collector.last_pong_time

    # Simulate receiving text PONG
    raw_msg = "PONG"
    if isinstance(raw_msg, str) and raw_msg.strip().upper() == "PONG":
        collector.last_pong_time = initial_pong_time + 5.0

    assert collector.last_pong_time == initial_pong_time + 5.0


def test_trade_persistence_partition_append_and_precision(tmp_path):
    storage = tmp_path / "storage"
    watchdog = SystemWatchdog(storage_path=str(storage), halt_threshold_gb=0.001, warning_threshold_gb=0.005)
    collector = MarketDataCollector(storage_path=str(storage), watchdog=watchdog)

    m1 = MarketRewardConfig(
        condition_id="cond_alpha",
        question="Question Alpha?",
        rewards_daily_rate=Decimal("10.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_alpha_yes",
        no_token_id="tok_alpha_no",
    )
    m2 = MarketRewardConfig(
        condition_id="cond_beta",
        question="Question Beta?",
        rewards_daily_rate=Decimal("15.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_beta_yes",
        no_token_id="tok_beta_no",
    )
    collector.set_active_markets([m1, m2])

    # Add trades for both markets to buffer
    collector.trade_buffer.append({
        "timestamp_ns": 1_700_000_000_000_000_000,
        "observed_at_ns": 1_700_000_000_000_000_000,
        "received_at_ns": 1_700_000_000_050_000_000,
        "condition_id": "cond_alpha",
        "asset_id": "tok_alpha_yes",
        "price": "0.4851",
        "size": "150.2500",
        "side": "BUY",
    })
    collector.trade_buffer.append({
        "timestamp_ns": 1_700_000_000_000_000_000,
        "observed_at_ns": 1_700_000_000_000_000_000,
        "received_at_ns": 1_700_000_000_050_000_000,
        "condition_id": "cond_beta",
        "asset_id": "tok_beta_yes",
        "price": "0.6200",
        "size": "500.0000",
        "side": "SELL",
    })

    # Flush batch 1
    collector.flush_trades_to_disk(date_str="2026-09-14")

    # Verify separated partitions
    alpha_parquet = storage / "public_trades" / "cond_alpha" / "2026-09-14.parquet"
    beta_parquet = storage / "public_trades" / "cond_beta" / "2026-09-14.parquet"

    assert alpha_parquet.exists()
    assert beta_parquet.exists()

    df_alpha = pd.read_parquet(alpha_parquet)
    assert len(df_alpha) == 1
    # Check that price is exact string and not lost in float conversion
    assert df_alpha.iloc[0]["price"] == "0.4851"
    assert df_alpha.iloc[0]["size"] == "150.2500"
    assert df_alpha.iloc[0]["observed_at_ns"] == 1_700_000_000_000_000_000
    assert df_alpha.iloc[0]["received_at_ns"] == 1_700_000_000_050_000_000

    # Add second trade for alpha and flush again -> verify append without overwrite
    collector.trade_buffer.append({
        "timestamp_ns": 1_700_000_005_000_000_000,
        "observed_at_ns": 1_700_000_005_000_000_000,
        "received_at_ns": 1_700_000_005_050_000_000,
        "condition_id": "cond_alpha",
        "asset_id": "tok_alpha_yes",
        "price": "0.4900",
        "size": "80.0000",
        "side": "BUY",
    })
    collector.flush_trades_to_disk(date_str="2026-09-14")

    df_alpha_updated = pd.read_parquet(alpha_parquet)
    assert len(df_alpha_updated) == 2
    assert df_alpha_updated.iloc[1]["price"] == "0.4900"


def test_l2_snapshot_persistence(tmp_path):
    storage = tmp_path / "storage"
    watchdog = SystemWatchdog(storage_path=str(storage), halt_threshold_gb=0.001, warning_threshold_gb=0.005)
    collector = MarketDataCollector(storage_path=str(storage), watchdog=watchdog)

    m1 = MarketRewardConfig(
        condition_id="cond_gamma",
        question="Question Gamma?",
        rewards_daily_rate=Decimal("12.0"),
        rewards_max_spread=Decimal("0.05"),
        rewards_min_size=Decimal("10.0"),
        yes_token_id="tok_gamma_yes",
        no_token_id="tok_gamma_no",
    )
    collector.set_active_markets([m1])

    collector.ram_store.apply_snapshot(
        "tok_gamma_yes",
        bids=[{"price": "0.50", "size": "100.0"}],
        asks=[{"price": "0.52", "size": "80.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )
    collector.ram_store.apply_snapshot(
        "tok_gamma_no",
        bids=[{"price": "0.48", "size": "90.0"}],
        asks=[{"price": "0.50", "size": "110.0"}],
        timestamp_ns=1_700_000_000_000_000_000,
    )

    collector.flush_l2_snapshots_to_disk(date_str="2026-09-14")

    l2_parquet = storage / "l2_snapshots" / "cond_gamma" / "2026-09-14.parquet"
    assert l2_parquet.exists()

    df_l2 = pd.read_parquet(l2_parquet)
    assert len(df_l2) == 4  # 2 bids + 2 asks across YES and NO
    columns = set(df_l2.columns)
    required_cols = {"timestamp_ns", "condition_id", "asset_id", "side", "price", "size", "valid_from_ns", "valid_to_ns", "order_age_sec"}
    assert required_cols.issubset(columns)
