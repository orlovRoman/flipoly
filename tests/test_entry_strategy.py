import pytest
from datetime import datetime, timedelta
from polyflip.backtesting.runner import BacktestRunner
from polyflip.backtesting.market_replay import MarketReplay, MarketTick


def make_tick(mid: float, ask: float, time_left: float, dt: datetime) -> MarketTick:
    # spread / 2 = ask - mid => spread = (ask - mid) * 2
    spread = (ask - mid) * 2
    return MarketTick(
        market_id="m1", asset="BTC", time_left_min=time_left,
        mid_price=mid, spread=spread, volume_5min=1000.0,
        price_velocity=0.0, hour_of_day=12, final_outcome="YES",
        recorded_at=dt
    )


def test_entry_strategy_first():
    """first должен брать самый первый подходящий сигнал (самый ранний)."""
    now = datetime.now()
    ticks = [
        make_tick(0.60, 0.61, 30.0, now - timedelta(minutes=30)),  # edge = 0.60/0.61 - 1 = -0.016
        make_tick(0.80, 0.70, 20.0, now - timedelta(minutes=20)),  # edge = 0.80/0.70 - 1 = 0.14
        make_tick(0.90, 0.80, 10.0, now - timedelta(minutes=10)),  # edge = 0.90/0.80 - 1 = 0.125
    ]
    # Насильно подменяем ticks в replay, обходя __init__
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m1"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "first",
        "STRATEGY_MODE": "PURE_FAVORITE",
        "FAVORITE_THRESHOLD": "0.55",
        "FAVORITE_MIN_EDGE": "-0.05",  # чтобы первый тик прошел
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "MIN_TIME_LEFT_MIN": "1.0",
        "MAX_TIME_LEFT_MIN": "60.0",
        "AUTO_DEAD_ZONE_WIDTH": "0.05",
        "TRADE_MIN_PRICE": "0.50",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "")
    runner.run_market(replay)
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.timestamp == ticks[0].recorded_at


def test_entry_strategy_best_edge():
    """best_edge должен выбрать тик со 2-й минуты (edge 0.14 > 0.125 и -0.016)."""
    now = datetime.now()
    ticks = [
        make_tick(0.60, 0.61, 30.0, now - timedelta(minutes=30)),
        make_tick(0.80, 0.70, 20.0, now - timedelta(minutes=20)),  # Best edge (0.14)
        make_tick(0.90, 0.80, 10.0, now - timedelta(minutes=10)),
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m1"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "best_edge",
        "STRATEGY_MODE": "PURE_FAVORITE",
        "FAVORITE_THRESHOLD": "0.55",
        "FAVORITE_MIN_EDGE": "-0.05",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "MIN_TIME_LEFT_MIN": "1.0",
        "MAX_TIME_LEFT_MIN": "60.0",
        "AUTO_DEAD_ZONE_WIDTH": "0.05",
        "TRADE_MIN_PRICE": "0.50",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "")
    runner.run_market(replay)
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.timestamp == ticks[1].recorded_at


def test_entry_strategy_confirmed():
    """confirmed должен пропустить 1-й тик и войти на 2-м, так как 2 подряд имеют action != SKIP."""
    now = datetime.now()
    ticks = [
        make_tick(0.60, 0.61, 30.0, now - timedelta(minutes=30)),  # 1st
        make_tick(0.80, 0.70, 20.0, now - timedelta(minutes=20)),  # 2nd (Confirmed!)
        make_tick(0.90, 0.80, 10.0, now - timedelta(minutes=10)),
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m1"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "confirmed",
        "STRATEGY_MODE": "PURE_FAVORITE",
        "FAVORITE_THRESHOLD": "0.55",
        "FAVORITE_MIN_EDGE": "-0.05",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "MIN_TIME_LEFT_MIN": "1.0",
        "MAX_TIME_LEFT_MIN": "60.0",
        "AUTO_DEAD_ZONE_WIDTH": "0.05",
        "TRADE_MIN_PRICE": "0.50",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "")
    runner.run_market(replay)
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.timestamp == ticks[1].recorded_at
