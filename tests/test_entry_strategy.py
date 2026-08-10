from polyflip.trading.trading_config import parse_trading_settings
import pytest
from datetime import datetime, timedelta
from polyflip.backtesting.runner import BacktestRunner
from polyflip.backtesting.market_replay import MarketReplay, MarketTick


def make_tick(mid: float, ask: float, time_left: float, dt: datetime, market_id: str = "m1") -> MarketTick:
    spread = abs(ask - mid) * 2
    return MarketTick(
        market_id=market_id, asset="BTC", time_left_min=time_left,
        mid_price=mid, spread=spread, volume_5min=1000.0,
        price_velocity=0.0, hour_of_day=12, final_outcome="YES",
        recorded_at=dt
    )


def test_entry_strategy_first():
    """first должен брать самый первый подходящий сигнал (самый ранний)."""
    now = datetime.now()
    ticks = [
        make_tick(0.70, 0.71, 30.0, now - timedelta(minutes=30)),
        make_tick(0.70, 0.71, 20.0, now - timedelta(minutes=20)),
        make_tick(0.70, 0.71, 10.0, now - timedelta(minutes=10)),
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m1"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "first",
        "STRATEGY_MODE": "OUTSIDER",
        "TRADE_ON_FLIP": "true",
        "FLIP_THRESHOLD": "0.30",
        "OUTSIDER_PWIN_DISCOUNT": "1.0",
        "OUTS_MIN_EDGE": "0.01",
        "OUTSIDER_MAX_PRICE": "0.45",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "OUTS_MIN_TIME_LEFT_SEC": "60",
        "OUTS_MAX_TIME_LEFT_SEC": "3600",
        "OUTS_MIN_TIME_LEFT_MIN": "1.0",
        "OUTS_MAX_TIME_LEFT_MIN": "60.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "", prediction_overrides={("m1", 30.0): 0.40, ("m1", 20.0): 0.50, ("m1", 10.0): 0.45})
    runner.run_market(replay)
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.timestamp == ticks[0].recorded_at


def test_entry_strategy_best_edge():
    """best_edge должен выбрать тик со 2-й минуты (edge выше)."""
    now = datetime.now()
    ticks = [
        make_tick(0.70, 0.71, 30.0, now - timedelta(minutes=30)),
        make_tick(0.70, 0.71, 20.0, now - timedelta(minutes=20)),  # Best edge (p_flip 0.50)
        make_tick(0.70, 0.71, 10.0, now - timedelta(minutes=10)),
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m1"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "best_edge",
        "STRATEGY_MODE": "OUTSIDER",
        "TRADE_ON_FLIP": "true",
        "FLIP_THRESHOLD": "0.30",
        "OUTSIDER_PWIN_DISCOUNT": "1.0",
        "OUTS_MIN_EDGE": "0.01",
        "OUTSIDER_MAX_PRICE": "0.45",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "OUTS_MIN_TIME_LEFT_SEC": "60",
        "OUTS_MAX_TIME_LEFT_SEC": "3600",
        "OUTS_MIN_TIME_LEFT_MIN": "1.0",
        "OUTS_MAX_TIME_LEFT_MIN": "60.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "", prediction_overrides={("m1", 30.0): 0.40, ("m1", 20.0): 0.50, ("m1", 10.0): 0.45})
    runner.run_market(replay)
    
    assert len(runner.trader.trades) == 1
    trade = runner.trader.trades[0]
    assert trade.timestamp == ticks[1].recorded_at


def test_confirmed_resets_on_action_change():
    """
    Если тик 1 = BUY_NO, тик 2 = BUY_YES — счётчик сбрасывается.
    Нет 2 подтверждений одного направления → нет сделки.
    """
    now = datetime.now()
    ticks = [
        make_tick(0.70, 0.71, 30.0, now - timedelta(minutes=30), market_id="m_reset"),  # YES is fav -> BUY_NO
        make_tick(0.30, 0.31, 20.0, now - timedelta(minutes=20), market_id="m_reset"),  # NO is fav -> BUY_YES (смена!)
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m_reset"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "confirmed",
        "STRATEGY_MODE": "OUTSIDER",
        "TRADE_ON_FLIP": "true",
        "FLIP_THRESHOLD": "0.30",
        "OUTSIDER_PWIN_DISCOUNT": "1.0",
        "OUTS_MIN_EDGE": "0.01",
        "OUTSIDER_MAX_PRICE": "0.45",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "OUTS_MIN_TIME_LEFT_SEC": "60",
        "OUTS_MAX_TIME_LEFT_SEC": "3600",
        "OUTS_MIN_TIME_LEFT_MIN": "1.0",
        "OUTS_MAX_TIME_LEFT_MIN": "60.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "", prediction_overrides={("m_reset", 30.0): 0.40, ("m_reset", 20.0): 0.40})
    runner.run_market(replay)

    assert len(runner.trader.trades) == 0, (
        f"При смене направления confirmed не должен входить, "
        f"но совершил {len(runner.trader.trades)} сделку(и)"
    )


def test_confirmed_enters_after_stable_sequence():
    """
    Если 3 тика подряд одного направления — вход на 2-м (первые 2 = подтверждение).
    """
    now = datetime.now()
    ticks = [
        make_tick(0.70, 0.71, 30.0, now - timedelta(minutes=30), market_id="m_stable"),  # BUY_NO #1
        make_tick(0.70, 0.71, 20.0, now - timedelta(minutes=20), market_id="m_stable"),  # BUY_NO #2 ← вход здесь
        make_tick(0.70, 0.71, 10.0, now - timedelta(minutes=10), market_id="m_stable"),  # BUY_NO #3
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m_stable"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "confirmed",
        "STRATEGY_MODE": "OUTSIDER",
        "TRADE_ON_FLIP": "true",
        "FLIP_THRESHOLD": "0.30",
        "OUTSIDER_PWIN_DISCOUNT": "1.0",
        "OUTS_MIN_EDGE": "0.01",
        "OUTSIDER_MAX_PRICE": "0.45",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "OUTS_MIN_TIME_LEFT_SEC": "60",
        "OUTS_MAX_TIME_LEFT_SEC": "3600",
        "OUTS_MIN_TIME_LEFT_MIN": "1.0",
        "OUTS_MAX_TIME_LEFT_MIN": "60.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "", prediction_overrides={("m_stable", 30.0): 0.40, ("m_stable", 20.0): 0.40, ("m_stable", 10.0): 0.40})
    runner.run_market(replay)

    assert len(runner.trader.trades) == 1
    assert runner.trader.trades[0].timestamp == ticks[1].recorded_at, (
        f"Вход должен быть на тике [1] ({ticks[1].recorded_at}), "
        f"но был на {runner.trader.trades[0].timestamp}"
    )


def test_bet_sizing_consistency_between_resolve_and_liquidity():
    """
    _resolve_final_bet должен давать тот же результат
    что и compute_bet_size_with_liquidity (так как он его вызывает под капотом).
    """
    from polyflip.trading.decision_logic import _resolve_final_bet
    from polyflip.trading.position_sizing import compute_bet_size_with_liquidity

    edge = 0.12
    volume = 300.0
    min_bet = 5.0
    max_bet = 50.0
    min_edge = 0.05
    max_edge = 0.20
    fraction = 0.05

    config = {
        "TRADE_BET_SIZE_USDC": str(min_bet),
        "MAX_BET_SIZE_USDC": str(max_bet),
        "OUTS_MIN_EDGE": str(min_edge),
        "MAX_BET_EDGE": str(max_edge),
        "LIQUIDITY_FRACTION": str(fraction),
    }

    bet_via_logic = _resolve_final_bet(edge, volume, parse_trading_settings(config), is_outsider=True)

    bet_via_sizing = compute_bet_size_with_liquidity(
        edge=edge, volume_5min=volume,
        min_bet_usdc=min_bet, max_bet_usdc=max_bet,
        min_edge=min_edge, max_edge=max_edge,
        liquidity_fraction=fraction,
    )

    assert abs(bet_via_logic - bet_via_sizing) < 0.01, (
        f"Расхождение в bet sizing: "
        f"logic={bet_via_logic}, sizing={bet_via_sizing}. "
        "Нужно унифицировать вызовы через compute_bet_size_with_liquidity."
    )


def test_evaluate_tick_no_import_overhead():
    """
    _evaluate_tick не должен делать import внутри вызова.
    Проверяем: 1000 вызовов должны выполниться быстро (< 1 сек).
    """
    import time
    now = datetime.now()
    tick = make_tick(0.70, 0.71, 30.0, now)

    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = [tick]
    replay.market_id = "m_perf"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "first",
        "STRATEGY_MODE": "OUTSIDER",
        "TRADE_ON_FLIP": "true",
        "OUTS_MIN_EDGE": "0.01",
        "OUTSIDER_MAX_PRICE": "0.45",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "OUTS_MIN_TIME_LEFT_SEC": "60",
        "OUTS_MAX_TIME_LEFT_SEC": "3600",
        "OUTS_MIN_TIME_LEFT_MIN": "1.0",
        "OUTS_MAX_TIME_LEFT_MIN": "60.0",
        "TRADE_MIN_PRICE": "0.05",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    runner = BacktestRunner(config, b"", "", prediction_overrides={("m_perf", 30.0): 0.40})

    start = time.perf_counter()
    for _ in range(1000):
        runner._evaluate_tick(tick)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"1000 вызовов _evaluate_tick заняли {elapsed:.2f}s > 1s"
    )

