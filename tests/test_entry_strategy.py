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


def test_confirmed_resets_on_action_change():
    """
    Если тик 1 = BUY_YES, тик 2 = BUY_NO — счётчик сбрасывается.
    Нет 2 подтверждений одного направления → нет сделки.
    """
    now = datetime.now()
    ticks = [
        make_tick(0.80, 0.70, 30.0, now - timedelta(minutes=30)),  # BUY_YES (edge>0)
        make_tick(0.20, 0.30, 20.0, now - timedelta(minutes=20)),  # BUY_NO  (смена!)
        # Только один тик после смены — нет 2 подтверждений
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m_reset"
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
        make_tick(0.80, 0.70, 30.0, now - timedelta(minutes=30)),  # BUY_YES #1
        make_tick(0.82, 0.72, 20.0, now - timedelta(minutes=20)),  # BUY_YES #2 ← вход здесь
        make_tick(0.84, 0.74, 10.0, now - timedelta(minutes=10)),  # BUY_YES #3
    ]
    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = ticks
    replay.market_id = "m_stable"
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
    # Вход должен быть на тике [1] (второй тик = первое подтверждение)
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
        "MIN_EDGE": str(min_edge),
        "MAX_EDGE": str(max_edge),
        "LIQUIDITY_FRACTION": str(fraction),
    }

    # Путь через decision_logic
    bet_via_logic = _resolve_final_bet(edge, volume, config)

    # Путь через position_sizing
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
    Проверяем косвенно: 1000 вызовов должны выполниться быстро (< 1 сек).
    """
    import time
    now = datetime.now()
    tick = make_tick(0.80, 0.70, 30.0, now)

    replay = MarketReplay.__new__(MarketReplay)
    replay.ticks = [tick]
    replay.market_id = "m_perf"
    replay.asset = "BTC"
    replay.final_outcome = "YES"

    config = {
        "ENTRY_STRATEGY": "first",
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

    start = time.perf_counter()
    for _ in range(1000):
        runner._evaluate_tick(tick)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"1000 вызовов _evaluate_tick заняли {elapsed:.2f}s > 1s — "
        "вероятно, import внутри цикла замедляет выполнение"
    )
