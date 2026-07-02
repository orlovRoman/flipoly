import pytest
from polyflip.trading.position_sizing import compute_bet_size_with_liquidity, compute_bet_size_edge_scaled
from polyflip.trading.decision_logic import decide_ml_trend
from polyflip.trading.feature_builder import MarketSignal


def test_liquidity_cap_active_on_thin_market():
    """На тонком рынке ставка ограничивается liquidity cap."""
    # volume_5min=40, fraction=0.05 → cap=max(2.0, 5.0)=5.0
    # raw_bet при edge=0.4 может быть 30+ USDC
    bet = compute_bet_size_with_liquidity(
        edge=0.4, volume_5min=40.0,
        min_bet_usdc=5.0, max_bet_usdc=50.0,
        liquidity_fraction=0.05
    )
    assert bet == 5.0, f"Ожидали 5.0 (liquidity cap), получили {bet}"


def test_liquidity_cap_inactive_on_deep_market():
    """На глубоком рынке cap не мешает масштабированию."""
    # volume_5min=10000, fraction=0.05 → cap=500 >> max_bet=50
    bet = compute_bet_size_with_liquidity(
        edge=0.4, volume_5min=10000.0,
        min_bet_usdc=5.0, max_bet_usdc=50.0,
        min_edge=0.0, max_edge=0.5,
        liquidity_fraction=0.05
    )
    raw = compute_bet_size_edge_scaled(0.4, 5.0, 50.0, 0.0, 0.5)
    assert bet == raw, "На ликвидном рынке cap не должен срабатывать"


def test_apply_liquidity_cap_never_below_min_bet():
    """Liquidity cap не должен опускать ставку ниже min_bet."""
    # volume_5min=1.0, fraction=0.05 → cap=0.05 < min_bet=5.0 → результат 5.0
    result = compute_bet_size_with_liquidity(
        edge=0.4, volume_5min=1.0,
        min_bet_usdc=5.0, max_bet_usdc=50.0,
        min_edge=0.0, max_edge=0.2,
        liquidity_fraction=0.05
    )
    assert result == 5.0


def test_apply_liquidity_cap_proportional():
    """Проверка пропорций: cap = max(vol * fraction, min_bet)."""
    # volume=200, fraction=0.05 → cap=10.0; bet > 10 → результат 10.0
    result = compute_bet_size_with_liquidity(
        edge=0.4, volume_5min=200.0,
        min_bet_usdc=5.0, max_bet_usdc=50.0,
        min_edge=0.0, max_edge=0.2,
        liquidity_fraction=0.05
    )
    assert result == 10.0


def test_decide_ml_trend_respects_liquidity():
    """decide_ml_trend должен ограничивать ставку по ликвидности."""
    signal = MarketSignal(
        asset="BTC", mid_price=0.75, spread=0.001,
        volume_5min=20.0,  # тонкий рынок! cap = max(1.0, 5.0) = 5.0
        price_velocity=0.0, hour_of_day=12, time_left_min=20.0
    )
    config = {
        "NO_FLIP_THRESHOLD": "0.40",
        "FAVORITE_THRESHOLD": "0.60",
        "MIN_EDGE": "-0.05",
        "MAX_EDGE": "0.20",
        "TRADE_BET_SIZE_USDC": "5",
        "MAX_BET_SIZE_USDC": "50",
        "AUTO_DEAD_ZONE_WIDTH": "0.05",
        "TRADE_MIN_PRICE": "0.50",
        "TRADE_MAX_PRICE": "0.95",
        "LIQUIDITY_FRACTION": "0.05",
    }
    # p_flip=0.20 < NO_FLIP_THRESHOLD=0.40 → ML_TREND → BUY_YES
    decision = decide_ml_trend(signal, p_flip=0.20, config=config)
    if decision.action != "SKIP":
        assert decision.bet_size_usdc <= 5.0, (
            f"На тонком рынке bet={decision.bet_size_usdc} превышает liquidity cap 5.0"
        )
