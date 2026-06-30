import pytest
from polyflip.trading.position_sizing import (
    compute_kelly_fraction, compute_bet_size,
    compute_edge, is_in_dead_zone, apply_polymarket_fee
)

def test_kelly_positive_edge():
    # Покупаем YES по 0.6, вероятность выигрыша 0.7 → должен быть > 0
    k = compute_kelly_fraction(win_prob=0.7, buy_price=0.6)
    assert k > 0

def test_kelly_negative_edge():
    # Покупаем YES по 0.8, вероятность выигрыша 0.5 → Kelly = 0
    k = compute_kelly_fraction(win_prob=0.5, buy_price=0.8)
    assert k == 0.0

def test_kelly_bounds():
    # Всегда в [0, 1]
    for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for price in [0.2, 0.4, 0.6, 0.8]:
            k = compute_kelly_fraction(prob, price)
            assert 0.0 <= k <= 1.0, f"Kelly={k} out of bounds for prob={prob}, price={price}"

def test_bet_size_respects_limits():
    bet = compute_bet_size(
        kelly_fraction=0.5, capital_usdc=1000,
        kelly_multiplier=0.25, min_bet_usdc=5, max_bet_usdc=50
    )
    assert 5 <= bet <= 50

def test_bet_size_zero_kelly():
    bet = compute_bet_size(0.0, 1000, 0.25, 5, 50)
    assert bet == 0.0

def test_edge_positive():
    # 70% шанс, цена 0.60 → edge > 0
    assert compute_edge(0.7, 0.6) > 0

def test_edge_negative():
    # 50% шанс, цена 0.80 → edge < 0
    assert compute_edge(0.5, 0.8) < 0

def test_dead_zone_center():
    assert is_in_dead_zone(0.5, dead_zone_width=0.1) is True

def test_dead_zone_outside():
    assert is_in_dead_zone(0.75, dead_zone_width=0.1) is False

def test_polymarket_fee():
    # 10$ gross → 9.98$ после комиссии 0.2%
    assert abs(apply_polymarket_fee(10.0) - 9.98) < 0.001
