import pytest
from polyflip.trading.position_sizing import (
    compute_bet_size_edge_scaled,
    compute_edge, is_in_dead_zone, apply_polymarket_fee
)

def test_edge_scaled_min():
    assert compute_bet_size_edge_scaled(0.05, 5.0, 50.0) == 5.0

def test_edge_scaled_max():
    assert compute_bet_size_edge_scaled(0.40, 5.0, 50.0) == 50.0

def test_edge_scaled_mid():
    # edge=0.225 (50% пути от 0.05 до 0.40) → 50% пути от 5 до 50 = 27.5
    result = compute_bet_size_edge_scaled(0.225, 5.0, 50.0)
    assert result == pytest.approx(27.5, abs=0.1)

def test_edge_zero_returns_zero():
    assert compute_bet_size_edge_scaled(0.0, 5.0, 50.0) == 0.0

def test_no_kelly_functions_exist():
    """Убеждаемся что Kelly-функции удалены."""
    import polyflip.trading.position_sizing as ps
    assert not hasattr(ps, "compute_kelly_fraction")
    assert not hasattr(ps, "compute_bet_size")

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

def test_compute_edge_positive_expectation():
    """При win_prob > buy_price edge должен быть положительным."""
    from polyflip.trading.position_sizing import compute_edge
    edge = compute_edge(0.7, 0.6)
    # EV/bet - 1: (0.7/0.6)-1 = 0.1667
    assert abs(edge - 0.1667) < 1e-3
    assert edge > 0, "edge должен быть положительным при win_prob > price"

def test_compute_edge_zero_price():
    from polyflip.trading.position_sizing import compute_edge
    assert compute_edge(0.7, 0.0) == -1.0

def test_compute_edge_consistent_with_engine():
    """engine.py не должен вручную вычислять edge — только через compute_edge."""
    import inspect
    from polyflip.trading import engine
    source = inspect.getsource(engine)
    assert "p_win - buy_price" not in source, \
        "edge должен вычисляться через compute_edge(), не вручную"
