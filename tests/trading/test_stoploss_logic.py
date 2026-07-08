# tests/trading/test_stoploss_logic.py — unit-тесты чистой логики (без БД)

from polyflip.trading.stoploss import compute_stop_price, evaluate_stop_loss


def test_compute_stop_price_50pct():
    assert compute_stop_price(0.5, 50.0) == 0.25


def test_compute_stop_price_clamp_low():
    # entry=0.01, pct=99 → raw=0.0001 → clamp → 0.01
    assert compute_stop_price(0.01, 99.0) == 0.01


def test_compute_stop_price_invalid_pct():
    import pytest
    with pytest.raises(ValueError):
        compute_stop_price(0.5, 0.0)
    with pytest.raises(ValueError):
        compute_stop_price(0.5, 100.0)


def test_evaluate_triggers_when_bid_below_stop():
    decision = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.20)
    assert decision.should_sell is True
    assert decision.stop_price == 0.25


def test_evaluate_no_trigger_when_bid_above_stop():
    decision = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.30)
    assert decision.should_sell is False


def test_evaluate_triggers_exactly_at_stop():
    """На границе — bid == stop_price → триггер (<=)."""
    decision = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.25)
    assert decision.should_sell is True
