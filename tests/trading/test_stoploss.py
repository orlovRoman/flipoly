import pytest
from polyflip.trading.stoploss import compute_stop_price, evaluate_stop_loss

def test_compute_stop_price_50pct():
    assert compute_stop_price(0.5, 50.0) == pytest.approx(0.25)

def test_compute_stop_price_20pct():
    assert compute_stop_price(0.8, 20.0) == pytest.approx(0.64)

def test_compute_stop_price_clamp_low():
    # entry=0.02, pct=99 → raw=0.0002, clamp → 0.01
    assert compute_stop_price(0.02, 99.0) == pytest.approx(0.01)

def test_compute_stop_price_invalid_pct():
    with pytest.raises(ValueError):
        compute_stop_price(0.5, 0.0)
    with pytest.raises(ValueError):
        compute_stop_price(0.5, 100.0)
    with pytest.raises(ValueError):
        compute_stop_price(0.5, -10.0)

def test_evaluate_triggers_when_bid_below_stop():
    dec = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.20)
    assert dec.should_sell is True
    assert dec.stop_price == pytest.approx(0.25)

def test_evaluate_no_trigger_when_bid_above_stop():
    dec = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.30)
    assert dec.should_sell is False

def test_evaluate_triggers_at_exact_stop():
    dec = evaluate_stop_loss(entry_price=0.5, stop_loss_pct=50.0, current_bid=0.25)
    assert dec.should_sell is True  # bid <= stop (включительно)
