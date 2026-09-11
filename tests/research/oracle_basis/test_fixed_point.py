"""E18 fixed-point parsing without floats (protocol self-check: OB arithmetic)."""

import pytest
from decimal import Decimal

from polyflip.collector.rtds_collector import E18, parse_price_e18, round_half_up_div


def test_e18_scale_is_exact():
    assert E18 == 10**18
    assert parse_price_e18("10.000000000000000001") == 10 * E18 + 1
    assert parse_price_e18(Decimal("0.5")) == E18 // 2


def test_float_and_bool_prices_rejected():
    for bad in (10.0, 0.5, True, False, None, 10, "", "abc", "-1", "0"):
        with pytest.raises((TypeError, ValueError)):
            parse_price_e18(bad)


def test_round_half_up_div():
    assert round_half_up_div(3, 2) == 2
    assert round_half_up_div(5, 2) == 3
    assert round_half_up_div(4, 2) == 2
    with pytest.raises(ValueError):
        round_half_up_div(-1, 2)
    with pytest.raises(ValueError):
        round_half_up_div(1, 0)
