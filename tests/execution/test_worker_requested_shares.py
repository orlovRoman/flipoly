from decimal import Decimal

import pytest


@pytest.mark.parametrize(
    ("requested", "spend", "price", "side", "expected"),
    [
        (None, "1.10", "0.82", "BUY", Decimal("1.10") / Decimal("0.82")),
        (Decimal("0"), "1.10", "0.82", "BUY", Decimal("1.10") / Decimal("0.82")),
        (Decimal("2"), "1.10", "0.82", "BUY", Decimal("2")),
        (None, "1.10", "0.82", "SELL", Decimal("0")),
        (None, "0", "0.82", "BUY", Decimal("0")),
        (None, "1.10", "0", "BUY", Decimal("0")),
    ],
)
def test_resolve_requested_shares(requested, spend, price, side, expected):
    from polyflip.execution.worker import _resolve_requested_shares

    result = _resolve_requested_shares(
        requested_shares=requested,
        max_spend_usdc=Decimal(spend),
        limit_price=Decimal(price),
        side=side,
    )

    assert result == expected
