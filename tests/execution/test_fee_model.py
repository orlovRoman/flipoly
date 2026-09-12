from decimal import Decimal

import pytest

from polyflip.execution.gateways.fake import FakeExecutionGateway
from polyflip.trading.fee_model import fee_for_fill, fee_per_share


def test_price_dependent_fee_matches_canonical_curve() -> None:
    expected = Decimal("0.07") * Decimal("0.27") * Decimal("0.73")
    assert fee_per_share(Decimal("0.27"), fee_rate="0.07") == expected


def test_fee_for_fill_rounds_at_venue_precision() -> None:
    fee = fee_for_fill("0.50", "10", fee_rate="0.07")
    assert fee == Decimal("0.17500")


def test_fake_gateway_uses_shared_fee_calculator() -> None:
    gateway = FakeExecutionGateway(
        profile="INSTANT",
        fee_model="POLYMARKET_PRICE_DEPENDENT",
        fee_rate="0.07",
        fee_exponent="1",
    )
    assert gateway._fee_per_share(Decimal("0.27")) == fee_per_share(
        Decimal("0.27"), fee_rate="0.07", fee_exponent="1"
    )


def test_fee_rejects_invalid_price() -> None:
    with pytest.raises(ValueError):
        fee_per_share("1.01", fee_rate="0.07")

