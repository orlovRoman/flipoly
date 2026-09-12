"""Canonical price-dependent fee calculations shared by PAPER and policies.

The fee parameter is the Polymarket curve coefficient, not a flat percentage
of turnover. Keeping this calculation in one module prevents PAPER, replay,
and decision code from silently using different economics.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def as_decimal(value: Any, default: str = "0") -> Decimal:
    """Return a finite Decimal without allowing malformed runtime settings."""
    try:
        parsed = Decimal(str(value if value is not None else default))
    except (ArithmeticError, TypeError, ValueError):
        parsed = Decimal(default)
    return parsed if parsed.is_finite() else Decimal(default)


def fee_per_share(
    price: Decimal | float | str,
    *,
    fee_rate: Decimal | float | str,
    fee_exponent: Decimal | float | str = Decimal("1"),
    role: str = "TAKER",
    maker_fee_rate: Decimal | float | str = Decimal("0"),
    fee_model: str = "POLYMARKET_PRICE_DEPENDENT",
) -> Decimal:
    """Calculate the fee for one outcome share.

    ``POLYMARKET_PRICE_DEPENDENT`` uses
    ``rate * (price * (1 - price)) ** exponent``. ``FLAT_NOTIONAL`` is
    retained for deterministic legacy tests and explicit scenarios.
    """
    p = as_decimal(price)
    rate = as_decimal(maker_fee_rate if str(role).upper() == "MAKER" else fee_rate)
    exponent = as_decimal(fee_exponent, "1")
    if p < 0 or p > 1 or rate < 0 or exponent < 0:
        raise ValueError("fee inputs must be finite, price must be in [0, 1]")
    model = str(fee_model or "POLYMARKET_PRICE_DEPENDENT").strip().upper()
    if model == "POLYMARKET_PRICE_DEPENDENT":
        return rate * ((p * (Decimal("1") - p)) ** exponent)
    if model == "FLAT_NOTIONAL":
        return p * rate
    raise ValueError(f"unsupported fee model: {fee_model}")


def fee_for_fill(
    price: Decimal | float | str,
    shares: Decimal | float | str,
    **kwargs: Any,
) -> Decimal:
    """Calculate and round the fee for a fill using venue precision."""
    amount = as_decimal(shares) * fee_per_share(price, **kwargs)
    return amount.quantize(Decimal("0.00001"))

