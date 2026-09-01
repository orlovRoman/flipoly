from decimal import Decimal

from polyflip.execution.config import POLYMARKET_MIN_ORDER_SHARES
from polyflip.execution.worker import _resolve_requested_shares, _smart_maker_order_mode


def test_smart_maker_routes_exact_minimum_to_gtc():
    assert _smart_maker_order_mode(POLYMARKET_MIN_ORDER_SHARES) == "GTC_TTL"


def test_smart_maker_routes_small_order_to_fak_retry():
    assert _smart_maker_order_mode(Decimal("4.999999")) == "FAK_RETRY"


def test_smart_maker_uses_resolved_budget_shares_for_paper_route():
    shares = _resolve_requested_shares(
        requested_shares=None,
        max_spend_usdc=Decimal("1.25"),
        limit_price=Decimal("0.25"),
        side="BUY",
    )
    assert shares >= POLYMARKET_MIN_ORDER_SHARES
    assert _smart_maker_order_mode(shares) == "GTC_TTL"
