from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.models import OrderbookLevel
from polyflip.research.lp_rewards.reconciler import BookReconciler


def test_reconciliation_within_tolerance():
    reconciler = BookReconciler(max_tick_discrepancy=Decimal("0.01"))
    local_bids = [OrderbookLevel(price=Decimal("0.49"), size=Decimal("50.0"))]
    local_asks = [OrderbookLevel(price=Decimal("0.51"), size=Decimal("50.0"))]

    # REST matches exactly
    ok = reconciler.reconcile_book("12345", local_bids, local_asks, Decimal("0.49"), Decimal("0.51"))
    assert ok is True
    assert reconciler.uncertain_markets["12345"] is False


def test_reconciliation_mismatch_triggers_uncertain():
    reconciler = BookReconciler(max_tick_discrepancy=Decimal("0.01"))
    local_bids = [OrderbookLevel(price=Decimal("0.49"), size=Decimal("50.0"))]
    local_asks = [OrderbookLevel(price=Decimal("0.51"), size=Decimal("50.0"))]

    # REST shows bid 0.45 (discrepancy 0.04 > 0.01)
    ok = reconciler.reconcile_book("12345", local_bids, local_asks, Decimal("0.45"), Decimal("0.51"))
    assert ok is False
    assert reconciler.uncertain_markets["12345"] is True


def test_quiet_market_not_marked_uncertain_without_mismatch():
    # Quiet market with static book
    reconciler = BookReconciler()
    local_bids = [OrderbookLevel(price=Decimal("0.50"), size=Decimal("100.0"))]
    local_asks = [OrderbookLevel(price=Decimal("0.52"), size=Decimal("100.0"))]

    # After time passes, REST still confirms 0.50 and 0.52
    ok = reconciler.reconcile_book("tok_quiet", local_bids, local_asks, Decimal("0.50"), Decimal("0.52"))
    assert ok is True
    assert reconciler.uncertain_markets.get("tok_quiet") is False
