from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.forecast_table import brier, logloss
from polyflip.research.canonical_models.underlying import (
    StrikePoint, UnderlyingPoint, attach_underlying,
)

DEC = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def test_units_and_availability():
    up = UnderlyingPoint(100.0, "USD", DEC - timedelta(seconds=10))
    st = StrikePoint(99.0, "USD", "canonical_confirmed", DEC - timedelta(seconds=20))
    assert attach_underlying(up, st, DEC).status == "OK"
    bad = StrikePoint(99.0, "BTC", "canonical_confirmed", DEC - timedelta(seconds=20))
    assert attach_underlying(up, bad, DEC).status == "UNIT_MISMATCH"
    late = StrikePoint(99.0, "USD", "canonical_confirmed", DEC + timedelta(seconds=5))
    assert attach_underlying(up, late, DEC).status == "NOT_YET_AVAILABLE"
    proxy = StrikePoint(99.0, "USD", "binance_proxy", None)
    assert attach_underlying(up, proxy, DEC).status == "OK"  # allowed only as separate column


def test_brier_once_per_market():
    assert abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12
    assert logloss([0.9], [1]) < logloss([0.6], [1])
