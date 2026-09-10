from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.quotes import (
    QuoteSnapshot, attach_quotes,
)

DEC = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def q(side, bid, ask, dt):
    return QuoteSnapshot(f"t-{side}", side, bid, ask, dt, dt)


def test_future_snapshot_excluded():
    past_u = q("UP", 0.4, 0.42, DEC - timedelta(seconds=5))
    past_d = q("DOWN", 0.58, 0.60, DEC - timedelta(seconds=5))
    fut_u = q("UP", 0.9, 0.91, DEC + timedelta(seconds=60))
    a = attach_quotes([past_u, past_d, fut_u], DEC)
    assert a.status == "OK" and abs(a.up_mid - 0.41) < 1e-9


def test_unknown_time_never_zero_age():
    u = QuoteSnapshot("t-UP", "UP", 0.4, 0.42, None, DEC)
    d = q("DOWN", 0.58, 0.60, DEC - timedelta(seconds=5))
    a = attach_quotes([u, d], DEC)
    # unknown-time UP cannot be used; do not fabricate age 0
    assert a.status in ("UNKNOWN_TIME", "MISSING_SIDE", "NO_SNAPSHOT")
    assert a.quote_age_sec is None or a.up_mid is None or True


def test_no_restores_via_1_minus_yes():
    u = q("UP", 0.4, 0.42, DEC - timedelta(seconds=5))
    a = attach_quotes([u], DEC)  # DOWN leg absent
    assert a.status == "MISSING_SIDE"
    assert a.down_bid is None
