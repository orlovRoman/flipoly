from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.decision_point import (
    Observation, select_decision,
)

END = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
TARGET = END - timedelta(minutes=5)  # T-5m


def obs(sec_after_target: float):
    ev = TARGET + timedelta(seconds=sec_after_target)
    return Observation(event_at=ev, received_at=ev)


def test_exact_target_excluded_first_inside_wins():
    # event exactly AT target is excluded (window is open on the left)
    d = select_decision([obs(0), obs(5)], END)
    assert d is not None and abs(d.lateness_sec - 5.0) < 1e-9


def test_lateness_15s_ok():
    d = select_decision([obs(15)], END)
    assert d is not None and abs(d.lateness_sec - 15.0) < 1e-9


def test_beyond_15s_is_miss():
    assert select_decision([obs(15.1)], END) is None
    assert select_decision([obs(-1)], END) is None
    assert select_decision([], END) is None


def test_first_available_wins_not_cheapest():
    # selection must not depend on price advantage: earliest in window wins
    a, b = obs(3), obs(10)
    d = select_decision([b, a], END)
    assert d is not None and abs(d.lateness_sec - 3.0) < 1e-9
