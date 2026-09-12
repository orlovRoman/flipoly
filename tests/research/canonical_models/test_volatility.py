from datetime import datetime, timedelta, timezone

from polyflip.research.canonical_models.volatility import Bar, past_volatility

DEC = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def bars(n, start, step_min=1, price=100.0):
    out = []
    t = start
    for _ in range(n):
        out.append(Bar(t, t + timedelta(minutes=step_min), price))
        t += timedelta(minutes=step_min)
        price *= 1.001
    return out


def test_future_bars_do_not_change_past():
    hist = bars(35, DEC - timedelta(minutes=40))
    v1 = past_volatility(hist, DEC)
    fut = hist + bars(10, DEC + timedelta(minutes=1))
    v2 = past_volatility(fut, DEC)
    assert abs((v1.sigma_per_sec or 0) - (v2.sigma_per_sec or 0)) < 1e-12
    assert v1.n_returns == v2.n_returns


def test_gaps_not_filled_short_history_flag():
    few = bars(5, DEC - timedelta(minutes=5))
    v = past_volatility(few, DEC)
    assert v.status in ("SHORT_HISTORY", "OK") and v.n_returns == 4
    assert past_volatility([], DEC).status == "NO_BARS"
