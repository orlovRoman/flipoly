from datetime import datetime, timedelta, timezone

import pytest

from polyflip.research.canonical_models.splits import (
    expanding_folds, holdout, split_rows,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def rows(n=6):
    return [{"market_id": f"m{i}",
             "decision_at": T0 + timedelta(days=i)} for i in range(n)]


def test_no_market_crosses_split():
    folds = expanding_folds([T0, T0 + timedelta(days=2), T0 + timedelta(days=4),
                             T0 + timedelta(days=6)])
    assert len(folds) == 3
    tr, va = split_rows(rows(), folds[0])
    assert {r["market_id"] for r in tr}.isdisjoint({r["market_id"] for r in va})


def test_duplicate_market_id_rejected():
    folds = expanding_folds([T0, T0 + timedelta(days=2), T0 + timedelta(days=6)])
    dup = rows(3) + [dict(rows(3)[0], decision_at=T0 + timedelta(days=5))]
    with pytest.raises(AssertionError):
        split_rows(dup, folds[1])


def test_holdout_disjoint_and_ordered():
    h = holdout(rows(), T0 + timedelta(days=4), T0 + timedelta(days=6))
    assert [r["market_id"] for r in h] == ["m4", "m5"]
