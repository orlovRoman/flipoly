from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from polyflip.models.temporal_validation import (
    grouped_walk_forward_folds,
    latest_group_holdout,
    market_balanced_weights,
)


def temporal_rows(markets: int = 12, rows_per_market: int = 3):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    groups = []
    timestamps = []
    for market in range(markets):
        for row in range(rows_per_market):
            groups.append(f"market-{market:02d}")
            timestamps.append(base + timedelta(minutes=15 * market, seconds=row))
    return pd.Series(groups), pd.Series(timestamps)


def test_walk_forward_uses_only_older_markets_for_training():
    groups, timestamps = temporal_rows()
    folds = grouped_walk_forward_folds(groups, timestamps, n_splits=4)

    assert len(folds) == 4
    for fold in folds:
        assert not set(fold.train_groups) & set(fold.validation_groups)
        assert timestamps.iloc[fold.train_index].max() < timestamps.iloc[
            fold.validation_index
        ].min()


def test_future_markets_never_train_past_validation_rows():
    groups, timestamps = temporal_rows(markets=9)
    folds = grouped_walk_forward_folds(groups, timestamps, n_splits=3)

    group_order = {f"market-{index:02d}": index for index in range(9)}
    for fold in folds:
        assert max(map(group_order.get, fold.train_groups)) < min(
            map(group_order.get, fold.validation_groups)
        )


def test_latest_holdout_reserves_newest_complete_markets():
    groups, timestamps = temporal_rows(markets=10, rows_per_market=2)
    train, validation = latest_group_holdout(
        groups, timestamps, validation_fraction=0.2
    )

    assert set(groups.iloc[validation]) == {"market-08", "market-09"}
    assert not set(groups.iloc[train]) & set(groups.iloc[validation])


def test_market_balancing_equalizes_total_market_influence():
    groups = pd.Series(["large"] * 10 + ["small"] * 2)
    weights = market_balanced_weights(groups)

    assert weights[groups.eq("large")].sum() == pytest.approx(
        weights[groups.eq("small")].sum()
    )
    assert weights.mean() == pytest.approx(1.0)


def test_market_balancing_preserves_relative_time_weights_inside_market():
    groups = pd.Series(["a", "a", "b", "b"])
    base = np.array([1.0, 2.0, 2.0, 4.0])
    weights = market_balanced_weights(groups, base)

    assert weights[1] / weights[0] == pytest.approx(2.0)
    assert weights[3] / weights[2] == pytest.approx(2.0)
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())


def test_invalid_timestamp_is_rejected():
    groups = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError, match="valid timestamps"):
        grouped_walk_forward_folds(groups, pd.Series([None, None, None]))
