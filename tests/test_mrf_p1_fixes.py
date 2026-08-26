import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from polyflip.crypto.market_regime import (
    HORIZON_4H,
    HORIZON_12H,
    HORIZON_24H,
    MIN_HISTORY_CANDLES,
    AssetRegimeFeatures,
    BasketRegimeFeatures,
    _efficiency_ratio,
    _log_returns,
    _volatility,
    build_regime_snapshot,
    compute_asset_features,
    validate_candle_continuity,
    MarketRegimeSnapshot,
)
from polyflip.crypto.market_regime_classifier import (
    MarketPhase,
    Regime,
    classify_asset_regime,
)
from polyflip.crypto.market_regime_integration import build_snapshot_from_candles
from polyflip.crypto.market_regime_policy import (
    FilterMode,
    StrategyType,
    evaluate_policy,
)
from polyflip.crypto.market_regime_audit import serialize_regime_audit


def _linear_data(count: int = MIN_HISTORY_CANDLES, start: float = 100.0, end: float = 110.0):
    closes = np.linspace(start, end, count).astype(np.float64)
    return {
        "closes": closes,
        "highs": closes * 1.01,
        "lows": closes * 0.99,
        "opens": closes - 0.1,
        "count": count,
    }


def _snapshot_with_basket(**basket_overrides) -> MarketRegimeSnapshot:
    defaults = dict(
        median_ret_4h=0.03,
        median_ret_12h=0.03,
        median_ret_24h=0.03,
        breadth_up_4h=1.0,
        breadth_up_12h=1.0,
        breadth_up_24h=1.0,
        dispersion_4h=0.0,
        dispersion_24h=0.0,
        market_efficiency_24h=0.5,
        ready_count=1,
        total_count=1,
        history_ready=True,
        strength=0.4,
    )
    defaults.update(basket_overrides)
    asset = AssetRegimeFeatures(
        symbol="BTC",
        ret_4h=0.03,
        ret_12h=0.03,
        ret_24h=0.03,
        efficiency_24h=0.5,
        history_ready=True,
        candle_count=MIN_HISTORY_CANDLES,
        strength_score=0.4,
    )
    return MarketRegimeSnapshot(
        as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
        assets={"BTC": asset},
        basket=BasketRegimeFeatures(**defaults),
    )


def test_log_returns_use_required_intervals():
    closes = np.linspace(100, 110, MIN_HISTORY_CANDLES)
    assert _log_returns(closes, HORIZON_24H) == pytest.approx(math.log(1.1))
    assert _log_returns(closes[:96], HORIZON_24H) == 0.0

    closes_4h = np.linspace(100, 105, HORIZON_4H + 1)
    assert _log_returns(closes_4h, HORIZON_4H) == pytest.approx(math.log(1.05))
    closes_12h = np.linspace(100, 105, HORIZON_12H + 1)
    assert _log_returns(closes_12h, HORIZON_12H) == pytest.approx(math.log(1.05))


def test_compute_asset_features_requires_array_and_count_history():
    data = _linear_data()
    short = np.linspace(100, 105, 50)
    assert not compute_asset_features(
        "BTC", short, short, short, short, candle_count=MIN_HISTORY_CANDLES
    ).history_ready
    assert not compute_asset_features(
        "BTC", data["closes"], data["highs"], data["lows"], data["opens"], candle_count=50
    ).history_ready
    assert compute_asset_features(
        "BTC", data["closes"], data["highs"], data["lows"], data["opens"],
        candle_count=MIN_HISTORY_CANDLES,
    ).history_ready


def test_validate_candle_continuity_catches_bad_windows():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good = [base + timedelta(minutes=15 * i) for i in range(MIN_HISTORY_CANDLES)]
    assert validate_candle_continuity(good, MIN_HISTORY_CANDLES) == (True, "ok")

    duplicate = list(good)
    duplicate[5] = duplicate[4]
    ok, reason = validate_candle_continuity(duplicate, MIN_HISTORY_CANDLES)
    assert not ok and "duplicates" in reason

    short = good[:5]
    ok, reason = validate_candle_continuity(short, MIN_HISTORY_CANDLES)
    assert not ok and "count_mismatch" in reason

    unsorted = [good[0], good[2], good[1], good[3]]
    ok, reason = validate_candle_continuity(unsorted, 4)
    assert not ok and "not_sorted" in reason


def test_numeric_guards_do_not_crash():
    assert _efficiency_ratio(np.ones(MIN_HISTORY_CANDLES)) == 0.0
    assert _volatility(np.zeros(100)) == 0.0


def test_snapshot_trims_future_candles():
    # A few fetched candles are in the future relative to the decision
    # timestamp; the assertion below proves the cutoff is applied rather than
    # merely checking a window that is already historical.
    as_of = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    data = _linear_data(100)
    data["open_times"] = [
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
        for i in range(100)
    ]
    snapshot = build_regime_snapshot(
        {"BTC": data}, as_of=as_of, max_open_time=as_of,
    )
    assert snapshot.assets["BTC"].history_ready
    assert snapshot.assets["BTC"].candle_count < 100


def test_integration_snapshot_trims_future_candles():
    class Candle:
        def __init__(self, open_time, value):
            self.open_time = open_time
            self.open = value
            self.high = value + 1
            self.low = value - 1
            self.close = value + 0.5

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [Candle(base + timedelta(minutes=15 * i), 100 + i) for i in range(100)]
    snapshot = build_snapshot_from_candles(
        candles, "BTC", base + timedelta(minutes=15 * 50)
    )
    assert snapshot.assets["BTC"].candle_count <= 51


def test_classifier_and_legacy_policy_contract():
    snapshot = build_regime_snapshot({"BTC": _linear_data()})
    cls = classify_asset_regime(snapshot.assets["BTC"])
    assert cls.phase in list(Regime)
    assert 0.0 <= cls.confidence <= 1.0

    mixed = _snapshot_with_basket(dispersion_24h=0.03)
    result = evaluate_policy(
        mixed,
        StrategyType.ML_TREND_FOLLOW,
        direction=1.0,
        mode=FilterMode.ACTIVE,
    )
    assert result.allow is True
    assert result.stake_multiplier == pytest.approx(0.5)

    weak_up = _snapshot_with_basket(
        median_ret_4h=0.04,
        median_ret_12h=0.04,
        median_ret_24h=0.04,
        market_efficiency_24h=0.5,
    )
    result = evaluate_policy(
        weak_up,
        StrategyType.ML_TREND_FOLLOW,
        direction=-1.0,
        mode=FilterMode.ACTIVE,
    )
    assert result.allow is False
    assert result.stake_multiplier == pytest.approx(0.0)


def test_audit_serialization_keeps_legacy_fields():
    snapshot = _snapshot_with_basket()
    policy = evaluate_policy(
        snapshot,
        StrategyType.OUTSIDER,
        direction=1.0,
        mode=FilterMode.SHADOW,
    )
    audit = serialize_regime_audit(snapshot, policy, FilterMode.SHADOW, mrf_version=1)
    assert all(
        key in audit
        for key in (
            "mode",
            "version",
            "as_of",
            "global_regime",
            "global_confidence",
        )
    )
