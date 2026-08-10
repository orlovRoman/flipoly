from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from polyflip.models.sequence_features import (
    FEATURE_EXPERIMENT_VARIANTS,
    SEQUENCE_DIRECTION_FEATURES,
    normalize_experiment_variant,
    SEQUENCE_CANDLE_FEATURES,
    attach_closed_candle_features,
    build_closed_candle_feature_frame,
    sequence_history_ready,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(index: int, direction: int, *, closed: bool = True) -> dict:
    opening = 100.0
    closing = opening * (1.001 if direction > 0 else 0.999 if direction < 0 else 1.0)
    return {
        "open_time": BASE + timedelta(minutes=15 * index),
        "close_time": BASE + timedelta(minutes=15 * (index + 1)),
        "is_closed": closed,
        "open": opening,
        "high": max(opening, closing) + 0.05,
        "low": min(opening, closing) - 0.05,
        "close": closing,
    }


def test_sequence_preserves_order_not_only_aggregate_return():
    first = build_closed_candle_feature_frame([
        candle(0, 1), candle(1, -1), candle(2, 1),
        candle(3, -1), candle(4, -1), candle(5, 1),
    ]).iloc[-1]
    second = build_closed_candle_feature_frame([
        candle(0, -1), candle(1, 1), candle(2, -1),
        candle(3, 1), candle(4, -1), candle(5, 1),
    ]).iloc[-1]

    assert first["direction_lag_3"] == -1
    assert second["direction_lag_3"] == 1
    assert first["direction_lag_1"] == second["direction_lag_1"] == 1


def test_neutral_threshold_maps_tiny_body_to_zero():
    tiny = candle(0, 1)
    tiny["close"] = tiny["open"] * 1.00001
    result = build_closed_candle_feature_frame([tiny])
    assert result.iloc[-1]["direction_lag_1"] == 0


def test_run_lengths_and_structure_features_are_interpretable():
    result = build_closed_candle_feature_frame([
        candle(0, -1), candle(1, -1), candle(2, 1),
        candle(3, 1), candle(4, 1), candle(5, 1),
    ]).iloc[-1]

    assert result["consecutive_up"] == 4
    assert result["consecutive_down"] == 0
    assert result["up_ratio_4"] == 1.0
    assert 0.0 <= result["alternation_rate_6"] <= 1.0
    assert 0.0 < result["signed_trend_efficiency_6"] <= 1.0
    assert 0.0 <= result["body_to_range"] <= 1.0


def test_incomplete_candle_is_never_used():
    candles = [candle(i, 1) for i in range(6)]
    future_open = candle(6, -1, closed=False)
    decisions = pd.DataFrame({"recorded_at": [BASE + timedelta(minutes=106)]})

    result = attach_closed_candle_features(decisions, [*candles, future_open])

    assert result.iloc[0]["sequence_asof_close_time"] == pd.Timestamp(
        BASE + timedelta(minutes=90)
    )
    assert result.iloc[0]["direction_lag_1"] == 1


def test_asof_join_never_uses_candle_closed_after_decision():
    candles = [candle(i, 1 if i < 5 else -1) for i in range(6)]
    decision_at = BASE + timedelta(minutes=89)
    decisions = pd.DataFrame({"recorded_at": [decision_at]})

    result = attach_closed_candle_features(decisions, candles)

    assert result.iloc[0]["sequence_asof_close_time"] == pd.Timestamp(
        BASE + timedelta(minutes=75)
    )
    assert result.iloc[0]["direction_lag_1"] == 1
    assert result.iloc[0]["sequence_asof_close_time"] <= pd.Timestamp(decision_at)


def test_history_readiness_requires_six_closed_candles():
    decisions = pd.DataFrame({
        "recorded_at": [
            BASE + timedelta(minutes=75),
            BASE + timedelta(minutes=90),
        ]
    })
    result = attach_closed_candle_features(
        decisions, [candle(i, 1) for i in range(6)]
    )
    readiness = sequence_history_ready(result)

    assert readiness.tolist() == [False, True]
    assert set(SEQUENCE_CANDLE_FEATURES).issubset(result.columns)


def test_invalid_decision_timestamp_is_rejected():
    with pytest.raises(ValueError, match="Decision timestamps"):
        attach_closed_candle_features(
            pd.DataFrame({"recorded_at": [None]}), [candle(0, 1)]
        )


def test_feature_experiment_variants_are_incremental():
    assert FEATURE_EXPERIMENT_VARIANTS["A"] == ()
    assert FEATURE_EXPERIMENT_VARIANTS["B"] == SEQUENCE_DIRECTION_FEATURES
    assert set(FEATURE_EXPERIMENT_VARIANTS["B"]).issubset(
        FEATURE_EXPERIMENT_VARIANTS["C"]
    )
    assert len(FEATURE_EXPERIMENT_VARIANTS["C"]) == 10


def test_feature_experiment_variant_validation():
    assert normalize_experiment_variant(" b ") == "B"
    assert normalize_experiment_variant(None) == "AUTO"

    import pytest
    with pytest.raises(ValueError, match="AUTO, A, B or C"):
        normalize_experiment_variant("D")
