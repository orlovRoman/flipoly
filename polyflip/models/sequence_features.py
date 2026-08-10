"""Leakage-safe sequence features built from fully closed underlying candles.

The feature timestamp is the candle ``close_time``.  Decision rows receive the
latest feature row whose timestamp is not later than ``recorded_at``.  This
contract is shared by training, backtesting and live inference.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


SEQUENCE_DIRECTION_FEATURES: tuple[str, ...] = (
    "direction_lag_1",
    "direction_lag_2",
    "direction_lag_3",
    "consecutive_up",
    "consecutive_down",
    "up_ratio_4",
    "alternation_rate_6",
)

SEQUENCE_CANDLE_FEATURES: tuple[str, ...] = (
    *SEQUENCE_DIRECTION_FEATURES,
    "signed_trend_efficiency_6",
    "signed_body_pct",
    "body_to_range",
)

SEQUENCE_AUDIT_COLUMNS: tuple[str, ...] = (
    "sequence_asof_close_time",
    "sequence_history_count",
)

FEATURE_EXPERIMENT_VARIANTS: dict[str, tuple[str, ...]] = {
    "A": (),
    "B": SEQUENCE_DIRECTION_FEATURES,
    "C": SEQUENCE_CANDLE_FEATURES,
}

FEATURE_EXPERIMENT_LABELS: dict[str, str] = {
    "A": "A-control-v1",
    "B": "B-direction-sequence-v1",
    "C": "C-candle-structure-v1",
}


def normalize_experiment_variant(value: str | None) -> str:
    variant = (value or "AUTO").strip().upper()
    if variant not in {"AUTO", *FEATURE_EXPERIMENT_VARIANTS}:
        raise ValueError("feature_set must be AUTO, A, B or C")
    return variant

SEQUENCE_FEATURE_SET_VERSION = "underlying-sequence-v1"
MIN_SEQUENCE_HISTORY = 6


def _read(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _run_length(direction: pd.Series, expected: int, cap: int = 8) -> pd.Series:
    result: list[float] = []
    run = 0
    for value in direction.fillna(0).astype(int):
        run = min(run + 1, cap) if value == expected else 0
        result.append(float(run))
    return pd.Series(result, index=direction.index, dtype=float)


def _alternation_rate(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    transitions = []
    for previous, current in zip(values[:-1], values[1:]):
        if previous == 0 or current == 0:
            transitions.append(0.0)
        else:
            transitions.append(float(previous != current))
    return float(np.mean(transitions)) if transitions else np.nan


def _signed_efficiency(values: np.ndarray) -> float:
    denominator = float(np.abs(values).sum())
    if denominator <= 1e-12:
        return 0.0
    return float(values.sum() / denominator)


def build_closed_candle_feature_frame(
    candles: Iterable[Any],
    *,
    neutral_return_threshold: float = 0.0002,
) -> pd.DataFrame:
    """Return one feature row per unique, fully closed candle.

    ``neutral_return_threshold`` is expressed as a fractional return (0.0002 is
    two basis points).  Incomplete candles are deliberately ignored.
    """
    rows = []
    for candle in candles:
        close_time = _read(candle, "close_time")
        is_closed = _read(candle, "is_closed")
        if is_closed is not True or close_time is None:
            continue
        rows.append(
            {
                "sequence_asof_close_time": close_time,
                "open": _read(candle, "open"),
                "high": _read(candle, "high"),
                "low": _read(candle, "low"),
                "close": _read(candle, "close"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "sequence_asof_close_time",
            *SEQUENCE_CANDLE_FEATURES,
            "sequence_history_count",
        ])

    frame = pd.DataFrame(rows)
    frame["sequence_asof_close_time"] = pd.to_datetime(
        frame["sequence_asof_close_time"], utc=True, errors="coerce"
    )
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=[
            "sequence_asof_close_time", "open", "high", "low", "close"
        ])
        .drop_duplicates("sequence_asof_close_time", keep="last")
        .sort_values("sequence_asof_close_time")
        .reset_index(drop=True)
    )

    candle_return = frame["close"].div(frame["open"].replace(0.0, np.nan)) - 1.0
    direction = pd.Series(
        np.where(
            candle_return.abs() < neutral_return_threshold,
            0.0,
            np.sign(candle_return),
        ),
        index=frame.index,
        dtype=float,
    ).fillna(0.0)

    frame["direction_lag_1"] = direction
    frame["direction_lag_2"] = direction.shift(1)
    frame["direction_lag_3"] = direction.shift(2)
    frame["consecutive_up"] = _run_length(direction, 1)
    frame["consecutive_down"] = _run_length(direction, -1)
    frame["up_ratio_4"] = (
        direction.eq(1).astype(float).rolling(4, min_periods=4).mean()
    )
    frame["alternation_rate_6"] = direction.rolling(6, min_periods=6).apply(
        _alternation_rate, raw=True
    )
    frame["signed_trend_efficiency_6"] = candle_return.rolling(
        6, min_periods=6
    ).apply(_signed_efficiency, raw=True)
    frame["signed_body_pct"] = candle_return
    candle_range = (frame["high"] - frame["low"]).abs()
    frame["body_to_range"] = (
        (frame["close"] - frame["open"]).abs()
        .div(candle_range.replace(0.0, np.nan))
        .clip(0.0, 1.0)
        .fillna(0.0)
    )
    frame["sequence_history_count"] = np.arange(1, len(frame) + 1)

    return frame[[
        "sequence_asof_close_time",
        *SEQUENCE_CANDLE_FEATURES,
        "sequence_history_count",
    ]]


def attach_closed_candle_features(
    decisions: pd.DataFrame,
    candles: Iterable[Any],
    *,
    decision_time_col: str = "recorded_at",
    neutral_return_threshold: float = 0.0002,
) -> pd.DataFrame:
    """Attach the most recent closed-candle features to decision rows.

    Rows without six prior closed candles retain NaNs in sequence features so
    callers can explicitly exclude them rather than silently treating missing
    history as a real zero-valued pattern.
    """
    if decision_time_col not in decisions.columns:
        raise ValueError(f"Missing decision timestamp column: {decision_time_col}")

    left = decisions.copy()
    left["_sequence_row_order"] = np.arange(len(left))
    left["_sequence_decision_time"] = pd.to_datetime(
        left[decision_time_col], utc=True, errors="coerce"
    )
    if left["_sequence_decision_time"].isna().any():
        raise ValueError("Decision timestamps must be valid timezone-aware datetimes")

    right = build_closed_candle_feature_frame(
        candles, neutral_return_threshold=neutral_return_threshold
    )
    if right.empty:
        for column in (*SEQUENCE_CANDLE_FEATURES, *SEQUENCE_AUDIT_COLUMNS):
            left[column] = pd.NaT if column == "sequence_asof_close_time" else np.nan
        return left.drop(columns=["_sequence_row_order", "_sequence_decision_time"])

    merged = pd.merge_asof(
        left.sort_values("_sequence_decision_time"),
        right.sort_values("sequence_asof_close_time"),
        left_on="_sequence_decision_time",
        right_on="sequence_asof_close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    future_mask = (
        merged["sequence_asof_close_time"].notna()
        & (merged["sequence_asof_close_time"] > merged["_sequence_decision_time"])
    )
    if future_mask.any():
        raise AssertionError("Closed-candle as-of join selected future data")

    return (
        merged.sort_values("_sequence_row_order")
        .drop(columns=["_sequence_row_order", "_sequence_decision_time"])
        .reset_index(drop=True)
    )


def sequence_history_ready(frame: pd.DataFrame) -> pd.Series:
    """Mask rows with enough closed history and all sequence values present."""
    if "sequence_history_count" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return (
        frame["sequence_history_count"].fillna(0).ge(MIN_SEQUENCE_HISTORY)
        & frame[list(SEQUENCE_CANDLE_FEATURES)].notna().all(axis=1)
    )
