"""Grouped, chronological validation helpers for market-level datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    train_index: np.ndarray
    validation_index: np.ndarray
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def _group_timeline(groups: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    if len(groups) != len(timestamps):
        raise ValueError("groups and timestamps must have equal length")
    frame = pd.DataFrame({
        "group": groups.astype(str).to_numpy(),
        "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
    })
    if frame["timestamp"].isna().any():
        raise ValueError("Temporal validation requires valid timestamps")
    return (
        frame.groupby("group", as_index=False)["timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values(["min", "max", "group"])
        .reset_index(drop=True)
    )


def _non_overlapping_time_cohorts(timeline: pd.DataFrame) -> list[pd.DataFrame]:
    """Keep simultaneous or overlapping markets in one indivisible cohort."""
    if timeline.empty:
        return []

    cohorts: list[pd.DataFrame] = []
    cohort_start_index = 0
    cohort_start = pd.Timestamp(timeline.iloc[0]["min"])
    cohort_end = pd.Timestamp(timeline.iloc[0]["max"])
    for index in range(1, len(timeline)):
        row_start = pd.Timestamp(timeline.iloc[index]["min"])
        row_end = pd.Timestamp(timeline.iloc[index]["max"])
        if row_start == cohort_start or row_start < cohort_end:
            cohort_end = max(cohort_end, row_end)
            continue
        cohorts.append(timeline.iloc[cohort_start_index:index].reset_index(drop=True))
        cohort_start_index = index
        cohort_start = row_start
        cohort_end = row_end
    cohorts.append(timeline.iloc[cohort_start_index:].reset_index(drop=True))
    return cohorts


def grouped_walk_forward_folds(
    groups: pd.Series,
    timestamps: pd.Series,
    *,
    n_splits: int = 5,
) -> list[TemporalFold]:
    """Build expanding-window folds with whole markets in each partition.

    The earliest block is training-only.  Every later block is validated once;
    consequently early observations intentionally have no OOF prediction.
    """
    timeline = _group_timeline(groups.reset_index(drop=True), timestamps)
    cohorts = _non_overlapping_time_cohorts(timeline)
    if len(cohorts) < 3:
        return []

    block_count = min(max(2, n_splits + 1), len(cohorts))
    cohort_blocks = [
        indexes for indexes in np.array_split(np.arange(len(cohorts)), block_count)
        if len(indexes)
    ]
    blocks = [
        pd.concat([cohorts[int(index)] for index in indexes], ignore_index=True)
        for indexes in cohort_blocks
    ]
    group_values = groups.astype(str).reset_index(drop=True)
    folds: list[TemporalFold] = []

    for block_index in range(1, len(blocks)):
        train_table = pd.concat(blocks[:block_index], ignore_index=True)
        validation_table = blocks[block_index]
        train_groups = tuple(train_table["group"].astype(str))
        validation_groups = tuple(validation_table["group"].astype(str))
        train_index = np.flatnonzero(group_values.isin(train_groups).to_numpy())
        validation_index = np.flatnonzero(
            group_values.isin(validation_groups).to_numpy()
        )
        if set(train_groups) & set(validation_groups):
            raise AssertionError("Market leakage between temporal train and validation")
        train_end = pd.Timestamp(train_table["max"].max())
        validation_start = pd.Timestamp(validation_table["min"].min())
        if train_end > validation_start:
            raise AssertionError(
                "Temporal leakage: training markets overlap validation markets"
            )
        folds.append(TemporalFold(
            train_index=train_index,
            validation_index=validation_index,
            train_groups=train_groups,
            validation_groups=validation_groups,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=pd.Timestamp(validation_table["max"].max()),
        ))
    return folds


def latest_group_holdout(
    groups: pd.Series,
    timestamps: pd.Series,
    *,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole markets chronologically, reserving the newest groups."""
    timeline = _group_timeline(groups.reset_index(drop=True), timestamps)
    cohorts = _non_overlapping_time_cohorts(timeline)
    if len(cohorts) < 2:
        raise ValueError(
            "At least two non-overlapping market cohorts are required for a temporal holdout"
        )
    validation_target = max(1, int(np.ceil(len(timeline) * validation_fraction)))
    validation_cohorts: list[pd.DataFrame] = []
    validation_size = 0
    for cohort in reversed(cohorts[1:]):
        validation_cohorts.append(cohort)
        validation_size += len(cohort)
        if validation_size >= validation_target:
            break
    validation_table = pd.concat(validation_cohorts, ignore_index=True)
    validation_groups = set(validation_table["group"].astype(str))
    train_groups = set(
        timeline.loc[
            ~timeline["group"].isin(validation_groups), "group"
        ].astype(str)
    )
    group_values = groups.astype(str).reset_index(drop=True)
    return (
        np.flatnonzero(group_values.isin(train_groups).to_numpy()),
        np.flatnonzero(group_values.isin(validation_groups).to_numpy()),
    )


def market_balanced_weights(
    groups: pd.Series,
    base_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Give every market equal total influence while preserving row weights."""
    group_values = groups.astype(str).reset_index(drop=True)
    weights = (
        np.ones(len(group_values), dtype=float)
        if base_weight is None
        else np.asarray(base_weight, dtype=float).copy()
    )
    if len(weights) != len(group_values):
        raise ValueError("base_weight and groups must have equal length")
    for group in group_values.unique():
        mask = group_values.eq(group).to_numpy()
        total = float(weights[mask].sum())
        if total > 0:
            weights[mask] /= total
    mean = float(weights.mean())
    return weights / mean if mean > 0 else np.ones(len(weights), dtype=float)
