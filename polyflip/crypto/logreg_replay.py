"""Pure helpers for deterministic LogReg OOF replay and calibration metrics."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from polyflip.crypto.logreg_polymarket_backtest import flip_probability_to_yes_probability

CLOSE_TIME_PRIORITY = ("market_close_at", "resolved_at", "end_time_est", "market_start")


def split_market_windows(
    frame: pd.DataFrame,
    close_times: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Partition unique markets chronologically by canonical close time."""
    rows: list[tuple[str, pd.Timestamp]] = []
    for market_id in frame.get("market_id", pd.Series(dtype=str)).dropna().astype(str).unique():
        value = close_times.get(market_id)
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(timestamp):
            rows.append((market_id, timestamp))
    rows.sort(key=lambda item: (item[1], item[0]))
    markets = [market_id for market_id, _ in rows]
    if not markets:
        return {"T1": set(), "T2": set(), "T3": set()}
    if len(markets) < 3:
        return {"T1": set(markets), "T2": set(), "T3": set()}
    first = len(markets) // 3
    second = 2 * len(markets) // 3
    return {
        "T1": set(markets[:first]),
        "T2": set(markets[first:second]),
        "T3": set(markets[second:]),
    }


def first_snapshot_per_market(
    frame: pd.DataFrame,
    p_flip: Iterable[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select one valid entry snapshot per market and convert p_flip to p_yes."""
    scores = np.asarray(list(p_flip), dtype=float)
    if len(scores) != len(frame):
        raise ValueError("p_flip must align with frame")
    if "market_id" not in frame.columns or "mid_price" not in frame.columns:
        raise ValueError("frame requires market_id and mid_price")
    working = frame.reset_index(drop=True).copy()
    working["_p_flip"] = scores
    if "recorded_at" in working.columns:
        working["recorded_at"] = pd.to_datetime(working["recorded_at"], utc=True, errors="coerce")
        working = working.sort_values(["market_id", "recorded_at"], kind="stable")
    else:
        working = working.sort_values(["market_id"], kind="stable")
    selected_indices: list[int] = []
    p_yes: list[float] = []
    seen: set[str] = set()
    for index, row in working.iterrows():
        market_id = str(row["market_id"])
        if market_id in seen or not np.isfinite(row["_p_flip"]):
            continue
        try:
            converted = flip_probability_to_yes_probability(row["_p_flip"], row["mid_price"])
        except (TypeError, ValueError):
            continue
        seen.add(market_id)
        selected_indices.append(index)
        p_yes.append(converted)
    selected = working.loc[selected_indices].drop(columns=["_p_flip"], errors="ignore")
    return selected.reset_index(drop=True), np.asarray(p_yes, dtype=float)


def classification_metrics(
    selected: pd.DataFrame,
    p_yes: Iterable[float],
    *,
    bins: int = 10,
) -> dict[str, float | int | None]:
    """Calculate Brier, log loss and expected calibration error without fitting."""
    scores = np.asarray(list(p_yes), dtype=float)
    if "final_outcome" not in selected.columns or len(scores) != len(selected):
        return {"brier": None, "ece": None, "log_loss": None, "n_scored": 0}
    labels = selected["final_outcome"].astype(str).str.upper().map({"YES": 1, "NO": 0})
    valid = labels.notna().to_numpy() & np.isfinite(scores)
    if not valid.any():
        return {"brier": None, "ece": None, "log_loss": None, "n_scored": 0}
    y = labels.to_numpy(dtype=float)[valid]
    p = np.clip(scores[valid], 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {"brier": brier, "ece": float(ece), "log_loss": log_loss, "n_scored": int(valid.sum())}
