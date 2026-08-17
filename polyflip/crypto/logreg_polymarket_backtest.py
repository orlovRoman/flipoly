"""Canonical Polymarket OOT evaluator for LogReg flip probabilities.

LogReg is trained on flip_vs_final_outcome: the probability is the chance
that the currently favoured outcome flips. Polymarket accounting needs a YES
probability, so this module converts the semantics explicitly, keeps one
entry snapshot per market, and delegates quote/PnL accounting to the shared
Polymarket evaluator.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from polyflip.crypto.polymarket_backtest import (
    METRICS_SCHEMA_VERSION,
    CanonicalBacktestMetrics,
    MetricsSchemaMismatchError,
    adapt_canonical_backtest_metrics,
    compute_oof_polymarket_backtest,
)


LOGREG_PREDICTION_SEMANTICS = "FLIP_VS_FINAL_OUTCOME"

CLOSE_TIME_PRIORITY_COLUMNS = (
    "market_close_at",
    "resolved_at",
    "end_time_est",
    "market_start",
)


def resolve_market_close_time(
    data: dict[str, Any] | pd.Series | Any,
) -> pd.Timestamp | None:
    """Resolve market close time for a single row/mapping/object using canonical fallback priority.

    Fallback chain priority:
      market_close_at -> resolved_at -> end_time_est -> market_start
    """
    for col in CLOSE_TIME_PRIORITY_COLUMNS:
        val = None
        if isinstance(data, dict):
            val = data.get(col)
        elif hasattr(data, col):
            val = getattr(data, col)
        elif hasattr(data, "__getitem__"):
            try:
                val = data[col]
            except (KeyError, IndexError):
                val = None
        if val is not None and not pd.isna(val):
            try:
                ts = pd.to_datetime(val, utc=True)
                if pd.notna(ts):
                    return ts
            except (ValueError, TypeError):
                continue
    return None


def resolve_market_close_time_series(
    df: pd.DataFrame,
) -> pd.Series:
    """Resolve market close time series for a DataFrame using canonical fallback priority."""
    if df.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    resolved = pd.Series(index=df.index, dtype="datetime64[ns, UTC]")
    for col in CLOSE_TIME_PRIORITY_COLUMNS:
        if col in df.columns:
            ts_series = pd.to_datetime(df[col], utc=True, errors="coerce")
            mask = resolved.isna() & ts_series.notna()
            resolved.loc[mask] = ts_series.loc[mask]
    return resolved


def flip_probability_to_yes_probability(p_flip: Any, mid_price: Any) -> float:
    """Convert a flip probability into P(YES wins).

    A YES midpoint above 0.5 means YES is the current favourite; otherwise
    NO is the favourite. p_flip is complemented only in the first case.
    """
    p_flip = float(p_flip)
    mid_price = float(mid_price)
    if not np.isfinite(p_flip) or not 0.0 <= p_flip <= 1.0:
        raise ValueError("p_flip must be finite and in [0, 1]")
    if not np.isfinite(mid_price) or not 0.0 <= mid_price <= 1.0:
        raise ValueError("mid_price must be finite and in [0, 1]")
    if mid_price == 0.5:
        raise ValueError("mid_price=0.5 has no canonical favourite")
    return float(1.0 - p_flip if mid_price > 0.5 else p_flip)


def _first_valid_per_market(
    frame: pd.DataFrame,
    scores: Iterable[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    if "market_id" not in frame.columns or "final_outcome" not in frame.columns:
        raise ValueError("LogReg OOT frame requires market_id and final_outcome")
    values = np.asarray(list(scores), dtype=float)
    if len(values) != len(frame):
        raise ValueError("oof_scores must align with frame")
    base = frame.reset_index(drop=True).copy()
    base["_p_flip"] = values
    if "recorded_at" in base.columns:
        base["recorded_at"] = pd.to_datetime(
            base["recorded_at"], utc=True, errors="coerce"
        )
        base = base.sort_values(["market_id", "recorded_at"], kind="stable")
    else:
        base = base.sort_values(["market_id"], kind="stable")
    base = base[
        base["final_outcome"].astype(str).str.upper().isin({"YES", "NO"})
    ]
    base = base[np.isfinite(base["_p_flip"].to_numpy())]
    if base.empty:
        return base.reset_index(drop=True), np.asarray([], dtype=float)
    valid_p_yes: dict[int, float] = {}
    for index, row in base.iterrows():
        try:
            valid_p_yes[index] = flip_probability_to_yes_probability(
                row["_p_flip"], row["mid_price"]
            )
        except (TypeError, ValueError):
            continue
    if not valid_p_yes:
        return base.iloc[0:0].reset_index(drop=True), np.asarray([], dtype=float)
    # Validate each row before de-duplicating.  An invalid earliest snapshot
    # (for example midpoint=0.5 with no canonical favourite) must not hide a
    # later valid entry snapshot for the same market.
    selected = base.loc[list(valid_p_yes)].drop_duplicates(
        "market_id", keep="first"
    )
    p_yes = np.asarray(
        [valid_p_yes[int(index)] for index in selected.index], dtype=float
    )
    return selected.reset_index(drop=True), p_yes


def compute_logreg_polymarket_backtest(
    frame: pd.DataFrame,
    oof_scores: Iterable[float],
    quotes: pd.DataFrame | None,
    *,
    strategy_branch: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate one saved LogReg OOF artifact on canonical Polymarket prices."""
    selected, p_yes = _first_valid_per_market(frame, oof_scores)
    if selected.empty:
        return compute_oof_polymarket_backtest(
            selected, p_yes, pd.DataFrame(), strategy_branch=strategy_branch, **kwargs
        )
    # Missing executable quotes are missing coverage, not a reason to reuse
    # the OOF/training frame as if it were a market snapshot.
    quote_frame = quotes.copy() if quotes is not None else pd.DataFrame()
    if quote_frame.empty:
        quote_frame = pd.DataFrame(columns=["market_id"])
    else:
        quote_frame["market_id"] = quote_frame["market_id"].astype(str)
    if "final_outcome" not in quote_frame.columns:
        quote_frame = quote_frame.merge(
            selected[["market_id", "final_outcome"]],
            on="market_id",
            how="left",
            suffixes=("", "_frame"),
        )
    canonical = selected.copy()
    canonical["target"] = (
        canonical["final_outcome"].astype(str).str.upper().eq("YES").astype(int)
    )
    return compute_oof_polymarket_backtest(
        canonical, p_yes, quote_frame, strategy_branch=strategy_branch, **kwargs
    )


def compute_logreg_polymarket_variants(
    frame: pd.DataFrame,
    oof_scores: Iterable[float],
    quotes: pd.DataFrame | None,
    *,
    branches: tuple[str, ...] = (
        "OUTSIDER_ONLY",
        "FAVORITE_ONLY",
        "COMBINED",
    ),
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    return {
        branch: compute_logreg_polymarket_backtest(
            frame, oof_scores, quotes, strategy_branch=branch, **kwargs
        )
        for branch in branches
    }


def _empty_single_window() -> dict[str, Any]:
    return {
        "status": "EMPTY",
        "start_close_time": None,
        "end_close_time": None,
        "unique_market_count": 0,
        "snapshot_count": 0,
        "trade_count": 0,
        "coverage": 0.0,
        "net_profit": None,
        "roi_pct": None,
        "max_drawdown": None,
        "n_markets": 0,
        "n_trades": 0,
        "total_pnl": None,
        "roi": None,
        "max_drawdown_usdc": None,
        "win_rate": None,
        "time_start": None,
        "time_end": None,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    }


def _empty_oot_windows() -> dict[str, Any]:
    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "T1": _empty_single_window(),
        "T2": _empty_single_window(),
        "T3": _empty_single_window(),
        "median_pnl": None,
        "non_negative_windows_count": 0,
    }


def split_chronological_oot_windows(
    frame: pd.DataFrame,
    oof_scores: Iterable[float] | np.ndarray,
    quotes: pd.DataFrame | None,
    *,
    strategy_branch: str = "COMBINED",
    **backtest_kwargs: Any,
) -> dict[str, Any]:
    """Split canonical evaluated markets into 3 chronological windows T1/T2/T3 strictly by market close time.

    Workflow:
      1. Deduplicate by market_id to extract unique markets.
      2. Assign each unique market a close time strictly via fallback priority:
         market_close_at -> resolved_at -> end_time_est -> market_start.
         Markets missing all 4 valid close-time fields are excluded from OOT partitions.
      3. Sort valid unique markets chronologically by (close_time, market_id).
      4. Partition the sequential series of valid unique markets into T1, T2, T3.
      5. Match all snapshots in frame to the window of their valid market.
      6. Compute canonical backtest metrics and store required window fields.
    """
    scores_arr = np.asarray(list(oof_scores), dtype=float) if oof_scores is not None else np.array([])
    if frame.empty or len(scores_arr) != len(frame):
        return _empty_oot_windows()

    working = frame.reset_index(drop=True).copy()
    working["_score"] = scores_arr

    # Resolve close times row-by-row / column-wise strictly from the 4 allowed sources.
    # Note: snapshot timestamps (recorded_at) are NEVER used as a fallback.
    if "_close_time" in working.columns and working["_close_time"].notna().any():
        close_times = pd.to_datetime(working["_close_time"], utc=True, errors="coerce")
    elif "close_time" in working.columns and working["close_time"].notna().any():
        close_times = pd.to_datetime(working["close_time"], utc=True, errors="coerce")
    else:
        close_times = resolve_market_close_time_series(working)

    working["_close_time"] = close_times

    # Step 1 & 2: Deduplicate by market_id and assign each market a valid close time.
    # Markets without a valid close_time from the allowed 4-source chain are strictly excluded.
    valid_working = working.dropna(subset=["_close_time"])
    if valid_working.empty:
        return _empty_oot_windows()

    market_meta = (
        valid_working
        .groupby("market_id", as_index=False)["_close_time"]
        .first()
    )
    market_meta = market_meta.dropna(subset=["_close_time"])
    if market_meta.empty:
        return _empty_oot_windows()

    # Step 3: Sort unique valid markets by close time (and market_id for deterministic ordering)
    market_meta = market_meta.sort_values(
        by=["_close_time", "market_id"], kind="stable"
    ).reset_index(drop=True)

    unique_markets = market_meta["market_id"].tolist()
    n_unique = len(unique_markets)

    # Step 4: Partition sequential series of markets into T1, T2, T3
    if n_unique < 3:
        t1_markets = unique_markets
        t2_markets = []
        t3_markets = []
    else:
        idx1 = n_unique // 3
        idx2 = 2 * n_unique // 3
        t1_markets = unique_markets[:idx1]
        t2_markets = unique_markets[idx1:idx2]
        t3_markets = unique_markets[idx2:]

    market_partitions = [
        ("T1", t1_markets),
        ("T2", t2_markets),
        ("T3", t3_markets),
    ]

    windows: dict[str, Any] = {"metrics_schema_version": METRICS_SCHEMA_VERSION}
    pnls: list[float] = []

    for label, m_list in market_partitions:
        if not m_list:
            windows[label] = _empty_single_window()
            continue

        m_set = set(m_list)
        # Step 5: Map snapshots to the window of their market
        sub_mask = working["market_id"].isin(m_set)
        sub_frame = working[sub_mask].reset_index(drop=True)
        sub_scores = sub_frame["_score"].to_numpy()
        sub_quotes = (
            quotes[quotes["market_id"].isin(m_set)].reset_index(drop=True)
            if quotes is not None and not quotes.empty and "market_id" in quotes.columns
            else None
        )

        sub_n_markets = len(m_list)
        sub_n_snapshots = len(sub_frame)

        # Window start and end close times
        sub_market_meta = market_meta[market_meta["market_id"].isin(m_set)]
        min_ct = sub_market_meta["_close_time"].min()
        max_ct = sub_market_meta["_close_time"].max()
        start_close_time = min_ct.isoformat() if pd.notna(min_ct) else None
        end_close_time = max_ct.isoformat() if pd.notna(max_ct) else None

        res = compute_logreg_polymarket_backtest(
            sub_frame,
            sub_scores,
            sub_quotes,
            strategy_branch=strategy_branch,
            **backtest_kwargs,
        )
        metrics = adapt_canonical_backtest_metrics(res)
        pnls.append(metrics.net_profit)
        coverage_val = float(res.get("coverage_pct", 0.0))

        windows[label] = {
            "status": "SPARSE" if sub_n_markets < 30 else "OK",
            "start_close_time": start_close_time,
            "end_close_time": end_close_time,
            "unique_market_count": sub_n_markets,
            "snapshot_count": sub_n_snapshots,
            "trade_count": metrics.n_trades,
            "coverage": coverage_val,
            "net_profit": round(metrics.net_profit, 4),
            "roi_pct": round(metrics.roi_pct, 4),
            "max_drawdown": round(metrics.max_drawdown, 4),
            # Backward-compatible aliases
            "n_markets": sub_n_markets,
            "n_trades": metrics.n_trades,
            "total_pnl": round(metrics.net_profit, 4),
            "roi": round(metrics.roi_pct, 4),
            "max_drawdown_usdc": round(metrics.max_drawdown, 4),
            "win_rate": round(metrics.win_rate, 4),
            "time_start": start_close_time,
            "time_end": end_close_time,
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
        }

    windows["median_pnl"] = round(float(np.median(pnls)), 4) if pnls else None
    windows["non_negative_windows_count"] = int(sum(p >= 0 for p in pnls))
    return windows


_split_chronological_windows = split_chronological_oot_windows
