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

from polyflip.crypto.polymarket_backtest import compute_oof_polymarket_backtest


LOGREG_PREDICTION_SEMANTICS = "FLIP_VS_FINAL_OUTCOME"


def flip_probability_to_yes_probability(p_flip: Any, mid_price: Any) -> float:
    """Convert a flip probability into P(YES wins).

    A YES midpoint at/above 0.5 means YES is the current favourite; otherwise
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
