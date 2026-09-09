"""Causal CT/CS regime features for strike-baseline Phase 2b (protocol v1.3, frozen).

CT (token): classify_local_regime over the last <= CT_MAX_MIDS valid token mids
  of the SAME market with recorded_at STRICTLY before the decision snapshot.
CS (spot): classify_spot_regime_short over the last <= CS_MAX_CANDLES closed
  Binance 1m candles with close_time <= decision snapshot time (window 10m,
  require_valid_autocorr=True; internal causal alignment + 120s staleness cap).

Both builders are pure functions of past data: appending any observation at or
after the decision moment cannot change the result (tested in synthetic_checks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from polyflip.research.regime_features import (
    classify_local_regime,
    classify_spot_regime_short,
)

CT_MAX_MIDS = 30
CT_MIN_OBS = 4
CS_MAX_CANDLES = 30

REGIME_STATES = ("REVERSION", "TREND", "QUIET", "UNCERTAIN")


def _clean_mid(bid: float, ask: float) -> float | None:
    if bid is None or ask is None:
        return None
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(b) and np.isfinite(a)):
        return None
    if 0.0 < b < 1.0 and 0.0 < a < 1.0 and b < a:
        return (b + a) / 2.0
    return None


def ct_for_decision(market_snaps: pd.DataFrame, snapshot_at: pd.Timestamp) -> dict:
    """Token regime from prior mids of one market (recorded_at < snapshot_at)."""
    prior = market_snaps[market_snaps["recorded_at"] < snapshot_at].sort_values("recorded_at")
    mids: list[float] = []
    for _, r in prior.iterrows():
        m = _clean_mid(r.get("best_bid"), r.get("best_ask"))
        if m is not None:
            mids.append(m)
    mids = mids[-CT_MAX_MIDS:]
    res = classify_local_regime(np.asarray(mids, dtype=float), min_observations=CT_MIN_OBS)
    return {"ct_state": res["state"], "ct_er": res["efficiency_ratio"],
            "ct_n_mids": len(mids), "ct_status": res["status"]}


def cs_for_decision(candle_closes: np.ndarray, candle_times: np.ndarray,
                    snapshot_at: pd.Timestamp) -> dict:
    """Spot regime from closed 1m candles (close_time <= snapshot_at)."""
    res = classify_spot_regime_short(
        np.asarray(candle_closes, dtype=float),
        timestamps=list(candle_times),
        as_of=snapshot_at,
        window_min=10.0,
        require_valid_autocorr=True,
    )
    return {"cs_state": res["state"], "cs_er": res["efficiency_ratio"],
            "cs_status": res.get("classification_reason", res.get("status"))}


def add_state_dummies(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """One-hot for REVERSION/TREND/QUIET; UNCERTAIN (or unknown) = reference."""
    for s in ("REVERSION", "TREND", "QUIET"):
        df[f"{prefix}_{s.lower()}"] = (df[col] == s).astype(int)
    return df
