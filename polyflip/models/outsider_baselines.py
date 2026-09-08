"""
polyflip/models/outsider_baselines.py

Transparent price baselines for outsider probability modeling (Item 2.5):
- M0: Market price baseline, p_win = outsider_mid
- Mlegacy: Legacy baseline adapting historical heuristic / leaning model
Evaluated using identical candidate_side, target, and canonical metrics.
Fees and ask are accounted for separately in EV, without contaminating probabilities.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from polyflip.models.probability_metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
)
from polyflip.crypto.edge import compute_net_ev_per_share


class MarketPriceBaseline:
    """
    Baseline M0: Pure market price probability.
    p_win = outsider_mid
    """

    def __init__(self, name: str = "M0_MARKET_PRICE"):
        self.name = name

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MarketPriceBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "outsider_mid" in df.columns:
            m = df["outsider_mid"]
        elif "mid_price" in df.columns:
            m = df["mid_price"]
        else:
            m = pd.Series(0.5, index=df.index)

        probs = pd.to_numeric(m, errors="coerce").fillna(0.5).to_numpy()
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        # Return 2D array [P(0), P(1)]
        return np.column_stack([1.0 - probs, probs])


class LegacyOutsiderBaseline:
    """
    Baseline Mlegacy: Adapts the legacy BTC leaning / momentum rule.
    Adjusts market mid with legacy velocity and spread terms, clipped to [0.01, 0.99].
    """

    def __init__(
        self,
        velocity_weight: float = 0.05,
        spread_penalty: float = 0.10,
        name: str = "MLEGACY_LEANING",
    ):
        self.name = name
        self.velocity_weight = velocity_weight
        self.spread_penalty = spread_penalty

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LegacyOutsiderBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "outsider_mid" in df.columns:
            m = df["outsider_mid"]
        elif "mid_price" in df.columns:
            m = df["mid_price"]
        else:
            m = pd.Series(0.5, index=df.index)

        base_p = pd.to_numeric(m, errors="coerce").fillna(0.5).to_numpy()

        vel_series = (
            df["price_velocity"]
            if "price_velocity" in df.columns
            else (
                df["legacy_last_poll_delta"]
                if "legacy_last_poll_delta" in df.columns
                else pd.Series(0.0, index=df.index)
            )
        )
        vel_vals = pd.to_numeric(vel_series, errors="coerce").fillna(0.0).to_numpy()

        spread_series = (
            df["spread"]
            if "spread" in df.columns
            else (
                df["candidate_spread"]
                if "candidate_spread" in df.columns
                else pd.Series(0.02, index=df.index)
            )
        )
        spr_vals = pd.to_numeric(spread_series, errors="coerce").fillna(0.02).to_numpy()

        adjusted = base_p + self.velocity_weight * vel_vals - self.spread_penalty * spr_vals
        probs = np.clip(adjusted, 0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


def evaluate_outsider_predictions(
    y_true: Sequence[int | float] | np.ndarray,
    p_win: Sequence[float] | np.ndarray,
    executable_ask: Sequence[float] | np.ndarray | None = None,
    fee_rate: float = 0.002,
    min_edge: float = 0.02,
) -> dict[str, Any]:
    """
    Evaluates predictions with canonical probability metrics (Brier, LogLoss, ECE)
    and economic simulation metrics (trades, win_rate, total_pnl, net_expectancy).
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p_win, dtype=float)

    n_samples = len(y_arr)
    if n_samples == 0:
        return {
            "n_samples": 0,
            "brier": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "expectancy": 0.0,
        }

    brier = brier_score(y_arr, p_arr)
    log_loss = log_loss_score(y_arr, p_arr)
    ece_val, _ = expected_calibration_error(y_arr, p_arr, n_bins=20)
    ece = float(ece_val) if ece_val is not None else 0.0

    # Economic simulation if ask is provided
    n_trades = 0
    total_pnl = 0.0
    wins = 0
    trade_pnls = []

    if executable_ask is not None:
        ask_arr = np.asarray(executable_ask, dtype=float)
        for i in range(n_samples):
            ask = ask_arr[i]
            p = p_arr[i]
            # Net EV in USDC per share
            net_ev = compute_net_ev_per_share(
                p_win=p,
                executable_ask=ask,
                fee_per_share=ask * fee_rate,
            )
            # Trade if net edge exceeds threshold
            if net_ev >= min_edge and ask < 0.95:
                n_trades += 1
                outcome = y_arr[i]
                fee = ask * fee_rate
                # PnL per share: 1 - ask - fee if win, -ask - fee if lose
                pnl = (1.0 - ask - fee) if outcome == 1.0 else (-ask - fee)
                total_pnl += pnl
                trade_pnls.append(pnl)
                if outcome == 1.0:
                    wins += 1

    win_rate = (wins / n_trades) if n_trades > 0 else 0.0
    expectancy = (total_pnl / n_trades) if n_trades > 0 else 0.0

    return {
        "n_samples": n_samples,
        "brier": round(float(brier), 6),
        "log_loss": round(float(log_loss), 6),
        "ece": round(float(ece), 6),
        "n_trades": n_trades,
        "win_rate": round(float(win_rate), 4),
        "total_pnl": round(float(total_pnl), 4),
        "expectancy": round(float(expectancy), 6),
    }
