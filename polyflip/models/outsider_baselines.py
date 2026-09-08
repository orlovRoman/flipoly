"""
polyflip/models/outsider_baselines.py

Transparent price baselines for outsider probability modeling (Item 2.5):
- M0: Market price baseline, p_win = outsider_mid
- Mlegacy: Legacy baseline adapting historical heuristic / leaning model
Evaluated using identical candidate_side, target, and canonical metrics.
Fees and ask are accounted for separately in EV, without contaminating probabilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
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
    Baseline Mlegacy: Adapts the authentic BTC_leaning@11 production model
    or legacy heuristic rule.
    Uses features [mid_price, spread, time_left_min] with trained coefficients,
    or legacy velocity/spread adjustments when parameters are provided.
    """

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        name: str = "MLEGACY_LEANING",
        velocity_weight: float | None = None,
        spread_penalty: float | None = None,
    ):
        self.name = name
        self.velocity_weight = velocity_weight
        self.spread_penalty = spread_penalty
        self.is_authentic_model = (velocity_weight is None and spread_penalty is None)
        self.model_version = 11
        self._model = None

        if self.is_authentic_model:
            if artifact_path is None:
                default_path = Path(__file__).resolve().parent / "artifacts" / "btc_leaning_v11.pkl"
                if default_path.exists():
                    artifact_path = default_path

            if artifact_path is not None and Path(artifact_path).exists():
                import pickle
                try:
                    with open(artifact_path, "rb") as f:
                        self._model = pickle.load(f)
                except Exception:
                    self._model = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LegacyOutsiderBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if "outsider_mid" in df.columns:
            m = df["outsider_mid"]
        elif "mid_price" in df.columns:
            m = df["mid_price"]
        else:
            m = pd.Series(0.5, index=df.index)
        mid_vals = pd.to_numeric(m, errors="coerce").fillna(0.5).to_numpy()

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

        if self.velocity_weight is not None or self.spread_penalty is not None:
            vw = self.velocity_weight or 0.0
            sp = self.spread_penalty or 0.0
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
            adjusted = mid_vals + vw * vel_vals - sp * spr_vals
            probs = np.clip(adjusted, 0.01, 0.99)
            return np.column_stack([1.0 - probs, probs])

        time_series = (
            df["time_left_min"]
            if "time_left_min" in df.columns
            else pd.Series(5.0, index=df.index)
        )
        time_vals = pd.to_numeric(time_series, errors="coerce").fillna(5.0).to_numpy()

        # Item 1.19: Determine candidate side and canonical YES mid
        if "candidate_side" in df.columns:
            sides = df["candidate_side"].astype(str).str.strip().str.upper().to_numpy()
        else:
            sides = np.array(["UP"] * len(df))

        if "yes_mid" in df.columns:
            yes_mid_vals = pd.to_numeric(df["yes_mid"], errors="coerce").fillna(0.5).to_numpy()
        elif "canonical_yes_mid" in df.columns:
            yes_mid_vals = pd.to_numeric(df["canonical_yes_mid"], errors="coerce").fillna(0.5).to_numpy()
        else:
            is_down = np.isin(sides, ["DOWN", "NO"])
            # If candidate is DOWN, outsider_mid is 1 - yes_mid -> yes_mid is 1 - outsider_mid
            yes_mid_vals = np.where(is_down, 1.0 - mid_vals, mid_vals)

        X_df = pd.DataFrame({
            "mid_price": yes_mid_vals,
            "spread": spr_vals,
            "time_left_min": time_vals,
        })

        if self._model is not None and hasattr(self._model, "predict_proba"):
            p_flip = self._model.predict_proba(X_df)[:, 1]
        else:
            # Fallback to authentic parameters from Model ID 827
            # Formula: logit(p_flip) = -0.75918691 - 0.20202834*yes_mid - 0.26294307*spread + 0.01182981*time_left_min
            coef = np.array([-0.20202834, -0.26294307, 0.01182981])
            intercept = -0.75918691
            z = X_df.to_numpy() @ coef + intercept
            p_flip = 1.0 / (1.0 + np.exp(-z))

        # BTC_leaning@11 was trained on target=flip; its output is p_flip directly (candidate win probability)
        # for both UP (yes_mid < 0.5) and DOWN (yes_mid > 0.5). Must NOT invert for DOWN!
        probs = np.clip(p_flip, 0.01, 0.99)
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
    valid_mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    coverage = float(np.mean(valid_mask)) if n_samples > 0 else 0.0

    if n_samples == 0 or not np.any(valid_mask):
        return {
            "n_samples": n_samples,
            "coverage": coverage,
            "brier": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "expectancy": 0.0,
        }

    brier = brier_score(y_arr[valid_mask], p_arr[valid_mask])
    log_loss = log_loss_score(y_arr[valid_mask], p_arr[valid_mask])
    ece_val, _ = expected_calibration_error(y_arr[valid_mask], p_arr[valid_mask], n_bins=20)
    ece = float(ece_val) if ece_val is not None else 0.0

    # Economic simulation if ask is provided
    n_trades = 0
    total_pnl = 0.0
    wins = 0
    trade_pnls = []

    if executable_ask is not None:
        ask_arr = np.asarray(executable_ask, dtype=float)
        for i in np.where(valid_mask)[0]:
            ask = ask_arr[i]
            p = p_arr[i]
            if not np.isfinite(ask) or ask <= 0.0 or ask >= 1.0:
                continue
            # Net EV in USDC per share
            net_ev = compute_net_ev_per_share(
                p_win=p,
                executable_ask=ask,
                fee_per_share=ask * fee_rate,
            )
            # Trade if net edge exceeds threshold
            if pd.notna(net_ev) and net_ev >= min_edge and ask < 0.95:
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
