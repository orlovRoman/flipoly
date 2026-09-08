"""
polyflip/research/reporting_helpers.py

Standardized accounting, drawdown, uncertainty, and reliability reporting helpers (Items 1.24, 1.25, 1.26, R9).
"""
from __future__ import annotations

from typing import Sequence, Any
import numpy as np
import pandas as pd


def compute_drawdown(pnls: Sequence[float], initial_equity: float = 0.0) -> float:
    """
    Computes maximum peak-to-trough drawdown from sequence of trade PnLs,
    properly anchored at initial_equity (default 0.0).
    DD([-1.0, -1.0]) -> 2.0.
    """
    if not pnls or len(pnls) == 0:
        return 0.0
    equity = np.insert(np.cumsum(pnls) + initial_equity, 0, initial_equity)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return round(float(np.max(drawdowns)), 4) if len(drawdowns) > 0 else 0.0


def compute_payoff_ratio(pnls: Sequence[float]) -> float:
    """
    Computes payoff ratio (average win / average loss).
    Returns 0.0 if no trades or no wins; returns inf if wins exist but no losses.
    """
    p_arr = np.asarray(pnls, dtype=float)
    if len(p_arr) == 0:
        return 0.0
    wins = p_arr[p_arr > 0]
    losses = np.abs(p_arr[p_arr < 0])
    if len(wins) == 0:
        return 0.0
    if len(losses) == 0:
        return float("inf")
    return round(float(np.mean(wins) / np.mean(losses)), 4)


def compute_clustered_uncertainty(
    trade_pnls: Sequence[float],
    cluster_ids: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes cluster-robust standard error and 95% confidence intervals for expectancy (mean PnL per trade).
    If <= 1 unique cluster, returns status='INSUFFICIENT_BLOCKS' and NaN intervals.
    """
    if not trade_pnls or len(trade_pnls) == 0:
        return {"se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_clusters": 0, "status": "NO_TRADES"}

    pnls_arr = np.asarray(trade_pnls, dtype=float)
    c_arr = np.asarray(cluster_ids)
    unique_clusters = np.unique(c_arr)
    n_c = len(unique_clusters)
    n_trades = len(pnls_arr)

    if n_c <= 1:
        return {
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_clusters": n_c,
            "status": "INSUFFICIENT_BLOCKS",
        }

    cluster_sums = np.array([np.sum(pnls_arr[c_arr == c]) for c in unique_clusters])
    cluster_counts = np.array([np.sum(c_arr == c) for c in unique_clusters])

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sampled_c = rng.choice(n_c, size=n_c, replace=True)
        tot_pnl = np.sum(cluster_sums[sampled_c])
        tot_trades = np.sum(cluster_counts[sampled_c])
        if tot_trades > 0:
            boot_means.append(tot_pnl / tot_trades)

    if boot_means and len(boot_means) >= 10:
        boot_arr = np.array(boot_means)
        se = float(np.std(boot_arr, ddof=1))
        ci_lower = float(np.percentile(boot_arr, 2.5))
        ci_upper = float(np.percentile(boot_arr, 97.5))
        return {
            "se": round(se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "n_clusters": n_c,
            "status": "OK",
        }

    return {
        "se": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_clusters": n_c,
        "status": "INSUFFICIENT_SAMPLES",
    }


def generate_price_bins_report(
    df: pd.DataFrame,
    price_col: str = "executable_ask",
    prob_col: str = "p_win",
    outcome_col: str = "outcome",
    pnl_col: str = "pnl",
    bin_width: float = 0.05,
    min_price: float = 0.0,
    max_price: float = 0.50,
) -> pd.DataFrame:
    """
    Generates calibration and financial reliability table across price bins of width bin_width.
    """
    if df.empty:
        return pd.DataFrame()

    p_vals = pd.to_numeric(df[price_col], errors="coerce")
    bins = np.arange(min_price, max_price + bin_width, bin_width)
    labels = [f"[{bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    df_bins = df.copy()
    df_bins["price_bin"] = pd.cut(p_vals, bins=bins, labels=labels, right=False, include_lowest=True)

    records = []
    for label in labels:
        sub = df_bins[df_bins["price_bin"] == label]
        n_rows = len(sub)
        n_markets = int(sub["market_id"].nunique()) if "market_id" in sub.columns else n_rows
        p_pred = float(sub[prob_col].mean()) if prob_col in sub.columns and n_rows > 0 else np.nan
        realized_freq = float(sub[outcome_col].mean()) if outcome_col in sub.columns and n_rows > 0 else np.nan
        total_pnl = float(sub[pnl_col].sum()) if pnl_col in sub.columns and n_rows > 0 else 0.0
        avg_cost = float(sub[price_col].mean()) if price_col in sub.columns and n_rows > 0 else np.nan

        records.append({
            "price_bin": label,
            "n_rows": n_rows,
            "n_markets": n_markets,
            "predicted_mean": round(p_pred, 4) if np.isfinite(p_pred) else None,
            "realized_frequency": round(realized_freq, 4) if np.isfinite(realized_freq) else None,
            "avg_cost": round(avg_cost, 4) if np.isfinite(avg_cost) else None,
            "total_pnl": round(total_pnl, 4),
        })

    return pd.DataFrame(records)


def compute_paired_cluster_delta(
    pnls_model: Sequence[float],
    indices_model: Sequence[int],
    pnls_ref: Sequence[float],
    indices_ref: Sequence[int],
    cluster_ids_full: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes paired cluster-level delta PnL between candidate model and reference model.
    For each cluster (market or date): delta_pnl_c = sum(pnl_cand in c) - sum(pnl_ref in c).
    """
    clusters_full = np.asarray(cluster_ids_full)
    unique_clusters = np.unique(clusters_full)
    n_c = len(unique_clusters)

    # Build cluster sum maps
    c_pnl_cand = {c: 0.0 for c in unique_clusters}
    for pnl, idx in zip(pnls_model, indices_model):
        c_pnl_cand[clusters_full[idx]] += pnl

    c_pnl_ref = {c: 0.0 for c in unique_clusters}
    for pnl, idx in zip(pnls_ref, indices_ref):
        c_pnl_ref[clusters_full[idx]] += pnl

    deltas = np.array([c_pnl_cand[c] - c_pnl_ref[c] for c in unique_clusters])
    total_delta = float(np.sum(deltas))

    if n_c <= 1:
        return {
            "delta_pnl": round(total_delta, 4),
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "status": "INSUFFICIENT_BLOCKS",
        }

    # Standard error of sum of cluster deltas
    se_delta = float(np.sqrt(n_c) * np.std(deltas, ddof=1))
    t_stat = total_delta / se_delta if se_delta > 1e-9 else 0.0

    # Cluster bootstrap for delta PnL
    rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(n_bootstrap):
        sampled_c = rng.choice(n_c, size=n_c, replace=True)
        boot_deltas.append(np.sum(deltas[sampled_c]))

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    return {
        "delta_pnl": round(total_delta, 4),
        "se": round(se_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "t_stat": round(t_stat, 3),
        "status": "OK",
    }
