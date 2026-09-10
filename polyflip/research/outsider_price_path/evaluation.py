"""
polyflip/research/outsider_price_path/evaluation.py

Economics accounting, stratified comparisons, policy evaluations, and day-block bootstrap (Stages 4 & 5).
Features:
- Pure line-item economics (gross, 0.2% fee, 0.1% scenario fee, zero fee).
- Fixed 5-policy evaluation matrix (Control, CT, Former Favorite, Rebound, Joint Filter).
- 2x2 trajectory matrix with orthogonal CT breakdown.
- Multi-dimensional stratification (asset, side, ask bin 0.05, calendar week) with common-support weighting.
- Day-block paired bootstrap preserving all markets of each calendar day together.
- Holm-Bonferroni multiple-comparison adjustment.
- Concentration audits (top 5 wins, best day exclusion, weekly breakdown).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence, Mapping, Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpportunityEconomics:
    """Economics of a single opportunity with $1 budget."""
    ask: float
    target: int  # 1 for win, 0 for loss
    budget_usdc: float = 1.0
    shares: float = 0.0
    gross_pnl: float = 0.0
    fee_02pct: float = 0.0
    net_pnl_02pct: float = 0.0
    fee_01pct: float = 0.0
    net_pnl_01pct: float = 0.0

    @classmethod
    def compute(cls, ask: float, target: int, budget: float = 1.0) -> OpportunityEconomics:
        if ask <= 0.0 or not math.isfinite(ask):
            return cls(ask=ask, target=target, budget_usdc=budget)
        shares = budget / ask
        gross_pnl = shares * (target - ask)
        fee_02 = budget * 0.002
        net_02 = gross_pnl - fee_02
        fee_01 = budget * 0.001
        net_01 = gross_pnl - fee_01
        return cls(
            ask=round(ask, 6),
            target=target,
            budget_usdc=budget,
            shares=round(shares, 6),
            gross_pnl=round(gross_pnl, 6),
            fee_02pct=round(fee_02, 6),
            net_pnl_02pct=round(net_02, 6),
            fee_01pct=round(fee_01, 6),
            net_pnl_01pct=round(net_01, 6),
        )


@dataclass
class PolicySummary:
    policy_name: str
    n_trades: int
    n_days: int
    mean_ask: float
    win_count: int
    win_rate: float
    turnover_usdc: float
    gross_pnl: float
    net_pnl_02pct: float
    net_pnl_01pct: float
    expectancy_usdc: float
    expectancy_roi_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "n_trades": self.n_trades,
            "n_days": self.n_days,
            "mean_ask": round(self.mean_ask, 4),
            "win_count": self.win_count,
            "win_rate": round(self.win_rate, 4),
            "turnover_usdc": round(self.turnover_usdc, 2),
            "gross_pnl": round(self.gross_pnl, 4),
            "net_pnl_02pct": round(self.net_pnl_02pct, 4),
            "net_pnl_01pct": round(self.net_pnl_01pct, 4),
            "expectancy_usdc": round(self.expectancy_usdc, 5),
            "expectancy_roi_pct": round(self.expectancy_roi_pct, 4),
        }


def summarize_policy(
    df_trades: pd.DataFrame,
    policy_name: str,
) -> PolicySummary:
    """Computes summary statistics for a trade selection."""
    n = len(df_trades)
    if n == 0:
        return PolicySummary(
            policy_name=policy_name,
            n_trades=0,
            n_days=0,
            mean_ask=0.0,
            win_count=0,
            win_rate=0.0,
            turnover_usdc=0.0,
            gross_pnl=0.0,
            net_pnl_02pct=0.0,
            net_pnl_01pct=0.0,
            expectancy_usdc=0.0,
            expectancy_roi_pct=0.0,
        )

    days = df_trades["calendar_date"].nunique() if "calendar_date" in df_trades else 0
    mean_ask = float(df_trades["ask"].mean())
    wins = int(df_trades["target"].sum())
    wr = wins / n
    turnover = float(df_trades["budget_usdc"].sum()) if "budget_usdc" in df_trades else float(n)
    gross = float(df_trades["gross_pnl"].sum())
    net_02 = float(df_trades["net_pnl_02pct"].sum())
    net_01 = float(df_trades["net_pnl_01pct"].sum()) if "net_pnl_01pct" in df_trades else (gross - turnover * 0.001)
    exp_usdc = net_02 / n
    exp_roi = (net_02 / turnover * 100.0) if turnover > 0 else 0.0

    return PolicySummary(
        policy_name=policy_name,
        n_trades=n,
        n_days=days,
        mean_ask=mean_ask,
        win_count=wins,
        win_rate=wr,
        turnover_usdc=turnover,
        gross_pnl=gross,
        net_pnl_02pct=net_02,
        net_pnl_01pct=net_01,
        expectancy_usdc=exp_usdc,
        expectancy_roi_pct=exp_roi,
    )


def compute_stratified_comparison(
    df: pd.DataFrame,
    group_col: str,
    group_a: str = "FORMER_FAVORITE",
    group_b: str = "OBSERVED_ALWAYS_OUTSIDER",
) -> dict[str, Any]:
    """
    Stratifies sample by (asset, side, ask_bin, calendar_week) to control for
    entry price, asset mix, and time period (Item 21).
    """
    req_cols = ["asset", "side", "ask_bin", "calendar_week", group_col, "target", "net_pnl_02pct"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} for stratification")

    sub = df[df[group_col].isin([group_a, group_b])].copy()
    strata_keys = ["asset", "side", "ask_bin", "calendar_week"]
    
    # Group by strata
    grouped = sub.groupby(strata_keys)
    
    common_strata = []
    lost_rows_a = 0
    lost_rows_b = 0

    for name, grp in grouped:
        counts = grp[group_col].value_counts()
        has_a = counts.get(group_a, 0)
        has_b = counts.get(group_b, 0)
        if has_a > 0 and has_b > 0:
            common_strata.append(name)
        else:
            lost_rows_a += has_a
            lost_rows_b += has_b

    common_df = sub[sub.set_index(strata_keys).index.isin(common_strata)].copy()

    total_sub_rows = len(sub)
    common_rows = len(common_df)
    coverage_loss_pct = (1.0 - (common_rows / total_sub_rows)) * 100.0 if total_sub_rows > 0 else 0.0

    # Within common strata, calculate stratum-weighted differences
    # Equal weight per common stratum
    stratum_diffs_wr = []
    stratum_diffs_pnl = []
    stratum_weights = []

    for name in common_strata:
        grp = common_df[
            (common_df["asset"] == name[0]) &
            (common_df["side"] == name[1]) &
            (common_df["ask_bin"] == name[2]) &
            (common_df["calendar_week"] == name[3])
        ]
        grp_a = grp[grp[group_col] == group_a]
        grp_b = grp[grp[group_col] == group_b]
        
        wr_a = grp_a["target"].mean()
        wr_b = grp_b["target"].mean()
        pnl_a = grp_a["net_pnl_02pct"].mean()
        pnl_b = grp_b["net_pnl_02pct"].mean()

        # Harmonic mean of sample sizes as stratum weight
        na, nb = len(grp_a), len(grp_b)
        w = (2.0 * na * nb) / (na + nb)

        stratum_diffs_wr.append((wr_a - wr_b) * w)
        stratum_diffs_pnl.append((pnl_a - pnl_b) * w)
        stratum_weights.append(w)

    sum_w = sum(stratum_weights) if stratum_weights else 0.0
    adjusted_wr_diff = (sum(stratum_diffs_wr) / sum_w) if sum_w > 0 else 0.0
    adjusted_pnl_diff = (sum(stratum_diffs_pnl) / sum_w) if sum_w > 0 else 0.0

    return {
        "group_a": group_a,
        "group_b": group_b,
        "total_rows": int(total_sub_rows),
        "common_support_rows": int(common_rows),
        "common_strata_count": int(len(common_strata)),
        "lost_rows_a": int(lost_rows_a),
        "lost_rows_b": int(lost_rows_b),
        "coverage_loss_pct": round(coverage_loss_pct, 2),
        "adjusted_win_rate_diff": round(adjusted_wr_diff, 5),
        "adjusted_expectancy_diff_usdc": round(adjusted_pnl_diff, 5),
    }


def run_day_block_bootstrap(
    df: pd.DataFrame,
    policies: Mapping[str, pd.Series],  # policy_name -> boolean mask of selected rows
    n_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Day-block paired bootstrap preserving all markets of each calendar day together (Item 25).
    Recomputes sums, counts, ratios and differences independently in each replicate.
    """
    rng = np.random.default_rng(seed)
    unique_days = np.array(sorted(df["calendar_date"].unique()))
    n_days = len(unique_days)
    if n_days == 0:
        return {}

    # Pre-index data by day for speed
    day_indices: dict[str, np.ndarray] = {}
    for d in unique_days:
        day_indices[d] = np.where(df["calendar_date"].values == d)[0]

    policy_names = list(policies.keys())
    policy_masks = {p: policies[p].values for p in policy_names}

    targets = df["target"].values.astype(float)
    net_pnls = df["net_pnl_02pct"].values.astype(float)
    budgets = df["budget_usdc"].values.astype(float)

    # Accumulators for bootstrap distributions
    boot_stats = {
        p: {
            "n_trades": np.zeros(n_replicates),
            "win_rate": np.zeros(n_replicates),
            "net_pnl": np.zeros(n_replicates),
            "expectancy": np.zeros(n_replicates),
        }
        for p in policy_names
    }

    # For paired differences vs Control
    ctrl_name = "Control"
    diff_stats: dict[str, dict[str, np.ndarray]] = {}
    for p in policy_names:
        if p != ctrl_name:
            diff_stats[f"{p}_minus_{ctrl_name}"] = {
                "d_win_rate": np.zeros(n_replicates),
                "d_net_pnl": np.zeros(n_replicates),
                "d_expectancy": np.zeros(n_replicates),
            }

    for b in range(n_replicates):
        sample_days = rng.choice(unique_days, size=n_days, replace=True)
        sample_row_idx = np.concatenate([day_indices[d] for d in sample_days])

        b_targets = targets[sample_row_idx]
        b_pnls = net_pnls[sample_row_idx]
        b_budgets = budgets[sample_row_idx]

        for p in policy_names:
            m = policy_masks[p][sample_row_idx]
            n_tr = int(np.sum(m))
            boot_stats[p]["n_trades"][b] = n_tr
            if n_tr > 0:
                p_targets = b_targets[m]
                p_pnls = b_pnls[m]
                wr = np.sum(p_targets) / n_tr
                tot_pnl = np.sum(p_pnls)
                exp = tot_pnl / n_tr
            else:
                wr = 0.0
                tot_pnl = 0.0
                exp = 0.0

            boot_stats[p]["win_rate"][b] = wr
            boot_stats[p]["net_pnl"][b] = tot_pnl
            boot_stats[p]["expectancy"][b] = exp

        if ctrl_name in policy_names:
            c_wr = boot_stats[ctrl_name]["win_rate"][b]
            c_pnl = boot_stats[ctrl_name]["net_pnl"][b]
            c_exp = boot_stats[ctrl_name]["expectancy"][b]
            for p in policy_names:
                if p != ctrl_name:
                    k = f"{p}_minus_{ctrl_name}"
                    diff_stats[k]["d_win_rate"][b] = boot_stats[p]["win_rate"][b] - c_wr
                    diff_stats[k]["d_net_pnl"][b] = boot_stats[p]["net_pnl"][b] - c_pnl
                    diff_stats[k]["d_expectancy"][b] = boot_stats[p]["expectancy"][b] - c_exp

    results: dict[str, Any] = {
        "n_replicates": n_replicates,
        "n_days": int(n_days),
        "policies": {},
        "paired_differences": {},
    }

    def _ci(arr: np.ndarray) -> list[float]:
        return [round(float(np.percentile(arr, 2.5)), 5), round(float(np.percentile(arr, 97.5)), 5)]

    for p in policy_names:
        results["policies"][p] = {
            "win_rate_mean": round(float(np.mean(boot_stats[p]["win_rate"])), 4),
            "win_rate_ci95": _ci(boot_stats[p]["win_rate"]),
            "net_pnl_mean": round(float(np.mean(boot_stats[p]["net_pnl"])), 2),
            "net_pnl_ci95": _ci(boot_stats[p]["net_pnl"]),
            "expectancy_mean": round(float(np.mean(boot_stats[p]["expectancy"])), 5),
            "expectancy_ci95": _ci(boot_stats[p]["expectancy"]),
        }

    raw_pvals = {}
    for k, d in diff_stats.items():
        # Empirical p-value for H0: difference <= 0 (fraction of bootstrap <= 0)
        # One-sided for superiority, two-sided p:
        p_superior = float(np.mean(d["d_expectancy"] <= 0.0))
        p_two_sided = min(1.0, 2.0 * min(p_superior, 1.0 - p_superior))
        raw_pvals[k] = p_two_sided

        results["paired_differences"][k] = {
            "d_win_rate_mean": round(float(np.mean(d["d_win_rate"])), 4),
            "d_win_rate_ci95": _ci(d["d_win_rate"]),
            "d_net_pnl_mean": round(float(np.mean(d["d_net_pnl"])), 2),
            "d_net_pnl_ci95": _ci(d["d_net_pnl"]),
            "d_expectancy_mean": round(float(np.mean(d["d_expectancy"])), 5),
            "d_expectancy_ci95": _ci(d["d_expectancy"]),
            "nominal_p_value": round(p_two_sided, 4),
        }

    # Holm-Bonferroni correction over family of comparisons
    sorted_comps = sorted(raw_pvals.items(), key=lambda x: x[1])
    m_comps = len(sorted_comps)
    for rank, (comp_name, p_val) in enumerate(sorted_comps):
        adj_p = min(1.0, p_val * (m_comps - rank))
        results["paired_differences"][comp_name]["holm_adj_p_value"] = round(adj_p, 4)
        results["paired_differences"][comp_name]["is_significant_05"] = bool(adj_p < 0.05)

    return results


def compute_concentration_audit(
    df_trades: pd.DataFrame,
) -> dict[str, Any]:
    """
    Audits trade concentration (Item 26):
    - Share of total profit from top 5 winning trades
    - Stress test without the most profitable calendar day
    - Weekly stability
    - Asset breakdown
    """
    if len(df_trades) == 0:
        return {}

    total_net = float(df_trades["net_pnl_02pct"].sum())
    
    # Top 5 wins
    wins = df_trades[df_trades["target"] == 1].sort_values("net_pnl_02pct", ascending=False)
    top5 = wins.head(5)
    top5_sum = float(top5["net_pnl_02pct"].sum())
    top5_share_pct = (top5_sum / total_net * 100.0) if total_net > 0 else 0.0

    # Best day exclusion stress test
    day_pnl = df_trades.groupby("calendar_date")["net_pnl_02pct"].sum()
    best_day = str(day_pnl.idxmax())
    best_day_pnl = float(day_pnl.max())
    pnl_without_best_day = total_net - best_day_pnl

    # Weekly breakdown
    weekly = df_trades.groupby("calendar_week").agg(
        trades=("market_id", "count"),
        wins=("target", "sum"),
        net_pnl=("net_pnl_02pct", "sum")
    ).reset_index()
    weekly["win_rate"] = weekly["wins"] / weekly["trades"]
    pos_weeks = int((weekly["net_pnl"] > 0).sum())
    total_weeks = len(weekly)

    # Asset breakdown
    by_asset = df_trades.groupby("asset").agg(
        trades=("market_id", "count"),
        wins=("target", "sum"),
        net_pnl=("net_pnl_02pct", "sum"),
        mean_ask=("ask", "mean"),
    ).reset_index()
    by_asset["win_rate"] = by_asset["wins"] / by_asset["trades"]

    return {
        "total_net_pnl": round(total_net, 4),
        "top5_wins_sum": round(top5_sum, 4),
        "top5_share_of_total_pct": round(top5_share_pct, 2),
        "best_day": best_day,
        "best_day_pnl": round(best_day_pnl, 4),
        "pnl_without_best_day": round(pnl_without_best_day, 4),
        "weekly_summary": {
            "total_weeks": total_weeks,
            "positive_weeks": pos_weeks,
            "positive_week_pct": round((pos_weeks / total_weeks * 100.0) if total_weeks > 0 else 0.0, 1),
            "weeks": weekly.to_dict("records"),
        },
        "by_asset": by_asset.to_dict("records"),
    }
