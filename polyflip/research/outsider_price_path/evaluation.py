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

from polyflip.research.trade_economics.pnl import apply_fee_to_budget, calculate_net_pnl


@dataclass(frozen=True)
class OpportunityEconomics:
    """Economics of a single opportunity with $1 budget using verified trade_economics."""
    ask: float
    target: int  # 1 for win, 0 for loss
    budget_usdc: float = 1.0
    shares: float = 0.0
    gross_pnl: float = 0.0
    fee_02pct: float = 0.0
    net_pnl_02pct: float = 0.0
    fee_01pct: float = 0.0
    net_pnl_01pct: float = 0.0
    confirmed_net_pnl: Optional[float] = None  # None indicates exchange fee schedule unconfirmed
    execution_type: str = "TOP_OF_BOOK_ASK"  # Depth simulation unavailable in historical snapshots

    @classmethod
    def compute(cls, ask: float, target: int, budget: float = 1.0) -> OpportunityEconomics:
        if ask <= 0.0 or not math.isfinite(ask):
            return cls(ask=ask, target=target, budget_usdc=budget)
        
        # Use verified trade_economics module
        actual_cost_02, shares, fee_02 = apply_fee_to_budget(
            total_budget=budget,
            commission_rate=0.002,
            price=ask,
            mode='exclusive'
        )
        payout = float(shares) if target == 1 else 0.0
        net_02 = calculate_net_pnl(
            sale_proceeds=0.0,
            settlement_proceeds=payout,
            purchase_cash=budget,
            platform_fees=float(fee_02),
            attributable_network_costs=0.0,
            confirmed_rebates=0.0,
            is_fully_closed=True,
            fee_is_uncertain=False
        ) or 0.0

        _, _, fee_01 = apply_fee_to_budget(
            total_budget=budget,
            commission_rate=0.001,
            price=ask,
            mode='exclusive'
        )
        net_01 = calculate_net_pnl(
            sale_proceeds=0.0,
            settlement_proceeds=payout,
            purchase_cash=budget,
            platform_fees=float(fee_01),
            attributable_network_costs=0.0,
            confirmed_rebates=0.0,
            is_fully_closed=True,
            fee_is_uncertain=False
        ) or 0.0

        gross_pnl = payout - budget

        return cls(
            ask=round(ask, 6),
            target=target,
            budget_usdc=budget,
            shares=round(float(shares), 6),
            gross_pnl=round(gross_pnl, 6),
            fee_02pct=round(float(fee_02), 6),
            net_pnl_02pct=round(net_02, 6),
            fee_01pct=round(float(fee_01), 6),
            net_pnl_01pct=round(net_01, 6),
            confirmed_net_pnl=None,
            execution_type="TOP_OF_BOOK_ASK",
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
    total_universe_trades: int = 0
    stream_expectancy_usdc: float = 0.0
    filter_pass_rate_pct: float = 0.0
    gross_expectancy_usdc: float = 0.0
    stream_gross_expectancy_usdc: float = 0.0

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
            "total_universe_trades": self.total_universe_trades,
            "stream_expectancy_usdc": round(self.stream_expectancy_usdc, 5),
            "filter_pass_rate_pct": round(self.filter_pass_rate_pct, 2),
            "gross_expectancy_usdc": round(self.gross_expectancy_usdc, 5),
            "stream_gross_expectancy_usdc": round(self.stream_gross_expectancy_usdc, 5),
        }


def summarize_policy(
    df_trades: pd.DataFrame,
    policy_name: str,
    total_universe_trades: Optional[int] = None,
) -> PolicySummary:
    """Computes summary statistics for a trade selection, including total-stream expectancy (Items 22 & 23)."""
    n = len(df_trades)
    u_total = total_universe_trades if total_universe_trades is not None and total_universe_trades > 0 else n
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
            total_universe_trades=u_total,
            stream_expectancy_usdc=0.0,
            filter_pass_rate_pct=0.0,
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
    gross_exp_usdc = gross / n
    exp_roi = (net_02 / turnover * 100.0) if turnover > 0 else 0.0
    stream_exp = net_02 / u_total if u_total > 0 else 0.0
    stream_gross_exp = gross / u_total if u_total > 0 else 0.0
    pass_rate = (n / u_total * 100.0) if u_total > 0 else 100.0

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
        total_universe_trades=u_total,
        stream_expectancy_usdc=stream_exp,
        filter_pass_rate_pct=pass_rate,
        gross_expectancy_usdc=gross_exp_usdc,
        stream_gross_expectancy_usdc=stream_gross_exp,
    )


def compute_stratified_comparison(
    df: pd.DataFrame,
    group_col: str,
    group_a: str = "FORMER_FAVORITE",
    group_b: str = "OBSERVED_ALWAYS_OUTSIDER",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Stratifies sample by (asset, side, ask_bin, calendar_week) to control for
    entry price, asset mix, and time period (Item 21).
    Computes precision-weighted and equal-weighted differences for:
    - win rate
    - expectancy ($)
    - residual ask price ($)
    And computes day-block bootstrap confidence intervals (B=n_boot).
    """
    req_cols = ["asset", "side", "ask_bin", "calendar_week", group_col, "target", "net_pnl_02pct"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} for stratification")

    sub = df[df[group_col].isin([group_a, group_b])].copy()
    strata_keys = ["asset", "side", "ask_bin", "calendar_week"]
    
    sub["_stratum"] = sub[strata_keys].astype(str).agg("_".join, axis=1)
    
    counts = sub.groupby(["_stratum", group_col]).size().unstack(fill_value=0)
    has_both = (counts.get(group_a, 0) > 0) & (counts.get(group_b, 0) > 0)
    common_strata = counts[has_both].index.tolist()

    lost_rows_a = int(counts.loc[~has_both, group_a].sum()) if group_a in counts else 0
    lost_rows_b = int(counts.loc[~has_both, group_b].sum()) if group_b in counts else 0

    common_df = sub[sub["_stratum"].isin(common_strata)].copy()
    total_sub_rows = len(sub)
    common_rows = len(common_df)
    coverage_loss_pct = (1.0 - (common_rows / total_sub_rows)) * 100.0 if total_sub_rows > 0 else 0.0

    if common_rows == 0:
        return {
            "group_a": group_a,
            "group_b": group_b,
            "total_rows": int(total_sub_rows),
            "common_support_rows": 0,
            "common_strata_count": 0,
            "lost_rows_a": int(lost_rows_a),
            "lost_rows_b": int(lost_rows_b),
            "coverage_loss_pct": 100.0,
            "adjusted_win_rate_diff": 0.0,
            "adjusted_expectancy_diff_usdc": 0.0,
            "equal_strata_win_rate_diff": 0.0,
            "equal_strata_expectancy_diff_usdc": 0.0,
            "residual_ask_diff_precision": 0.0,
            "residual_ask_diff_equal": 0.0,
            "raw_mean_ask_a": 0.0,
            "raw_mean_ask_b": 0.0,
            "adjusted_win_rate_diff_ci95": [0.0, 0.0],
            "adjusted_expectancy_diff_ci95": [0.0, 0.0],
            "equal_strata_expectancy_diff_ci95": [0.0, 0.0],
            "residual_ask_diff_precision_ci95": [0.0, 0.0],
            "residual_ask_diff_equal_ci95": [0.0, 0.0],
        }

    # Calculate stratum aggregates
    stratum_agg = common_df.groupby(["_stratum", group_col]).agg(
        n=("target", "count"),
        wins=("target", "sum"),
        tot_pnl=("net_pnl_02pct", "sum"),
        tot_ask=("ask", "sum") if "ask" in common_df.columns else ("target", "count")
    ).unstack(fill_value=0)

    na = stratum_agg[("n", group_a)].values
    nb = stratum_agg[("n", group_b)].values
    wra = stratum_agg[("wins", group_a)].values / na
    wrb = stratum_agg[("wins", group_b)].values / nb
    pnla = stratum_agg[("tot_pnl", group_a)].values / na
    pnlb = stratum_agg[("tot_pnl", group_b)].values / nb
    
    if "ask" in common_df.columns:
        aska = stratum_agg[("tot_ask", group_a)].values / na
        askb = stratum_agg[("tot_ask", group_b)].values / nb
    else:
        aska = np.zeros_like(na, dtype=float)
        askb = np.zeros_like(nb, dtype=float)

    # Weights: harmonic mean of sample sizes
    w = (2.0 * na * nb) / (na + nb)
    sum_w = float(np.sum(w))

    adj_wr_diff = float(np.sum((wra - wrb) * w) / sum_w) if sum_w > 0 else 0.0
    adj_pnl_diff = float(np.sum((pnla - pnlb) * w) / sum_w) if sum_w > 0 else 0.0
    eq_wr_diff = float(np.mean(wra - wrb)) if len(wra) > 0 else 0.0
    eq_pnl_diff = float(np.mean(pnla - pnlb)) if len(pnla) > 0 else 0.0

    res_ask_prec = float(np.sum((aska - askb) * w) / sum_w) if sum_w > 0 else 0.0
    res_ask_eq = float(np.mean(aska - askb)) if len(aska) > 0 else 0.0

    result: dict[str, Any] = {
        "group_a": group_a,
        "group_b": group_b,
        "total_rows": int(total_sub_rows),
        "common_support_rows": int(common_rows),
        "common_strata_count": int(len(common_strata)),
        "lost_rows_a": int(lost_rows_a),
        "lost_rows_b": int(lost_rows_b),
        "coverage_loss_pct": round(coverage_loss_pct, 2),
        "adjusted_win_rate_diff": round(adj_wr_diff, 5),
        "adjusted_expectancy_diff_usdc": round(adj_pnl_diff, 5),
        "equal_strata_win_rate_diff": round(eq_wr_diff, 5),
        "equal_strata_expectancy_diff_usdc": round(eq_pnl_diff, 5),
        "residual_ask_diff_precision": round(res_ask_prec, 5),
        "residual_ask_diff_equal": round(res_ask_eq, 5),
        "raw_mean_ask_a": round(float(common_df[common_df[group_col] == group_a]["ask"].mean()), 4) if "ask" in common_df else 0.0,
        "raw_mean_ask_b": round(float(common_df[common_df[group_col] == group_b]["ask"].mean()), 4) if "ask" in common_df else 0.0,
    }

    # Day-block bootstrap for confidence intervals
    if n_boot > 0 and "calendar_date" in common_df.columns:
        rng = np.random.default_rng(seed)
        unique_days = np.array(sorted(common_df["calendar_date"].unique()))
        n_days = len(unique_days)
        day_indices = {d: np.where(common_df["calendar_date"].values == d)[0] for d in unique_days}

        strata_arr = common_df["_stratum"].values
        is_a_arr = (common_df[group_col].values == group_a)
        tgt_arr = common_df["target"].values.astype(float)
        pnl_arr = common_df["net_pnl_02pct"].values.astype(float)
        ask_arr = common_df["ask"].values.astype(float) if "ask" in common_df.columns else np.zeros(len(common_df))

        boot_wr = []
        boot_exp_w = []
        boot_exp_eq = []
        boot_ask_w = []
        boot_ask_eq = []

        for _ in range(n_boot):
            sample_days = rng.choice(unique_days, size=n_days, replace=True)
            row_idx = np.concatenate([day_indices[d] for d in sample_days])
            
            b_df = pd.DataFrame({
                "s": strata_arr[row_idx],
                "a": is_a_arr[row_idx],
                "t": tgt_arr[row_idx],
                "p": pnl_arr[row_idx],
                "k": ask_arr[row_idx],
            })
            b_agg = b_df.groupby(["s", "a"]).agg(
                n=("t", "count"),
                wins=("t", "sum"),
                tot_pnl=("p", "sum"),
                tot_ask=("k", "sum"),
            ).unstack(fill_value=0)

            valid = (b_agg[("n", True)] > 0) & (b_agg[("n", False)] > 0)
            if not valid.any():
                continue
            b_val = b_agg[valid]
            b_na = b_val[("n", True)].values
            b_nb = b_val[("n", False)].values
            b_wra = b_val[("wins", True)].values / b_na
            b_wrb = b_val[("wins", False)].values / b_nb
            b_pnla = b_val[("tot_pnl", True)].values / b_na
            b_pnlb = b_val[("tot_pnl", False)].values / b_nb
            b_aska = b_val[("tot_ask", True)].values / b_na
            b_askb = b_val[("tot_ask", False)].values / b_nb

            bw = (2.0 * b_na * b_nb) / (b_na + b_nb)
            sum_bw = np.sum(bw)
            if sum_bw > 0:
                boot_wr.append(np.sum((b_wra - b_wrb) * bw) / sum_bw)
                boot_exp_w.append(np.sum((b_pnla - b_pnlb) * bw) / sum_bw)
                boot_exp_eq.append(np.mean(b_pnla - b_pnlb))
                boot_ask_w.append(np.sum((b_aska - b_askb) * bw) / sum_bw)
                boot_ask_eq.append(np.mean(b_aska - b_askb))

        def _ci(arr):
            if len(arr) == 0:
                return [0.0, 0.0]
            return [round(float(np.percentile(arr, 2.5)), 5), round(float(np.percentile(arr, 97.5)), 5)]

        result["adjusted_win_rate_diff_ci95"] = _ci(boot_wr)
        result["adjusted_expectancy_diff_ci95"] = _ci(boot_exp_w)
        result["equal_strata_expectancy_diff_ci95"] = _ci(boot_exp_eq)
        result["residual_ask_diff_precision_ci95"] = _ci(boot_ask_w)
        result["residual_ask_diff_equal_ci95"] = _ci(boot_ask_eq)
        if len(boot_exp_eq) > 0:
            result["p_value_equal_strata_expectancy"] = round(float(np.mean(np.array(boot_exp_eq) <= 0.0)), 4)
        if len(boot_exp_w) > 0:
            result["p_value_adjusted_expectancy"] = round(float(np.mean(np.array(boot_exp_w) <= 0.0)), 4)

    return result


def run_day_block_bootstrap(
    df: pd.DataFrame,
    policies: Mapping[str, pd.Series],  # policy_name -> boolean mask of selected rows
    n_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Day-block paired bootstrap preserving all markets of each calendar day together (Item 25).
    Recomputes sums, counts, ratios and differences independently in each replicate.
    Disentangles:
    1. Own policy performance and confidence intervals.
    2. Stream expectancy difference on the full opportunity universe (0 on skips).
    3. Trade expectancy difference on selected trades.
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
    gross_pnls = df["gross_pnl"].values.astype(float) if "gross_pnl" in df.columns else net_pnls

    u_total_orig = len(df)
    ctrl_name = "Control"

    # 1. Point estimates on the original sample
    point_stats = {}
    for p in policy_names:
        m = policy_masks[p]
        n_tr = int(np.sum(m))
        p_targets = targets[m]
        p_pnls = net_pnls[m]
        p_gross = gross_pnls[m]
        wr = float(np.mean(p_targets)) if n_tr > 0 else 0.0
        tot_pnl = float(np.sum(p_pnls))
        tot_gross = float(np.sum(p_gross))
        trade_exp = (tot_pnl / n_tr) if n_tr > 0 else 0.0
        stream_exp = (tot_pnl / u_total_orig) if u_total_orig > 0 else 0.0
        gross_trade_exp = (tot_gross / n_tr) if n_tr > 0 else 0.0
        point_stats[p] = {
            "n_trades": n_tr,
            "win_rate": round(wr, 4),
            "gross_pnl": round(tot_gross, 4),
            "net_pnl": round(tot_pnl, 4),
            "trade_expectancy": round(trade_exp, 5),
            "stream_expectancy": round(stream_exp, 5),
            "gross_trade_expectancy": round(gross_trade_exp, 5),
        }

    # 2. Accumulators for bootstrap distributions
    boot_stats = {
        p: {
            "n_trades": np.zeros(n_replicates),
            "win_rate": np.zeros(n_replicates),
            "net_pnl": np.zeros(n_replicates),
            "trade_expectancy": np.zeros(n_replicates),
            "stream_expectancy": np.zeros(n_replicates),
        }
        for p in policy_names
    }

    diff_stats: dict[str, dict[str, np.ndarray]] = {}
    for p in policy_names:
        if p != ctrl_name:
            diff_stats[f"{p}_minus_{ctrl_name}"] = {
                "d_win_rate": np.zeros(n_replicates),
                "d_net_pnl": np.zeros(n_replicates),
                "d_trade_expectancy": np.zeros(n_replicates),
                "d_stream_expectancy": np.zeros(n_replicates),
            }

    for b in range(n_replicates):
        sample_days = rng.choice(unique_days, size=n_days, replace=True)
        sample_row_idx = np.concatenate([day_indices[d] for d in sample_days])
        b_universe = len(sample_row_idx)

        b_targets = targets[sample_row_idx]
        b_pnls = net_pnls[sample_row_idx]

        for p in policy_names:
            m = policy_masks[p][sample_row_idx]
            n_tr = int(np.sum(m))
            boot_stats[p]["n_trades"][b] = n_tr
            if n_tr > 0:
                p_targets = b_targets[m]
                p_pnls = b_pnls[m]
                wr = np.sum(p_targets) / n_tr
                tot_pnl = np.sum(p_pnls)
                t_exp = tot_pnl / n_tr
            else:
                wr = 0.0
                tot_pnl = 0.0
                t_exp = 0.0

            s_exp = (tot_pnl / b_universe) if b_universe > 0 else 0.0

            boot_stats[p]["win_rate"][b] = wr
            boot_stats[p]["net_pnl"][b] = tot_pnl
            boot_stats[p]["trade_expectancy"][b] = t_exp
            boot_stats[p]["stream_expectancy"][b] = s_exp

        if ctrl_name in policy_names:
            c_wr = boot_stats[ctrl_name]["win_rate"][b]
            c_pnl = boot_stats[ctrl_name]["net_pnl"][b]
            c_t_exp = boot_stats[ctrl_name]["trade_expectancy"][b]
            c_s_exp = boot_stats[ctrl_name]["stream_expectancy"][b]
            for p in policy_names:
                if p != ctrl_name:
                    k = f"{p}_minus_{ctrl_name}"
                    diff_stats[k]["d_win_rate"][b] = boot_stats[p]["win_rate"][b] - c_wr
                    diff_stats[k]["d_net_pnl"][b] = boot_stats[p]["net_pnl"][b] - c_pnl
                    diff_stats[k]["d_trade_expectancy"][b] = boot_stats[p]["trade_expectancy"][b] - c_t_exp
                    diff_stats[k]["d_stream_expectancy"][b] = boot_stats[p]["stream_expectancy"][b] - c_s_exp

    results: dict[str, Any] = {
        "n_replicates": n_replicates,
        "n_days": int(n_days),
        "total_universe_trades": u_total_orig,
        "policies": {},
        "paired_differences": {},
    }

    def _ci(arr: np.ndarray) -> list[float]:
        return [round(float(np.percentile(arr, 2.5)), 5), round(float(np.percentile(arr, 97.5)), 5)]

    for p in policy_names:
        own_p_le_zero = float(np.mean(boot_stats[p]["net_pnl"] <= 0.0))
        results["policies"][p] = {
            "point_estimate": point_stats[p],
            "win_rate_mean": round(float(np.mean(boot_stats[p]["win_rate"])), 4),
            "win_rate_ci95": _ci(boot_stats[p]["win_rate"]),
            "net_pnl_mean": round(float(np.mean(boot_stats[p]["net_pnl"])), 2),
            "net_pnl_ci95": _ci(boot_stats[p]["net_pnl"]),
            "expectancy_mean": round(float(np.mean(boot_stats[p]["trade_expectancy"])), 5),
            "expectancy_ci95": _ci(boot_stats[p]["trade_expectancy"]),
            "trade_expectancy_mean": round(float(np.mean(boot_stats[p]["trade_expectancy"])), 5),
            "trade_expectancy_ci95": _ci(boot_stats[p]["trade_expectancy"]),
            "stream_expectancy_mean": round(float(np.mean(boot_stats[p]["stream_expectancy"])), 5),
            "stream_expectancy_ci95": _ci(boot_stats[p]["stream_expectancy"]),
            "p_value_own_profitability": round(own_p_le_zero, 4),
            "p_value_own_profitability_str": f"< {1.0/n_replicates:.4f}" if own_p_le_zero == 0 else f"{own_p_le_zero:.4f}",
        }

    raw_pvals_trade = {}
    raw_pvals_stream = {}

    ctrl_point = point_stats.get(ctrl_name, {})

    for k, d in diff_stats.items():
        pol_name = k.replace(f"_minus_{ctrl_name}", "")
        p_point = point_stats.get(pol_name, {})

        point_diff = {
            "d_win_rate": round(p_point.get("win_rate", 0.0) - ctrl_point.get("win_rate", 0.0), 4),
            "d_net_pnl": round(p_point.get("net_pnl", 0.0) - ctrl_point.get("net_pnl", 0.0), 2),
            "d_trade_expectancy": round(p_point.get("trade_expectancy", 0.0) - ctrl_point.get("trade_expectancy", 0.0), 5),
            "d_stream_expectancy": round(p_point.get("stream_expectancy", 0.0) - ctrl_point.get("stream_expectancy", 0.0), 5),
        }

        # Empirical p-values for H0: diff <= 0 (one-sided for superiority, two-sided)
        p_sup_trade = float(np.mean(d["d_trade_expectancy"] <= 0.0))
        p_two_trade = min(1.0, 2.0 * min(p_sup_trade, 1.0 - p_sup_trade))
        raw_pvals_trade[k] = p_two_trade

        p_sup_stream = float(np.mean(d["d_stream_expectancy"] <= 0.0))
        p_two_stream = min(1.0, 2.0 * min(p_sup_stream, 1.0 - p_sup_stream))
        raw_pvals_stream[k] = p_two_stream

        min_res_str = f"< {1.0/n_replicates:.4f}"

        results["paired_differences"][k] = {
            "point_estimate": point_diff,
            "d_win_rate_mean": round(float(np.mean(d["d_win_rate"])), 4),
            "d_win_rate_ci95": _ci(d["d_win_rate"]),
            "d_net_pnl_mean": round(float(np.mean(d["d_net_pnl"])), 2),
            "d_net_pnl_ci95": _ci(d["d_net_pnl"]),
            # Trade expectancy (selected trades)
            "d_expectancy_mean": round(float(np.mean(d["d_trade_expectancy"])), 5),
            "d_expectancy_ci95": _ci(d["d_trade_expectancy"]),
            "d_trade_expectancy_mean": round(float(np.mean(d["d_trade_expectancy"])), 5),
            "d_trade_expectancy_ci95": _ci(d["d_trade_expectancy"]),
            "trade_exp_nominal_p_value": round(p_two_trade, 4),
            "trade_exp_nominal_p_str": min_res_str if p_two_trade == 0 else f"{p_two_trade:.4f}",
            # Stream expectancy (full universe, 0 on skips)
            "d_stream_expectancy_mean": round(float(np.mean(d["d_stream_expectancy"])), 5),
            "d_stream_expectancy_ci95": _ci(d["d_stream_expectancy"]),
            "stream_exp_nominal_p_value": round(p_two_stream, 4),
            "stream_exp_nominal_p_str": min_res_str if p_two_stream == 0 else f"{p_two_stream:.4f}",
            # Backward compatibility
            "nominal_p_value": round(p_two_trade, 4),
            "nominal_p_str": min_res_str if p_two_trade == 0 else f"{p_two_trade:.4f}",
            "resolution_note": f"Bootstrap replicates B={n_replicates}. Minimum resolution is {1.0/n_replicates:.4f}. H0: diff <= 0.",
        }

    # Holm-Bonferroni correction over family of comparisons
    m_comps = len(raw_pvals_trade)
    sorted_comps_trade = sorted(raw_pvals_trade.items(), key=lambda x: x[1])
    for rank, (comp_name, p_val) in enumerate(sorted_comps_trade):
        adj_p = min(1.0, p_val * (m_comps - rank))
        results["paired_differences"][comp_name]["holm_adj_p_value"] = round(adj_p, 4)
        results["paired_differences"][comp_name]["trade_exp_holm_adj_p_value"] = round(adj_p, 4)
        results["paired_differences"][comp_name]["is_significant_05"] = bool(adj_p < 0.05)
        results["paired_differences"][comp_name]["trade_exp_is_significant_05"] = bool(adj_p < 0.05)

    sorted_comps_stream = sorted(raw_pvals_stream.items(), key=lambda x: x[1])
    for rank, (comp_name, p_val) in enumerate(sorted_comps_stream):
        adj_p = min(1.0, p_val * (m_comps - rank))
        results["paired_differences"][comp_name]["stream_exp_holm_adj_p_value"] = round(adj_p, 4)
        results["paired_differences"][comp_name]["stream_exp_is_significant_05"] = bool(adj_p < 0.05)

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
    - Side breakdown (Item 26: распределение выигрышей между активами и сторонами)
    """
    if len(df_trades) == 0:
        return {}

    total_net = float(df_trades["net_pnl_02pct"].sum())
    
    # Top 5 wins
    wins = df_trades[df_trades["target"] == 1].sort_values("net_pnl_02pct", ascending=False)
    top5 = wins.head(5)
    top5_sum = float(top5["net_pnl_02pct"].sum())
    total_wins_pnl = float(wins["net_pnl_02pct"].sum())
    top5_share_of_wins_pct = (top5_sum / total_wins_pnl * 100.0) if total_wins_pnl > 0 else 0.0
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

    # Asset breakdown (Item 26)
    by_asset = df_trades.groupby("asset").agg(
        trades=("market_id", "count"),
        wins=("target", "sum"),
        net_pnl=("net_pnl_02pct", "sum"),
        mean_ask=("ask", "mean"),
    ).reset_index()
    by_asset["win_rate"] = by_asset["wins"] / by_asset["trades"]
    total_all_wins = int(df_trades["target"].sum())
    by_asset["win_share_pct"] = (by_asset["wins"] / total_all_wins * 100.0) if total_all_wins > 0 else 0.0

    # Side breakdown (Item 26)
    by_side = df_trades.groupby("side").agg(
        trades=("market_id", "count"),
        wins=("target", "sum"),
        net_pnl=("net_pnl_02pct", "sum"),
        mean_ask=("ask", "mean"),
    ).reset_index()
    by_side["win_rate"] = by_side["wins"] / by_side["trades"]
    by_side["win_share_pct"] = (by_side["wins"] / total_all_wins * 100.0) if total_all_wins > 0 else 0.0

    return {
        "total_net_pnl": round(total_net, 4),
        "total_wins_pnl": round(total_wins_pnl, 4),
        "top5_wins_sum": round(top5_sum, 4),
        "top5_share_of_total_pct": round(top5_share_pct, 2),
        "top5_share_of_wins_pct": round(top5_share_of_wins_pct, 2),
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
        "by_side": by_side.to_dict("records"),
    }
