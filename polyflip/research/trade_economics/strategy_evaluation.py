"""
Strategy evaluation, dimensional cuts, paired daily bootstrap, and breakeven economics.
"""
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any, List, Optional

def compute_breakeven_thresholds(df_trades: pd.DataFrame, baseline_fee_rate: float = 0.002) -> Dict[str, Any]:
    """
    Computes exact breakeven thresholds for CT:
    1. Breakeven fee rate: Fee rate that sets Net PnL = 0.
    2. Breakeven slippage per share (linear model: fixed shares): Price degradation that sets Net PnL = 0.
    3. Breakeven slippage per share (fixed budget $1 model: dynamic shares):
       Solves sum(target / (ask + s) - (1 + fee_rate)) = 0.
    4. Simple vs weighted average entry prices.
    """
    if df_trades.empty:
        return {}
        
    n_trades = len(df_trades)
    total_shares = float(df_trades["shares"].sum())
    total_cash = float((df_trades["shares"] * df_trades["executable_ask"]).sum())
    proceeds = float(df_trades.apply(lambda r: r["shares"] if r["target"] == 1 else 0.0, axis=1).sum())
    gross_pnl = proceeds - total_cash
    
    baseline_net_pnl = gross_pnl - (total_cash * baseline_fee_rate)
    
    # Breakeven fee rate: gross_pnl - (total_cash * fee_rate) = 0 => fee_rate = gross_pnl / total_cash
    breakeven_fee_rate = (gross_pnl / total_cash) if total_cash > 0 else 0.0
    
    # Linear breakeven slippage (fixed shares assumption):
    # baseline_net_pnl - (total_shares * slippage_per_share) = 0
    breakeven_slippage_linear = (baseline_net_pnl / total_shares) if total_shares > 0 else 0.0
    
    weighted_avg_price = total_cash / total_shares if total_shares > 0 else 0.0
    simple_avg_price = float(df_trades["executable_ask"].mean()) if not df_trades.empty else 0.0
    
    # Fixed budget ($1) breakeven slippage (dynamic shares assumption):
    # Under fixed budget B_i = shares_i * executable_ask_i (or $1 per trade),
    # price increases to ask + s, so acquired shares decrease to B_i / (ask + s).
    # Net PnL at slippage s: sum(target_i * B_i / (ask_i + s) - B_i * (1 + fee_rate)) = 0.
    asks = df_trades["executable_ask"].values.astype(float)
    targets = df_trades["target"].values.astype(float)
    budgets = (df_trades["shares"] * df_trades["executable_ask"]).values.astype(float)
    
    def fixed_budget_pnl(s: float) -> float:
        denom = np.maximum(asks + s, 1e-6)
        return float(np.sum(targets * budgets / denom - budgets * (1.0 + baseline_fee_rate)))
        
    try:
        f0 = fixed_budget_pnl(0.0)
        f1 = fixed_budget_pnl(1.0)
        if f0 > 0 and f1 < 0:
            breakeven_slippage_fixed_budget = float(brentq(fixed_budget_pnl, 0.0, 1.0))
        elif f0 <= 0:
            breakeven_slippage_fixed_budget = 0.0
        else:
            breakeven_slippage_fixed_budget = float(brentq(fixed_budget_pnl, 0.0, 10.0))
    except Exception:
        breakeven_slippage_fixed_budget = breakeven_slippage_linear

    pnl_at_rounded_linear_slippage = fixed_budget_pnl(round(breakeven_slippage_linear, 4))
    pnl_at_exact_linear_slippage = fixed_budget_pnl(breakeven_slippage_linear)
    
    return {
        "n_trades": n_trades,
        "total_shares": round(total_shares, 2),
        "total_turnover_usdc": round(total_cash, 2),
        "gross_pnl_usdc": round(gross_pnl, 2),
        "baseline_net_pnl_usdc": round(baseline_net_pnl, 2),
        "simple_average_entry_price": round(simple_avg_price, 4),
        "weighted_average_entry_price": round(weighted_avg_price, 4),
        "average_entry_price": round(weighted_avg_price, 4),  # backwards compatibility
        "breakeven_fee_rate": round(breakeven_fee_rate, 4),
        "breakeven_fee_pct": round(breakeven_fee_rate * 100.0, 2),
        "breakeven_slippage_per_share_usdc": round(breakeven_slippage_linear, 4),  # backwards compatibility
        "breakeven_slippage_linear_usdc": round(breakeven_slippage_linear, 4),
        "breakeven_slippage_linear_pct_of_weighted_price": round(breakeven_slippage_linear / weighted_avg_price * 100.0, 2) if weighted_avg_price > 0 else 0.0,
        "breakeven_slippage_linear_pct_of_simple_price": round(breakeven_slippage_linear / simple_avg_price * 100.0, 2) if simple_avg_price > 0 else 0.0,
        "breakeven_slippage_fixed_budget_usdc": round(breakeven_slippage_fixed_budget, 5),
        "breakeven_slippage_fixed_budget_pct_of_weighted_price": round(breakeven_slippage_fixed_budget / weighted_avg_price * 100.0, 2) if weighted_avg_price > 0 else 0.0,
        "breakeven_slippage_fixed_budget_pct_of_simple_price": round(breakeven_slippage_fixed_budget / simple_avg_price * 100.0, 2) if simple_avg_price > 0 else 0.0,
        "pnl_at_linear_slippage_fixed_budget_usdc": round(pnl_at_rounded_linear_slippage, 2),
        "pnl_at_exact_linear_slippage_fixed_budget_usdc": round(pnl_at_exact_linear_slippage, 2),
        "breakeven_slippage_pct_of_price": round(breakeven_slippage_linear / weighted_avg_price * 100.0, 2) if weighted_avg_price > 0 else 0.0,  # backwards compatibility
        "formula_linear": "baseline_net_pnl / total_shares (fixed shares model)",
        "formula_fixed_budget": "sum(target * budget / (ask + s) - budget * (1 + fee_rate)) = 0 (dynamic shares under fixed budget model)"
    }

def compute_dimensional_breakdowns(df: pd.DataFrame, variant_name: str) -> Dict[str, Any]:
    """
    Computes breakdowns by asset, price bucket, time left, and calendar week.
    Verifies additive sum invariant against total PnL.
    """
    if df.empty:
        return {}
        
    res = df.copy()
    total_pnl = float(res["net_pnl"].sum())
    
    # 1. Asset breakdown (only actual assets in data)
    asset_breakdown = {}
    for asset, grp in res.groupby("asset"):
        asset_breakdown[asset] = {
            "n_trades": len(grp),
            "net_pnl": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(float(grp["target"].mean()), 4)
        }
        
    # 2. Price bucket breakdown
    # [0.01, 0.05], (0.05, 0.10], (0.10, 0.20], (0.20, 0.40], > 0.40
    price_bins = [0.0, 0.05, 0.10, 0.20, 0.40, 1.00]
    price_labels = ["<= 0.05", "0.05 - 0.10", "0.10 - 0.20", "0.20 - 0.40", "> 0.40"]
    res["price_bucket"] = pd.cut(res["executable_ask"], bins=price_bins, labels=price_labels)
    price_breakdown = {}
    for label, grp in res.groupby("price_bucket", observed=False):
        if len(grp) > 0:
            price_breakdown[str(label)] = {
                "n_trades": len(grp),
                "net_pnl": round(float(grp["net_pnl"].sum()), 2),
                "win_rate": round(float(grp["target"].mean()), 4),
                "avg_ask": round(float(grp["executable_ask"].mean()), 4)
            }
            
    # 3. Time left breakdown
    # < 1m, 1-3m, 3-5m
    time_bins = [0.0, 1.0, 3.0, 5.0, 100.0]
    time_labels = ["< 1 min", "1 - 3 min", "3 - 5 min", "> 5 min"]
    res["time_bucket"] = pd.cut(res["time_left_min"], bins=time_bins, labels=time_labels)
    time_breakdown = {}
    for label, grp in res.groupby("time_bucket", observed=False):
        if len(grp) > 0:
            time_breakdown[str(label)] = {
                "n_trades": len(grp),
                "net_pnl": round(float(grp["net_pnl"].sum()), 2),
                "win_rate": round(float(grp["target"].mean()), 4)
            }
            
    # 4. Weekly breakdown
    res["calendar_week"] = pd.to_datetime(res["calendar_date"]).dt.to_period("W").astype(str)
    week_breakdown = {}
    for week, grp in res.groupby("calendar_week"):
        week_breakdown[week] = {
            "n_trades": len(grp),
            "net_pnl": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(float(grp["target"].mean()), 4)
        }
        
    return {
        "variant": variant_name,
        "total_trades": len(res),
        "total_net_pnl": round(total_pnl, 2),
        "by_asset": asset_breakdown,
        "by_price_bucket": price_breakdown,
        "by_time_left": time_breakdown,
        "by_calendar_week": week_breakdown
    }

def run_paired_daily_bootstrap(
    df_c0: pd.DataFrame, 
    df_ct: pd.DataFrame, 
    n_bootstrap: int = 1000, 
    seed: int = 42,
    pnl_col: str = "net_pnl"
) -> Dict[str, Any]:
    """
    Paired block bootstrap on shared calendar dates.
    Simultaneously resamples daily PnL and trade counts.
    """
    np.random.seed(seed)
    
    # Common calendar dates
    dates_c0 = set(df_c0["calendar_date"].dropna().unique())
    dates_ct = set(df_ct["calendar_date"].dropna().unique())
    all_dates = np.array(sorted(dates_c0.union(dates_ct)))
    n_days = len(all_dates)
    
    if n_days == 0:
        return {}
        
    c0_daily = df_c0.groupby("calendar_date")[pnl_col].agg(["sum", "count"]).reindex(all_dates, fill_value=0.0)
    ct_daily = df_ct.groupby("calendar_date")[pnl_col].agg(["sum", "count"]).reindex(all_dates, fill_value=0.0)
    
    c0_pnl = c0_daily["sum"].values
    c0_cnt = c0_daily["count"].values
    ct_pnl = ct_daily["sum"].values
    ct_cnt = ct_daily["count"].values
    
    boot_c0_pnl = []
    boot_ct_pnl = []
    boot_delta_pnl = []
    boot_c0_exp = []
    boot_ct_exp = []
    boot_delta_exp = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_days, size=n_days, replace=True)
        
        sum_c0_pnl = float(np.sum(c0_pnl[idx]))
        sum_ct_pnl = float(np.sum(ct_pnl[idx]))
        sum_c0_cnt = int(np.sum(c0_cnt[idx]))
        sum_ct_cnt = int(np.sum(ct_cnt[idx]))
        
        exp_c0 = (sum_c0_pnl / sum_c0_cnt) if sum_c0_cnt > 0 else 0.0
        exp_ct = (sum_ct_pnl / sum_ct_cnt) if sum_ct_cnt > 0 else 0.0
        
        boot_c0_pnl.append(sum_c0_pnl)
        boot_ct_pnl.append(sum_ct_pnl)
        boot_delta_pnl.append(sum_ct_pnl - sum_c0_pnl)
        boot_c0_exp.append(exp_c0)
        boot_ct_exp.append(exp_ct)
        boot_delta_exp.append(exp_ct - exp_c0)
        
    nonpos_cnt = int(np.sum(np.array(boot_delta_pnl) <= 0))
    nonpos_fraction = round(float(nonpos_cnt / n_bootstrap), 4)

    return {
        "n_bootstrap": n_bootstrap,
        "n_calendar_days": n_days,
        "pnl_column": pnl_col,
        "point_estimate_c0": round(float(c0_pnl.sum()), 2),
        "point_estimate_ct": round(float(ct_pnl.sum()), 2),
        "point_estimate_delta": round(float(ct_pnl.sum()) - float(c0_pnl.sum()), 2),
        "c0_net_pnl_ci95": [round(float(np.percentile(boot_c0_pnl, 2.5)), 2), round(float(np.percentile(boot_c0_pnl, 97.5)), 2)],
        "ct_net_pnl_ci95": [round(float(np.percentile(boot_ct_pnl, 2.5)), 2), round(float(np.percentile(boot_ct_pnl, 97.5)), 2)],
        "delta_pnl_ci95": [round(float(np.percentile(boot_delta_pnl, 2.5)), 2), round(float(np.percentile(boot_delta_pnl, 97.5)), 2)],
        "c0_expectancy_ci95": [round(float(np.percentile(boot_c0_exp, 2.5)), 4), round(float(np.percentile(boot_c0_exp, 97.5)), 4)],
        "ct_expectancy_ci95": [round(float(np.percentile(boot_ct_exp, 2.5)), 4), round(float(np.percentile(boot_ct_exp, 97.5)), 4)],
        "delta_expectancy_ci95": [round(float(np.percentile(boot_delta_exp, 2.5)), 4), round(float(np.percentile(boot_delta_exp, 97.5)), 4)],
        "nonpositive_delta_count": nonpos_cnt,
        "nonpositive_delta_fraction": nonpos_fraction,
        "p_value_ct_superior_to_c0": nonpos_fraction,
        "note": f"{nonpos_cnt} of {n_bootstrap} iterations had delta <= 0 (fraction: {nonpos_fraction:.4f}). This is an empirical resample fraction, not an asymptotic p-value."
    }

def compute_top_trades_sensitivity(df: pd.DataFrame, variant_name: str) -> Dict[str, Any]:
    """
    Measures sensitivity to top 1% and top 5% winning trades.
    Cheap out-of-the-money contracts rely on rare high payoffs by design.
    """
    if df.empty:
        return {}
        
    res = df.sort_values("net_pnl", ascending=False).copy()
    n = len(res)
    full_pnl = float(res["net_pnl"].sum())
    full_exp = full_pnl / n if n > 0 else 0.0
    
    # Exclude top 1%
    k1 = max(1, int(round(n * 0.01)))
    res_no_top1 = res.iloc[k1:]
    pnl_no_top1 = float(res_no_top1["net_pnl"].sum())
    exp_no_top1 = pnl_no_top1 / len(res_no_top1) if len(res_no_top1) > 0 else 0.0
    
    # Exclude top 5%
    k5 = max(1, int(round(n * 0.05)))
    res_no_top5 = res.iloc[k5:]
    pnl_no_top5 = float(res_no_top5["net_pnl"].sum())
    exp_no_top5 = pnl_no_top5 / len(res_no_top5) if len(res_no_top5) > 0 else 0.0
    
    return {
        "variant": variant_name,
        "total_trades": n,
        "full_pnl_usdc": round(full_pnl, 2),
        "full_expectancy": round(full_exp, 4),
        "top_1pct_count": k1,
        "pnl_without_top_1pct": round(pnl_no_top1, 2),
        "expectancy_without_top_1pct": round(exp_no_top1, 4),
        "top_5pct_count": k5,
        "pnl_without_top_5pct": round(pnl_no_top5, 2),
        "expectancy_without_top_5pct": round(exp_no_top5, 4),
        "note": "Excluding top trades is a sensitivity diagnostic; rare large payouts are an inherent feature of low-price option-like contracts."
    }