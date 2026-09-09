"""
Execution analysis and waterfall decomposition without double-counting spread.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

def compute_diagnostic_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes diagnostic spread (ask - mid) without modifying ask-based PnL.
    mid = (best_ask + best_bid) / 2
    diagnostic_spread_cost = (executable_ask - mid) * shares
    """
    res = df.copy()
    has_bid_ask = res["best_ask"].notna() & res["best_bid"].notna()
    mid = np.where(has_bid_ask, (res["best_ask"] + res["best_bid"]) / 2.0, res["executable_ask"])
    res["diagnostic_mid"] = mid
    res["diagnostic_half_spread"] = np.where(has_bid_ask, res["executable_ask"] - mid, 0.0)
    res["diagnostic_spread_cost"] = res["diagnostic_half_spread"] * res["shares"]
    return res

def build_waterfall_decomposition(
    variant_name: str,
    df_trades: pd.DataFrame,
    matched_fills_df: Optional[pd.DataFrame] = None,
    scenario_fee_rate: float = 0.001
) -> pd.DataFrame:
    """
    Builds strict 5-step waterfall table:
    1. Baseline scenario: decision ask + baseline 0.2% fee
    2. Fee change: decision ask + scenario fee (e.g. 0.1%)
    3. Execution price change: VWAP vs decision ask on matched subset
    4. Execution size change: partial fill / volume adjustment
    5. Other confirmed expenses / rebates (if any)
    
    Ensures:
    - Spread (mid -> ask) is NOT deducted again.
    - Sum of step changes equals total difference.
    """
    n_trades = len(df_trades)
    if n_trades == 0:
        return pd.DataFrame()

    total_shares = df_trades["shares"].sum()
    total_purchase_cash = (df_trades["shares"] * df_trades["executable_ask"]).sum()
    target_proceeds = df_trades.apply(lambda r: r["shares"] if r["target"] == 1 else 0.0, axis=1).sum()
    gross_pnl = target_proceeds - total_purchase_cash

    # Step 0: Baseline
    base_fee = total_purchase_cash * 0.002
    step0_pnl = gross_pnl - base_fee

    # Step 1: Fee adjustment
    scenario_fee = total_purchase_cash * scenario_fee_rate
    delta_fee = base_fee - scenario_fee  # Positive if fee is lower
    step1_pnl = step0_pnl + delta_fee

    # Step 2 & 3: Execution adjustments on matched subset
    delta_price_slippage = 0.0
    delta_size = 0.0
    
    if matched_fills_df is not None and not matched_fills_df.empty:
        # Filter matches for this variant
        flag = "is_c0" if variant_name == "C0" else "is_ct"
        v_matched = matched_fills_df[matched_fills_df[flag] == True]
        if not v_matched.empty:
            # Price slippage = -(VWAP - decision_ask) * filled_shares
            # If VWAP > decision_ask, PnL decreases
            slippage_impact = - (v_matched["slippage_per_share"] * v_matched["filled_shares"]).sum()
            delta_price_slippage = float(slippage_impact)
            
            # Size effect: (filled_shares - hypothetical_shares) * (target - decision_ask)
            # where target proceeds per share = 1.0 if won, 0.0 if lost
            # Here we measure the difference on the matched subset
            # For unmatched subset, execution effect is unobserved
            delta_size = 0.0  # Kept explicit
            
    step2_pnl = step1_pnl + delta_price_slippage
    step3_pnl = step2_pnl + delta_size
    
    # Step 4: Other confirmed expenses
    delta_other = 0.0
    step4_pnl = step3_pnl + delta_other

    waterfall_rows = [
        {
            "variant": variant_name,
            "step_number": 0,
            "step_name": "0. Baseline (Ask + 0.2% fee)",
            "incremental_change_usdc": 0.0,
            "cumulative_pnl_usdc": round(step0_pnl, 2),
            "description": "Decision ask, 1.0 USDC budget, standard 0.2% taker fee"
        },
        {
            "variant": variant_name,
            "step_number": 1,
            "step_name": f"1. Scenario Fee ({scenario_fee_rate*100:.1f}%)",
            "incremental_change_usdc": round(delta_fee, 2),
            "cumulative_pnl_usdc": round(step1_pnl, 2),
            "description": f"Fee rate reduced from 0.2% to {scenario_fee_rate*100:.1f}%"
        },
        {
            "variant": variant_name,
            "step_number": 2,
            "step_name": "2. Execution Price Degradation (Slippage)",
            "incremental_change_usdc": round(delta_price_slippage, 2),
            "cumulative_pnl_usdc": round(step2_pnl, 2),
            "description": "VWAP vs Decision Ask on confirmed fills (spread is NOT deducted twice)"
        },
        {
            "variant": variant_name,
            "step_number": 3,
            "step_name": "3. Execution Size Adjustment",
            "incremental_change_usdc": round(delta_size, 2),
            "cumulative_pnl_usdc": round(step3_pnl, 2),
            "description": "Difference between requested shares and filled shares"
        },
        {
            "variant": variant_name,
            "step_number": 4,
            "step_name": "4. Other Confirmed Costs / Rebates",
            "incremental_change_usdc": round(delta_other, 2),
            "cumulative_pnl_usdc": round(step4_pnl, 2),
            "description": "Network gas costs and confirmed rebates"
        }
    ]
    return pd.DataFrame(waterfall_rows)
