"""
Database loader and temporal candidate heuristic matcher for research opportunities and execution records.
Matches opportunities (buying YES) with fills in polyflip_db using market_id, side, and [-5, +120]s timing window.
Note: This is a proximity candidate heuristic, not a proven causal execution link.
"""
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from polyflip.research.trade_economics.pnl import calculate_fill_position

def load_execution_fills_from_db() -> pd.DataFrame:
    """
    Query polyflip_db for all execution requests, attempts, and fills.
    Returns DataFrame with columns:
    ['request_id', 'market_id', 'outcome_to_buy', 'requested_mode', 
     'request_created_at', 'state', 'fill_id', 'fill_price', 
     'fill_shares', 'fill_fee', 'fill_timestamp', 'gateway']
    """
    cmd = [
        "docker", "exec", "polyflip_db", "psql", 
        "-U", "polyflip", "-d", "polyflip", 
        "-t", "-A", "-F\t", "-c",
        """
        SELECT 
            r.id AS request_id, 
            r.market_id, 
            r.outcome_to_buy, 
            r.requested_mode, 
            r.created_at AS request_created_at,
            r.state,
            f.id AS fill_id,
            f.price AS fill_price,
            f.shares AS fill_shares,
            f.fee_usdc AS fill_fee,
            f.timestamp AS fill_timestamp,
            f.gateway
        FROM execution_requests r
        JOIN execution_attempts a ON r.id = a.request_id
        JOIN execution_fills f ON a.id = f.attempt_id
        ORDER BY r.market_id, r.created_at, f.timestamp;
        """
    ]
    try:
        raw_out = subprocess.check_output(cmd).decode("utf-8")
    except Exception:
        return pd.DataFrame()
        
    lines = [l.strip().split("\t") for l in raw_out.strip().split("\n") if l.strip()]
    if not lines or len(lines[0]) < 12:
        return pd.DataFrame()
        
    cols = [
        'request_id', 'market_id', 'outcome_to_buy', 'requested_mode', 
        'request_created_at', 'state', 'fill_id', 'fill_price', 
        'fill_shares', 'fill_fee', 'fill_timestamp', 'gateway'
    ]
    df = pd.DataFrame(lines, columns=cols)
    df['request_created_at'] = pd.to_datetime(df['request_created_at'], utc=True, format='mixed')
    df['fill_timestamp'] = pd.to_datetime(df['fill_timestamp'], utc=True, format='mixed')
    df['fill_price'] = df['fill_price'].astype(float)
    df['fill_shares'] = df['fill_shares'].astype(float)
    df['fill_fee'] = df['fill_fee'].astype(float)
    return df

def match_opportunities_with_fills(
    opportunities: List[Dict[str, Any]],
    df_db_fills: pd.DataFrame,
    causality_window_sec: float = 120.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs candidate matching between research opportunities (C0/CT buying YES)
    and execution records based on market, direction, and [-5, causality_window_sec] second window.
    
    Caveats & Methodological Limits:
    1. Proximity heuristic, not causal attribution: Direct links to specific research decisions
       are unrecorded in historical DB tables.
    2. Timing window [-5.0s, +causality_window_sec]: Admits requests created slightly before
       the recorded decision moment (e.g. -1.77s observed due to clock drift or early trigger).
    3. Over 60-105s latency, market prices drift substantially; requests may belong to
       other concurrent strategies/policies running in the bot.
    4. Ambiguous matches: multiple requests in window are flagged (is_ambiguous=True) but retained.
    5. actual_net_pnl assumes position held to contract outcome resolution, not verified cash redemptions.
    
    Returns:
      (matched_live_df, matched_paper_df, unmatched_df, reconciliation_summary)
    """
    matched_live_records = []
    matched_paper_records = []
    unmatched_records = []
    
    if not df_db_fills.empty:
        fills_by_market = {mid: group for mid, group in df_db_fills.groupby("market_id")}
    else:
        fills_by_market = {}

    unique_matched_requests = set()
    unique_matched_fills = set()

    for opp in opportunities:
        opp_id = opp.get("opportunity_id")
        mid = str(opp.get("market_id"))
        dec_at = pd.to_datetime(opp.get("decision_at"), utc=True)
        is_c0 = opp.get("variants", {}).get("C0", False)
        is_ct = opp.get("variants", {}).get("CT", False)
        
        target_outcome = "YES" # C0/CT buys YES
        
        m_fills = fills_by_market.get(mid)
        if m_fills is None or m_fills.empty:
            unmatched_records.append({
                "opportunity_id": opp_id,
                "market_id": mid,
                "is_c0": is_c0,
                "is_ct": is_ct,
                "decision_at": dec_at.isoformat(),
                "reason": "NO_REQUEST_FOR_MARKET",
                "detail": "No execution records found in database for this market"
            })
            continue

        direction_mask = m_fills["outcome_to_buy"] == target_outcome
        if not direction_mask.any():
            db_outcomes = list(m_fills["outcome_to_buy"].unique())
            unmatched_records.append({
                "opportunity_id": opp_id,
                "market_id": mid,
                "is_c0": is_c0,
                "is_ct": is_ct,
                "decision_at": dec_at.isoformat(),
                "reason": "DIRECTION_MISMATCH_NO",
                "detail": f"Market had requests for {db_outcomes}, but strategy bought {target_outcome}"
            })
            continue

        dir_fills = m_fills[direction_mask]
        t_diffs = (dir_fills["request_created_at"] - dec_at).dt.total_seconds()
        
        timing_mask = (t_diffs >= -5.0) & (t_diffs <= causality_window_sec)
        
        if not timing_mask.any():
            min_diff = t_diffs.min()
            max_diff = t_diffs.max()
            if max_diff < -5.0:
                reason = "TIMING_MISMATCH_PRIOR"
                detail = f"All YES requests were created before decision moment (min_dt={min_diff:.1f}s, max_dt={max_diff:.1f}s)"
            else:
                reason = "TIMING_MISMATCH_TOO_LATE"
                detail = f"All YES requests were created outside causal window (min_dt={min_diff:.1f}s, max_dt={max_diff:.1f}s)"
                
            unmatched_records.append({
                "opportunity_id": opp_id,
                "market_id": mid,
                "is_c0": is_c0,
                "is_ct": is_ct,
                "decision_at": dec_at.isoformat(),
                "reason": reason,
                "detail": detail
            })
            continue

        causal_fills = dir_fills[timing_mask]
        req_ids = causal_fills["request_id"].unique()
        is_ambiguous = len(req_ids) > 1
        
        live_fills = causal_fills[causal_fills["requested_mode"] == "LIVE"]
        paper_fills = causal_fills[causal_fills["requested_mode"] == "PAPER"]
        
        def build_matched_row(fills_subset: pd.DataFrame, mode: str) -> Dict[str, Any]:
            fills_list = [
                {"shares": r["fill_shares"], "price": r["fill_price"], "fee_usdc": r["fill_fee"]}
                for _, r in fills_subset.iterrows()
            ]
            agg = calculate_fill_position(fills_list)
            
            req_id = fills_subset["request_id"].iloc[0]
            req_created = fills_subset["request_created_at"].iloc[0]
            dt_sec = (req_created - dec_at).total_seconds()
            
            for fid in fills_subset["fill_id"]:
                unique_matched_fills.add(fid)
            unique_matched_requests.add(req_id)
            
            target = opp.get("target", 1 if opp.get("final_outcome") == "YES" else 0)
            settlement_proceeds = agg["filled_shares"] if target == 1 else 0.0
            actual_net_pnl = settlement_proceeds - agg["purchase_cash"] - agg["total_fee_usdc"]
            
            return {
                "opportunity_id": opp_id,
                "market_id": mid,
                "is_c0": is_c0,
                "is_ct": is_ct,
                "mode": mode,
                "gateway": fills_subset["gateway"].iloc[0],
                "decision_at": dec_at.isoformat(),
                "request_created_at": req_created.isoformat(),
                "latency_delta_sec": round(dt_sec, 3),
                "decision_ask": opp.get("executable_ask"),
                "hypothetical_shares": opp.get("shares"),
                "hypothetical_net_pnl": opp.get("net_pnl"),
                "target": target,
                "filled_shares": agg["filled_shares"],
                "purchase_cash": agg["purchase_cash"],
                "vwap": agg["vwap"],
                "fee_usdc": agg["total_fee_usdc"],
                "fill_count": agg["fill_count"],
                "slippage_per_share": agg["vwap"] - (opp.get("executable_ask") or 0.0),
                "slippage_cash": (agg["vwap"] - (opp.get("executable_ask") or 0.0)) * agg["filled_shares"],
                "actual_net_pnl": actual_net_pnl,
                "is_ambiguous": is_ambiguous
            }

        if not live_fills.empty:
            matched_live_records.append(build_matched_row(live_fills, "LIVE"))
        if not paper_fills.empty:
            matched_paper_records.append(build_matched_row(paper_fills, "PAPER"))

    matched_live_df = pd.DataFrame(matched_live_records)
    matched_paper_df = pd.DataFrame(matched_paper_records)
    unmatched_df = pd.DataFrame(unmatched_records)
    
    live_c0 = len(matched_live_df[matched_live_df["is_c0"] == True]) if not matched_live_df.empty else 0
    live_ct = len(matched_live_df[matched_live_df["is_ct"] == True]) if not matched_live_df.empty else 0
    paper_c0 = len(matched_paper_df[matched_paper_df["is_c0"] == True]) if not matched_paper_df.empty else 0
    paper_ct = len(matched_paper_df[matched_paper_df["is_ct"] == True]) if not matched_paper_df.empty else 0

    summary = {
        "matching_methodology": "temporal_candidate_heuristic_window_minus_5_plus_120s",
        "total_opportunities_evaluated": len(opportunities),
        "matched_live_count": len(matched_live_df),
        "matched_live_c0_count": live_c0,
        "matched_live_ct_count": live_ct,
        "applicable_live_coverage_c0_ct": 0,
        "matched_paper_count": len(matched_paper_df),
        "matched_paper_c0_count": paper_c0,
        "matched_paper_ct_count": paper_ct,
        "unmatched_count": len(unmatched_df),
        "unique_matched_requests": len(unique_matched_requests),
        "unique_matched_fills": len(unique_matched_fills),
        "unmatched_reason_counts": unmatched_df["reason"].value_counts().to_dict() if not unmatched_df.empty else {},
        "note": "Temporal proximity matching [-5, +120s] represents candidate association, not confirmed causal execution link."
    }
    
    return matched_live_df, matched_paper_df, unmatched_df, summary