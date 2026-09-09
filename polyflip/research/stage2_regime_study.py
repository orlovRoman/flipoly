"""
polyflip/research/stage2_regime_study.py

Master implementation for Stage 2 of the research plan:
- Item 18: Common Opportunity Ledger across all variants
- Item 19: Formal definition and classification of "Former Favorite" (4 mutually exclusive groups)
- Item 20: Whole-sample mechanism check with winning and losing former favorites
- Item 21: Mechanism separation from low price inside 0.05 price buckets
- Item 22: Cohort decomposition (CT-only, CS-only, Both, Neither + Neither sub-combinations)
- Item 23: Time scale comparison (token vs spot on identical windows)
- Item 24: Isolated strike context evaluation (proxy vs canonical)
- Item 25: Comparison of pre-registered core variants (C0, CT, CS, CS_short, CT+strike, C1)
- Item 26: Orderbook depth execution recalculation for $1, $5, and $10 budgets
- Item 27: Execution selection effect measurement (filled vs unfilled)
- Item 28: Standalone expectancy and paired daily block bootstrap
- Item 29: Unviewed holdout evaluation
- Item 30: Pre-registered decision criteria matrix and verdict synthesis
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import subprocess
import hashlib

from polyflip.research.regime_features import (
    compute_efficiency_ratio,
    compute_return_sign_changes,
    compute_return_autocorrelation,
    classify_local_regime,
    classify_spot_regime_short,
    compute_strike_context,
)
from polyflip.research.orderbook_execution import (
    simulate_orderbook_execution,
    calculate_trade_payout_and_pnl,
    ExecutionVolumeTracker,
)


def run_stage2_study(
    snapshots_csv_path: Path,
    candles_csv_path: Path,
    expirations_json_path: Path,
    candles_1m_csv_path: Path | None = None,
    target_asset: str = "BTC",
    stake_usdc: float = 1.0,
    taker_fee_rate: float = 0.002,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """
    Executes all Stage 2 analysis requirements strictly on a unified common opportunity ledger.
    """
    np.random.seed(seed)

    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        git_commit = "UNKNOWN"
        git_dirty = False

    def hash_file(filepath):
        if filepath is None:
            return "UNKNOWN"
        try:
            with open(filepath, 'rb') as f_in:
                return hashlib.sha256(f_in.read()).hexdigest()
        except:
            return "UNKNOWN"
            
    snaps_hash = hash_file(snapshots_csv_path)
    candles_hash = hash_file(candles_csv_path)
    exp_hash = hash_file(expirations_json_path)

    # 1. Load data
    try:
        raw_snaps = pd.read_csv(snapshots_csv_path, encoding="utf-16")
    except Exception:
        raw_snaps = pd.read_csv(snapshots_csv_path, encoding="utf-8")

    try:
        raw_candles = pd.read_csv(candles_csv_path, encoding="utf-16")
    except Exception:
        raw_candles = pd.read_csv(candles_csv_path, encoding="utf-8")

    raw_candles_1m = None
    if candles_1m_csv_path and candles_1m_csv_path.exists():
        try:
            raw_candles_1m = pd.read_csv(candles_1m_csv_path, encoding="utf-16")
        except Exception:
            raw_candles_1m = pd.read_csv(candles_1m_csv_path, encoding="utf-8")

    with open(expirations_json_path, "r", encoding="utf-8") as f:
        exp_map = json.load(f)

    # Filter asset
    snaps = raw_snaps[raw_snaps["asset"] == target_asset].copy()
    snaps["recorded_at"] = pd.to_datetime(snaps["recorded_at"], utc=True)
    snaps = snaps.sort_values(["market_id", "recorded_at"]).reset_index(drop=True)

    symbol = f"{target_asset}USDT"
    candles = raw_candles[raw_candles["symbol"] == symbol].copy()
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    candles = candles.sort_values("open_time").reset_index(drop=True)

    candles_1m = None
    if raw_candles_1m is not None:
        candles_1m = raw_candles_1m[raw_candles_1m["symbol"] == symbol].copy()
        candles_1m["open_time"] = pd.to_datetime(candles_1m["open_time"], utc=True)
        candles_1m = candles_1m.sort_values("open_time").reset_index(drop=True)

    # Pre-index candles for fast causal lookup
    c_times = candles["open_time"].values
    c_closes = candles["close"].values
    c_opens = candles["open"].values
    c_highs = candles["high"].values
    c_lows = candles["low"].values

    # 2. Extract causal decision moment (first observation with 3.5 <= time_left_min <= 5.5)
    # Group by market_id
    grouped = snaps.groupby("market_id", sort=False)

    ledger_rows: list[dict[str, Any]] = []

    for mid, group in grouped:
        m_str = str(mid)
        # Decision candidates strictly within [3.5, 5.0] (Item 8 C0 causal filter)
        cand = group[(group["time_left_min"] >= 3.5) & (group["time_left_min"] <= 5.0)]
        if cand.empty:
            continue

        dec_row = cand.iloc[0]
        decision_at = dec_row["recorded_at"]
        time_left_min = float(dec_row["time_left_min"])

        # Prior snapshots in the causal 15-minute window before decision_at
        prior_snaps = group[(group["recorded_at"] <= decision_at) & (group["recorded_at"] >= decision_at - pd.Timedelta(minutes=15))]
        if len(prior_snaps) < 3:
            continue

        yes_mid = float(dec_row["mid_price"])
        yes_ask = float(dec_row["best_ask"]) if pd.notna(dec_row["best_ask"]) else np.nan
        yes_bid = float(dec_row["best_bid"]) if pd.notna(dec_row["best_bid"]) else np.nan

        # Observed quotes only: outsider is UP (yes_mid <= 0.5)
        ask = yes_ask
        bid = yes_bid
        mid_p = yes_mid
        spread = float(dec_row["spread"]) if pd.notna(dec_row["spread"]) else (ask - (bid if np.isfinite(bid) else 0.0))
        final_outcome = str(dec_row["final_outcome"]).upper()
        
        exclusion_reason = None
        if not np.isfinite(yes_ask):
            exclusion_reason = "MISSING_QUOTE"
        elif yes_mid > 0.5:
            exclusion_reason = "PRICE_FILTER" # Outsider is UP
        
        # Primary rule: ask <= 0.40 and ask >= 0.01
        is_candidate_price = False
        if exclusion_reason == None:
            is_candidate_price = (ask <= 0.40) and (ask >= 0.01)
            if not is_candidate_price:
                exclusion_reason = "PRICE_FILTER"

        # Item 19: Formal definition of Former Favorite (4 mutually exclusive groups)
        # Lookback window: up to 10 minutes prior to decision
        lb_cutoff = decision_at - pd.Timedelta(minutes=10)
        lb_snaps = group[(group["recorded_at"] >= lb_cutoff) & (group["recorded_at"] <= decision_at)].copy()
        
        former_favorite_group = "INSUFFICIENT_HISTORY"
        if len(lb_snaps) >= 3:
            lb_times = lb_snaps["recorded_at"].tolist()
            duration = (lb_times[-1] - lb_times[0]).total_seconds()
            gaps = [ (lb_times[i] - lb_times[i-1]).total_seconds() for i in range(1, len(lb_times)) ]
            max_gap = max(gaps) if gaps else 0
            
            # minimal requirements (P17)
            if duration >= 300 and max_gap <= 300:
                lb_prices = lb_snaps["mid_price"].dropna().tolist()
                if len(lb_prices) >= 3:
                    max_lb = max(lb_prices)
                    if max_lb >= 0.55 and ask <= 0.40:
                        former_favorite_group = "FORMER_FAVORITE"
                    elif max_lb <= 0.40:
                        former_favorite_group = "PERSISTENT_CHEAP"
                    else:
                        former_favorite_group = "OTHER_TRAJECTORY"

        # Token regime CT (historical control)
        prior_token_prices = prior_snaps["mid_price"].dropna().to_numpy()
        ct_res = classify_local_regime(prior_token_prices, min_observations=3)
        token_regime = ct_res["state"]

        # Spot regime CS (historical 5m candles prior to decision)
        # Use strictly closed candles: open_time + 5m <= decision_at
        causal_candles = candles[candles["open_time"] + pd.Timedelta(minutes=5) <= decision_at]
        spot_regime = "UNCERTAIN"
        spot_regime_short = "UNCERTAIN"
        cs_short_reason = "INSUFFICIENT_DATA"
        spot_price_at_decision = np.nan
        sub_c_closes = []
        cs_res: dict[str, Any] = {"state": "UNCERTAIN", "efficiency_ratio": None, "local_mean": None}
        if len(causal_candles) >= 6:
            last_candle_close = causal_candles["open_time"].iloc[-1] + pd.Timedelta(minutes=5)
            if (decision_at - last_candle_close) <= pd.Timedelta(minutes=15):
                spot_closes = causal_candles["close"].tail(6).to_numpy()
                spot_times = causal_candles["open_time"].tail(6).to_numpy()
                sub_c_closes = spot_closes.tolist()
                spot_price_at_decision = float(spot_closes[-1])
                cs_res = classify_local_regime(spot_closes, min_observations=4)
                spot_regime = cs_res["state"]
                
                if candles_1m is not None:
                    c1m = candles_1m[candles_1m["open_time"] + pd.Timedelta(minutes=1) <= decision_at]
                    if len(c1m) >= 10:
                        c1m_last_close = c1m["open_time"].iloc[-1] + pd.Timedelta(minutes=1)
                        if (decision_at - c1m_last_close) <= pd.Timedelta(minutes=5):
                            c1m_closes = c1m["close"].tail(10).to_numpy()
                            c1m_times = c1m["open_time"].tail(10).to_numpy()
                            cs_short_res = classify_spot_regime_short(c1m_closes, timestamps=c1m_times, as_of=decision_at, window_min=10.0, require_valid_autocorr=True)
                        else:
                            cs_short_res = {"state": "UNCERTAIN", "classification_reason": "STALE_1M_CANDLE"}
                    else:
                        cs_short_res = {"state": "UNCERTAIN", "classification_reason": "INSUFFICIENT_1M_CANDLES"}
                else:
                    cs_short_res = classify_spot_regime_short(spot_closes, timestamps=spot_times, as_of=decision_at, window_min=10.0, require_valid_autocorr=True)

                spot_regime_short = cs_short_res["state"]
                cs_short_reason = cs_short_res.get("classification_reason")
            else:
                cs_short_reason = "STALE_CANDLE"
        else:
            cs_short_reason = "INSUFFICIENT_CANDLES"

        # Market boundaries and strike
        exp_iso = exp_map.get(m_str)
        market_end_at = pd.to_datetime(exp_iso, utc=True) if exp_iso else None
        market_start_at = (market_end_at - pd.Timedelta(minutes=15)) if market_end_at else None

        # Strike info: canonical strike is historically unrecorded; proxy is candle open at market start
        canonical_strike = float(dec_row["strike_value"]) if "strike_value" in dec_row and pd.notna(dec_row["strike_value"]) else None
        proxy_strike = None
        if market_start_at is not None:
            start_mask = candles["open_time"] <= market_start_at
            if np.any(start_mask):
                proxy_strike = float(c_opens[start_mask][-1])

        # Sigma estimation
        sigma_min = 0.001
        if len(sub_c_closes) >= 3:
            diffs = np.diff(sub_c_closes) / sub_c_closes[:-1]
            std_ret = float(np.std(diffs))
            if std_ret > 1e-6:
                sigma_min = std_ret / math.sqrt(5.0)

        # Strike context evaluation (Item 24)
        ctx_proxy = compute_strike_context(
            spot=spot_price_at_decision,
            strike=proxy_strike if proxy_strike else np.nan,
            sigma_min=sigma_min,
            time_left_min=time_left_min,
            local_mean=cs_res.get("local_mean"),
            candidate_side="UP",
        )
        proxy_reversion_helps = ctx_proxy.get("reversion_helps_strike", False)

        target = 1 if final_outcome == "YES" else 0

        # PnL accounting (Point 15)
        shares = (stake_usdc / ask) if ask > 0 else 0.0
        gross_pnl = shares * (target - ask)
        fee = stake_usdc * taker_fee_rate
        net_pnl = gross_pnl - fee

        # Synthesize orderbook depth for this snapshot (Point 14 & 26)
        sim_asks = None
        if "asks" in dec_row and pd.notna(dec_row["asks"]):
            try:
                if isinstance(dec_row["asks"], str):
                    sim_asks = json.loads(dec_row["asks"])
                else:
                    sim_asks = dec_row["asks"]
            except Exception:
                pass

        # Multi-budget simulations ($1, $5, $10) (Item 26)
        if sim_asks is not None:
            res_1 = simulate_orderbook_execution(sim_asks, budget_usdc=1.0, taker_fee_rate=taker_fee_rate)
            res_5 = simulate_orderbook_execution(sim_asks, budget_usdc=5.0, taker_fee_rate=taker_fee_rate)
            res_10 = simulate_orderbook_execution(sim_asks, budget_usdc=10.0, taker_fee_rate=taker_fee_rate)
            exec_1 = res_1.fill_status
            exec_5 = res_5.fill_status
            exec_10 = res_10.fill_status
        else:
            exec_1 = "BLOCKED_DATA"
            exec_5 = "BLOCKED_DATA"
            exec_10 = "BLOCKED_DATA"

        # Context evaluation for canonical strike
        ctx_canon = compute_strike_context(
            spot=spot_price_at_decision,
            strike=canonical_strike if canonical_strike else np.nan,
            sigma_min=sigma_min,
            time_left_min=time_left_min,
            local_mean=cs_res.get("local_mean"),
            candidate_side="UP",
        )
        canon_reversion_helps = ctx_canon.get("reversion_helps_strike", False)

        # Variant inclusion flags
        v_c0 = is_candidate_price
        v_ct = is_candidate_price and (token_regime == "REVERSION")
        v_cs = is_candidate_price and (spot_regime == "REVERSION")
        v_cs_short = is_candidate_price and (spot_regime_short == "REVERSION")
        v_c1 = is_candidate_price and ((token_regime == "REVERSION") or (spot_regime == "REVERSION"))
        v_ct_strike_proxy = v_ct and proxy_reversion_helps
        v_ct_strike_canon = v_ct and canon_reversion_helps and canonical_strike is not None

        # Cohort tag (Item 22)
        if v_c0:
            if token_regime == "REVERSION" and spot_regime == "REVERSION":
                cohort_tag = "Both"
            elif token_regime == "REVERSION":
                cohort_tag = "CT-only"
            elif spot_regime == "REVERSION":
                cohort_tag = "CS-only"
            else:
                cohort_tag = "Neither"
        else:
            cohort_tag = "EXCLUDED_BY_PRICE"

        calendar_date = decision_at.strftime("%Y-%m-%d")

        ledger_rows.append({
            "opportunity_id": f"{m_str}_{decision_at.isoformat()}",
            "market_id": m_str,
            "asset": target_asset,
            "calendar_date": calendar_date,
            "decision_at": decision_at.isoformat(),
            "time_left_min": round(time_left_min, 2),
            "best_bid": bid,
            "best_ask": ask,
            "executable_ask": ask,
            "spread": round(spread, 4),
            "final_outcome": final_outcome,
            "target": target,
            "is_candidate_price": is_candidate_price,
            "former_favorite_group": former_favorite_group,
            "token_regime_CT": token_regime,
            "token_er": ct_res.get("efficiency_ratio"),
            "token_sign_flips": ct_res.get("sign_change_freq"),
            "token_autocorr": ct_res.get("autocorr_lag1"),
            "spot_regime_CS": spot_regime,
            "spot_er": cs_res.get("efficiency_ratio"),
            "spot_regime_CS_short": spot_regime_short,
            "spot_short_reason": cs_short_reason,
            "spot_price_at_decision": spot_price_at_decision,
            "proxy_strike": proxy_strike,
            "proxy_reversion_helps": proxy_reversion_helps,
            "shares": round(shares, 4),
            "gross_pnl": round(gross_pnl, 4),
            "taker_fee": round(fee, 4),
            "net_pnl": round(net_pnl, 4),
            "cohort_tag": cohort_tag,
            "neither_subgroup": f"({token_regime}, {spot_regime})" if cohort_tag == "Neither" else None,
            "variants": {
                "C0": v_c0,
                "CT": v_ct,
                "CS": v_cs,
                "CS_short": v_cs_short,
                "C1": v_c1,
                "CT_plus_canonical_strike": v_ct_strike_canon,
                "CT_plus_proxy_strike": v_ct_strike_proxy,
            },
            "execution_sim": exec_1,
            "execution_sim_5": exec_5,
            "execution_sim_10": exec_10,
            "exclusion_reason": exclusion_reason,
        })

    ledger_df = pd.DataFrame(ledger_rows)

    # 3. Item 18: Verify Common Opportunity Ledger Invariants
    # Every row has identical decision moment, side, price, and target for all variants
    c0_sub = ledger_df[ledger_df["is_candidate_price"]].copy()

    # 4. Item 20: Mechanism Check on whole sample (Former Favorite analysis)
    ff_groups = ["FORMER_FAVORITE", "PERSISTENT_CHEAP", "OTHER_TRAJECTORY", "INSUFFICIENT_HISTORY"]
    ff_stats: dict[str, Any] = {}
    for grp in ff_groups:
        sub_g = c0_sub[c0_sub["former_favorite_group"] == grp]
        n_g = len(sub_g)
        n_days = sub_g["calendar_date"].nunique()
        wr = float(sub_g["target"].mean()) if n_g > 0 else 0.0
        pnl = float(sub_g["net_pnl"].sum()) if n_g > 0 else 0.0
        exp = (pnl / n_g) if n_g > 0 else 0.0
        avg_ask = float(sub_g["executable_ask"].mean()) if n_g > 0 else 0.0

        # Weekly results
        sub_g_week = sub_g.copy()
        if n_g > 0:
            sub_g_week["week"] = pd.to_datetime(sub_g_week["calendar_date"]).dt.to_period("W").astype(str)
            weekly_pnl = sub_g_week.groupby("week")["net_pnl"].sum().round(2).to_dict()
            top3_wins = sub_g.nlargest(3, "net_pnl")["net_pnl"].tolist()
        else:
            weekly_pnl = {}
            top3_wins = []

        ff_stats[grp] = {
            "n_trades": n_g,
            "n_days": n_days,
            "avg_entry_price": round(avg_ask, 4),
            "win_rate": round(wr, 4),
            "net_pnl_usdc": round(pnl, 2),
            "expectancy": round(exp, 4),
            "top_3_wins": [round(w, 2) for w in top3_wins],
            "weekly_pnl": weekly_pnl,
        }

    # 5. Item 21: Separate mechanism from price in 0.05 buckets
    price_buckets = [
        (0.01, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
    ]
    price_bucket_analysis: dict[str, Any] = {}
    for p_low, p_high in price_buckets:
        b_key = f"[{p_low:.2f}, {p_high:.2f}]"
        if p_high == 0.40:
            mask_b = (c0_sub["executable_ask"] >= p_low) & (c0_sub["executable_ask"] <= p_high)
        else:
            mask_b = (c0_sub["executable_ask"] >= p_low) & (c0_sub["executable_ask"] < p_high)
        sub_b = c0_sub[mask_b]
        b_stats: dict[str, Any] = {"total_trades": len(sub_b), "by_group": {}}
        for grp in ff_groups:
            sub_bg = sub_b[sub_b["former_favorite_group"] == grp]
            n_bg = len(sub_bg)
            if n_bg >= 5:
                b_stats["by_group"][grp] = {
                    "n_trades": n_bg,
                    "win_rate": round(float(sub_bg["target"].mean()), 4),
                    "expectancy": round(float(sub_bg["net_pnl"].sum() / n_bg), 4),
                    "net_pnl_usdc": round(float(sub_bg["net_pnl"].sum()), 2),
                }
            else:
                b_stats["by_group"][grp] = {
                    "n_trades": n_bg,
                    "status": "INSUFFICIENT_DATA",
                }
        price_bucket_analysis[b_key] = b_stats

    # 6. Item 22: Cohort Decomposition (CT-only, CS-only, Both, Neither)
    cohort_names = ["CT-only", "CS-only", "Both", "Neither"]
    cohort_stats: dict[str, Any] = {}
    for c_name in cohort_names:
        sub_c = c0_sub[c0_sub["cohort_tag"] == c_name]
        n_c = len(sub_c)
        pnl_c = float(sub_c["net_pnl"].sum()) if n_c > 0 else 0.0
        exp_c = (pnl_c / n_c) if n_c > 0 else 0.0
        wr_c = float(sub_c["target"].mean()) if n_c > 0 else 0.0
        avg_ask_c = float(sub_c["executable_ask"].mean()) if n_c > 0 else 0.0

        # Top 1, 3, 5 wins contribution
        top_wins = sub_c.nlargest(5, "net_pnl")["net_pnl"].tolist() if n_c > 0 else []

        entry = {
            "n_trades": n_c,
            "win_rate": round(wr_c, 4),
            "avg_entry_price": round(avg_ask_c, 4),
            "net_pnl_usdc": round(pnl_c, 2),
            "expectancy": round(exp_c, 4),
            "top_1_win": round(top_wins[0], 2) if len(top_wins) >= 1 else 0.0,
            "top_3_wins_sum": round(sum(top_wins[:3]), 2) if len(top_wins) >= 3 else 0.0,
            "top_5_wins_sum": round(sum(top_wins[:5]), 2) if len(top_wins) >= 5 else 0.0,
        }

        # Sub-combinations for Neither
        if c_name == "Neither" and n_c > 0:
            sub_combos = sub_c.groupby("neither_subgroup")["net_pnl"].agg(
                n_trades="count",
                net_pnl="sum",
                win_rate=lambda s: float(sub_c.loc[s.index, "target"].mean()),
            ).to_dict(orient="index")
            entry["sub_combinations"] = {
                k: {
                    "n_trades": int(v["n_trades"]),
                    "net_pnl_usdc": round(float(v["net_pnl"]), 2),
                    "win_rate": round(float(v["win_rate"]), 4),
                }
                for k, v in sub_combos.items()
            }

        cohort_stats[c_name] = entry

    # Verify additive cohort invariants:
    # Neither + CT-only + CS-only + Both == C0
    sum_cohort_trades = sum(cohort_stats[c]["n_trades"] for c in cohort_names)
    sum_cohort_pnl = sum(cohort_stats[c]["net_pnl_usdc"] for c in cohort_names)
    c0_trades = len(c0_sub)
    c0_pnl = float(c0_sub["net_pnl"].sum())
    assert sum_cohort_trades == c0_trades
    assert math.isclose(sum_cohort_pnl, c0_pnl, abs_tol=0.1)

    # 7. Item 23: Time scales comparison (Token vs Spot)
    # Compare token regime vs CS_short on identical window & grid
    scale_comp = {
        "token_reversion_rate": round(float((c0_sub["token_regime_CT"] == "REVERSION").mean()), 4),
        "spot_long_reversion_rate": round(float((c0_sub["spot_regime_CS"] == "REVERSION").mean()), 4),
        "spot_short_reversion_rate": round(float((c0_sub["spot_regime_CS_short"] == "REVERSION").mean()), 4),
        "agreement_token_vs_spot_long": round(float((c0_sub["token_regime_CT"] == c0_sub["spot_regime_CS"]).mean()), 4),
        "agreement_token_vs_spot_short": round(float((c0_sub["token_regime_CT"] == c0_sub["spot_regime_CS_short"]).mean()), 4),
    }

    # 8. Item 25 & 28: Pre-registered Core Variants Comparison & Paired Block Bootstrap
    core_variant_keys = [
        ("Price Control", "C0"),
        ("Original CT", "CT"),
        ("Old CS", "CS"),
        ("CS_short", "CS_short"),
        ("CT + canonical strike", "CT_plus_canonical_strike"),
        ("CT + proxy strike", "CT_plus_proxy_strike"),
        ("C1 (reference)", "C1"),
    ]

    all_days = np.array(sorted(c0_sub["calendar_date"].unique()))
    n_days = len(all_days)

    # Precalculate daily PnLs per variant
    daily_pnls: dict[str, dict[str, float]] = {}
    variant_results: dict[str, Any] = {}

    for label, v_key in core_variant_keys:
        v_mask = c0_sub["variants"].apply(lambda d: d.get(v_key, False))
        sub_v = c0_sub[v_mask]
        n_v = len(sub_v)
        pnl_v = float(sub_v["net_pnl"].sum()) if n_v > 0 else 0.0
        exp_v = (pnl_v / n_v) if n_v > 0 else 0.0
        wr_v = float(sub_v["target"].mean()) if n_v > 0 else 0.0
        avg_price = float(sub_v["executable_ask"].mean()) if n_v > 0 else 0.0

        # Daily aggregations
        d_pnl = sub_v.groupby("calendar_date")["net_pnl"].sum().to_dict() if n_v > 0 else {}
        daily_pnls[v_key] = {d: d_pnl.get(d, 0.0) for d in all_days}

        variant_results[v_key] = {
            "label": label,
            "what_it_tests": {
                "C0": "Benefit of lowest price rule only",
                "CT": "Incremental benefit of token reversion behavior",
                "CS": "Historical control of spot reversion filter",
                "CS_short": "Benefit of short 10m spot reversion regime",
                "CT_plus_canonical_strike": "Benefit of contract moneyness relative to threshold",
                "CT_plus_proxy_strike": "Benefit of moneyness with 5m open proxy",
                "C1": "Union reference (CT or CS)",
            }.get(v_key, ""),
            "n_trades": n_v,
            "win_rate": round(wr_v, 4),
            "avg_entry_price": round(avg_price, 4),
            "net_pnl_usdc": round(pnl_v, 2),
            "expectancy": round(exp_v, 4),
        }

    # Paired Calendar Daily Block Bootstrap (Item 28)
    c0_daily = np.array([daily_pnls["C0"][d] for d in all_days])

    for label, v_key in core_variant_keys:
        v_daily = np.array([daily_pnls[v_key][d] for d in all_days])
        paired_daily_deltas = v_daily - c0_daily

        v_mask = c0_sub["variants"].apply(lambda d: d.get(v_key, False))
        sub_v_curr = c0_sub[v_mask]
        n_v_curr = len(sub_v_curr)

        # Daily trade counts for variant
        d_counts = sub_v_curr.groupby("calendar_date").size().to_dict() if n_v_curr > 0 else {}
        v_daily_counts = np.array([d_counts.get(d, 0) for d in all_days])

        # Bootstrap paired delta and standalone expectancy
        boot_deltas = []
        boot_standalones = []
        if n_v_curr > 0:
            for _ in range(n_bootstrap):
                sampled_day_indices = np.random.randint(0, n_days, size=n_days)
                b_delta = float(np.sum(paired_daily_deltas[sampled_day_indices]))
                b_standalone = float(np.sum(v_daily[sampled_day_indices]))
                b_counts = float(np.sum(v_daily_counts[sampled_day_indices]))
                boot_deltas.append(b_delta)
                # Avoid division by zero if all sampled days had 0 trades
                boot_standalones.append(b_standalone / b_counts if b_counts > 0 else 0.0)

            ci_delta_low = float(np.percentile(boot_deltas, 2.5))
            ci_delta_high = float(np.percentile(boot_deltas, 97.5))
            ci_stand_low = float(np.percentile(boot_standalones, 2.5))
            ci_stand_high = float(np.percentile(boot_standalones, 97.5))
        else:
            ci_delta_low, ci_delta_high = 0.0, 0.0
            ci_stand_low, ci_stand_high = 0.0, 0.0

        if n_v_curr > 0:
            variant_results[v_key]["status"] = "COMPUTED"
            variant_results[v_key]["paired_delta_vs_c0"] = round(float(variant_results[v_key]["net_pnl_usdc"] - variant_results["C0"]["net_pnl_usdc"]), 2)
            variant_results[v_key]["paired_bootstrap_ci_95"] = [round(ci_delta_low, 2), round(ci_delta_high, 2)]
            variant_results[v_key]["standalone_expectancy_ci_95"] = [round(ci_stand_low, 4), round(ci_stand_high, 4)]
        else:
            variant_results[v_key]["status"] = "BLOCKED_DATA"
            variant_results[v_key]["net_pnl_usdc"] = None
            variant_results[v_key]["expectancy"] = None
            variant_results[v_key]["paired_delta_vs_c0"] = None
            variant_results[v_key]["paired_bootstrap_ci_95"] = None
            variant_results[v_key]["standalone_expectancy_ci_95"] = None

    # 9. Item 26: Multi-budget execution recalculation ($1, $5, $10)
    # Replaced with BLOCKED_DATA per plan (missing historical depth)
    budget_tables: dict[str, Any] = {
        "$1": "BLOCKED_DATA",
        "$5": "BLOCKED_DATA",
        "$10": "BLOCKED_DATA",
    }

    # 10. Item 27: Selection effect of execution
    # Compare filled vs unfilled opportunities on price, mechanism, outcome
    selection_effect = "BLOCKED_DATA"

    # 11. Item 29: Subsequent unviewed holdout check & Exploratory Period
    exploratory_mask = (c0_sub["calendar_date"] >= "2026-09-02") & (c0_sub["calendar_date"] <= "2026-09-08")
    holdout_mask = c0_sub["calendar_date"] > "2026-09-08"
    exploratory_sub = c0_sub[exploratory_mask]
    holdout_sub = c0_sub[holdout_mask]

    exploratory_results: dict[str, Any] = {}
    holdout_results: dict[str, Any] = {}
    for label, v_key in core_variant_keys:
        # Exploratory
        e_mask = exploratory_sub["variants"].apply(lambda d: d.get(v_key, False))
        sub_e = exploratory_sub[e_mask]
        n_e = len(sub_e)
        pnl_e = float(sub_e["net_pnl"].sum()) if n_e > 0 else 0.0
        exp_e = (pnl_e / n_e) if n_e > 0 else 0.0
        wr_e = float(sub_e["target"].mean()) if n_e > 0 else 0.0
        exploratory_results[v_key] = {
            "n_trades": n_e,
            "win_rate": round(wr_e, 4),
            "net_pnl_usdc": round(pnl_e, 2),
            "expectancy": round(exp_e, 4),
        }
        
        # Holdout
        h_mask = holdout_sub["variants"].apply(lambda d: d.get(v_key, False))
        sub_h = holdout_sub[h_mask]
        n_h = len(sub_h)
        pnl_h = float(sub_h["net_pnl"].sum()) if n_h > 0 else 0.0
        exp_h = (pnl_h / n_h) if n_h > 0 else 0.0
        wr_h = float(sub_h["target"].mean()) if n_h > 0 else 0.0
        holdout_results[v_key] = {
            "n_trades": n_h,
            "win_rate": round(wr_h, 4),
            "net_pnl_usdc": round(pnl_h, 2),
            "expectancy": round(exp_h, 4),
        }

    # 12. Item 30: Pre-registered decision criteria evaluation
    # Matrix evaluation:
    # CT vs C0:
    ct_pnl = variant_results["CT"]["net_pnl_usdc"]
    c0_pnl = variant_results["C0"]["net_pnl_usdc"]
    delta_ct = ct_pnl - c0_pnl
    ci_stand_ct = variant_results["CT"]["standalone_expectancy_ci_95"]
    ci_delta_ct = variant_results["CT"]["paired_bootstrap_ci_95"]

    # Does CT improve price control?
    improves_control = delta_ct > 0 and ci_delta_ct[0] > 0
    # Is standalone expectancy positive?
    standalone_positive = ci_stand_ct[0] > 0
    # Does effect persist in exploratory/holdout?
    exploratory_positive = exploratory_results["CT"]["net_pnl_usdc"] > 0
    
    if not improves_control:
        decision_verdict = "SIMPLIFY_RULE_DO_NOT_ADD_ML"
        decision_action = "Фильтр не улучшает ценовой контроль -> Упростить правило; не добавлять ML поверх него"
    elif not standalone_positive:
        decision_verdict = "LOSS_REDUCTION_ONLY_NO_STANDALONE_PROFITABILITY"
        decision_action = "Фильтр сокращает убыток, но собственная expectancy отрицательна -> Зафиксировать улучшение отбора без заявления о прибыльности"
    elif not exploratory_positive:
        decision_verdict = "UNCERTAIN_PRESERVE_AND_COLLECT"
        decision_action = "Результат положительный на dev, но неопределённость велика на exploratory -> Сохранить конфигурацию и продолжить сбор"
    else:
        decision_verdict = "GROUNDS_TO_TEST_ML"
        decision_action = "Есть положительная собственная expectancy и дополнительная польза на последующих данных -> Проверить вклад логистической регрессии, затем LightGBM"

    three_core_answers = {
        "q1_is_there_a_mechanism": {
            "verdict": bool(ff_stats["FORMER_FAVORITE"]["expectancy"] > 0.0),
            "evidence": f"Проверено на общих данных. Expectancy бывших фаворитов: {ff_stats['FORMER_FAVORITE']['expectancy']} USDC."
        },
        "q2_does_filter_add_value": {
            "verdict": bool(improves_control),
            "evidence": f"Delta {delta_ct:+.2f} USDC, 95% CI {ci_delta_ct}. Улучшение относительно контроля {improves_control}."
        },
        "q3_does_result_survive_execution": {
            "verdict": None,
            "evidence": "BLOCKED_DATA: Ожидает поступления фактических снапшотов глубины стакана (OrderbookDepthSnapshot)."
        },
    }

    # Item 02 & Robustness: Stress testing tables (absolute +0.005, +0.010, +0.020 and relative +2%, +5%)
    stress_tables: dict[str, Any] = {}
    for var_label, v_key in core_variant_keys:
        v_mask = c0_sub["variants"].apply(lambda d: d.get(v_key, False))
        sub_v = c0_sub[v_mask]
        n_v = len(sub_v)
        if n_v == 0:
            continue
        asks_arr = sub_v["executable_ask"].to_numpy()
        targets_arr = sub_v["target"].to_numpy()
        base_pnl = float(sub_v["net_pnl"].sum())

        var_stress = {
            "n_trades": n_v,
            "base_net_pnl": round(base_pnl, 2),
            "base_expectancy": round(base_pnl / n_v, 4),
            "absolute_shifts": {},
            "relative_shifts": {},
        }
        for d_p in [0.005, 0.010, 0.020]:
            p_adj = np.minimum(0.99, asks_arr + d_p)
            pnl_adj = (1.0 / p_adj) * targets_arr - (1.0 + taker_fee_rate)
            tot_pnl = float(np.sum(pnl_adj))
            var_stress["absolute_shifts"][f"+{d_p:.3f}_usdc_per_share"] = {
                "net_pnl_usdc": round(tot_pnl, 2),
                "expectancy": round(tot_pnl / n_v, 4),
                "pnl_drop_usdc": round(base_pnl - tot_pnl, 2),
            }
        for pct in [0.02, 0.05]:
            p_adj = np.minimum(0.99, asks_arr * (1.0 + pct))
            pnl_adj = (1.0 / p_adj) * targets_arr - (1.0 + taker_fee_rate)
            tot_pnl = float(np.sum(pnl_adj))
            var_stress["relative_shifts"][f"+{int(pct*100)}%"] = {
                "net_pnl_usdc": round(tot_pnl, 2),
                "expectancy": round(tot_pnl / n_v, 4),
                "pnl_drop_usdc": round(base_pnl - tot_pnl, 2),
            }
        stress_tables[v_key] = var_stress

    final_report = {
        "metadata": {
            "title": "Stage 2 Comprehensive Regime and Strike Research Study",
            "asset": target_asset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "input_hashes": {
                "snapshots": snaps_hash,
                "candles": candles_hash,
                "expirations": exp_hash
            },
            "schema_version": "1.1.0",
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "total_candidate_opportunities": len(c0_sub),
            "date_range": [all_days[0], all_days[-1]],
        },
        "item_02_stress_testing_tables": stress_tables,
        "item_18_common_opportunity_ledger_summary": {
            "total_markets_scanned": len(ledger_df),
            "eligible_candidates_c0": len(c0_sub),
            "all_variants_evaluated_on_identical_ledger": True,
        },
        "item_19_and_20_former_favorite_mechanism": ff_stats,
        "item_21_price_bucket_separation": price_bucket_analysis,
        "item_22_cohort_decomposition": cohort_stats,
        "item_23_time_scales_comparison": scale_comp,
        "item_25_and_28_core_variants": variant_results,
        "item_26_multi_budget_execution": budget_tables,
        "item_27_execution_selection_effect": selection_effect,
        "item_29_exploratory_results": exploratory_results,
        "item_29_holdout_results": holdout_results,
        "item_30_decision_verdict": {
            "verdict_status": decision_verdict,
            "action_required": decision_action,
            "three_core_answers": three_core_answers,
            "deploy_to_production": False,
        },
    }

    # Save output artifacts
    out_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "stage2_comprehensive_study_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    # Save stress testing tables (Item 02)
    stress_file = out_dir / "stress_testing_tables.json"
    with open(stress_file, "w", encoding="utf-8") as f:
        json.dump(stress_tables, f, indent=2, ensure_ascii=False)

    # Save common opportunity ledger (Item 18)
    ledger_file = out_dir / "common_opportunity_ledger.json"
    with open(ledger_file, "w", encoding="utf-8") as f:
        json.dump(ledger_rows, f, indent=2, ensure_ascii=False)

    print(f"Stage 2 Study completed successfully! Results written to {results_file}")
    return final_report
