"""load.py v1.0.0 — chunk readers for shared dataset builder (spec lgbm_shared_v1)."""
import glob
import os

import pandas as pd

DATA = r"D:\lgbm-audit-v1\data"
EXP = os.path.join(DATA, "exp")

FUNNEL_COLS = ["id", "created_at", "market_id", "asset", "condition_id",
    "trading_mode", "execution_mode", "decision_run_id", "final_action",
    "skip_reason", "proposed_action",
    "direction_model_key", "direction_model_version", "direction_regime",
    "direction_status", "direction_probability", "direction_value",
    "direction_p_up", "direction_p_down", "direction_threshold_up",
    "direction_threshold_down", "direction_discount_mult",
    "primary_model_key", "primary_model_version",
    "confirm_model_key", "confirm_model_version", "confirm_direction",
    "confirm_passed", "entry_model_key", "entry_model_version",
    "entry_model_phase", "entry_model_source", "entry_status", "entry_model_ece",
    "fallback_reason", "p_flip", "p_market_yes", "p_logreg_yes", "p_lgbm_yes",
    "p_logreg_win", "p_candidate_win", "candidate_side", "candidate_ask",
    "gross_edge", "cost_buffer", "net_edge", "edge", "min_edge_used",
    "fresh_price", "threshold_lower", "threshold_upper",
    "strike_source", "strike_proxy", "underlying_price", "distance_to_strike_pct",
    "max_acceptable_price", "mrf_mode", "mrf_phase", "mrf_asset_phase",
    "mrf_applied", "mrf_strength", "mrf_confidence", "mrf_multiplier",
    "mrf_policy_version", "mrf_gate_would_block", "mrf_gate_reason",
    "mrf_final_action", "mrf_final_bet", "weighted_policy_mode",
    "weighted_selected_side", "weighted_p_market_yes", "weighted_p_logreg_yes",
    "weighted_p_lgbm_yes", "weighted_p_final_yes", "weighted_market_weight",
    "weighted_logreg_weight", "weighted_lgbm_weight", "weighted_yes_net_ev",
    "weighted_no_net_ev", "weighted_fee_rate", "weighted_fee_source",
    "weighted_selection_reason", "weighted_policy_id", "weighted_edge_lower_bound",
    "weighted_missing_components", "weighted_expected_execution_price"]

SNAP_COLS = ["id", "market_id", "asset", "recorded_at", "received_timestamp",
    "market_timestamp", "mid_price", "best_bid", "best_ask", "poly_up_best_bid",
    "poly_up_best_ask", "poly_up_mid", "poly_down_best_bid", "poly_down_best_ask",
    "poly_down_mid", "spread", "time_left_seconds", "market_end_at",
    "final_outcome", "binance_spot_mid", "binance_perp_mid", "binance_price",
    "oracle_price", "strike_value", "strike_source", "volume_5min",
    "price_velocity"]


def read_funnel_day(day):
    p = os.path.join(EXP, "funnel", "funnel_%s.csv.gz" % day)
    df = pd.read_csv(p, usecols=FUNNEL_COLS, dtype={"market_id": str,
                     "decision_run_id": str}, low_memory=False)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")
    return df


def read_snaps_day(day):
    p = os.path.join(EXP, "snapsdec", "snapsdec_%s.csv.gz" % day)
    df = pd.read_csv(p, dtype={"market_id": str}, low_memory=False)
    for c in ("recorded_at", "received_timestamp"):
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
    return df


def funnel_days():
    return sorted(f[7:17] for f in os.listdir(os.path.join(EXP, "funnel"))
                  if f.startswith("funnel_") and f.endswith(".csv.gz"))


def read_candles():
    import gzip
    p = os.path.join(EXP, "full", "candles.csv.gz")
    df = pd.read_csv(p)
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed", errors="coerce")
    return df


def read_livemarkets():
    p = os.path.join(EXP, "full", "livemarkets.csv.gz")
    return pd.read_csv(p, dtype={"market_id": str}, low_memory=False)


def read_trades():
    p = os.path.join(EXP, "full", "trades.csv.gz")
    df = pd.read_csv(p, dtype={"market_id": str, "decision_run_id": str,
                               "model_key": str}, low_memory=False)
    return df


def read_exec(name):
    p = os.path.join(EXP, "full", "%s.csv.gz" % name)
    return pd.read_csv(p, low_memory=False)


def read_inventory():
    return pd.read_csv(r"D:\lgbm-audit-v1\out\MODEL_INVENTORY.csv",
                       dtype={"id": str})
