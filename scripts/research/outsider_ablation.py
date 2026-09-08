"""
scripts/research/outsider_ablation.py

Sequential feature ablation study for Outsider Models (Item 2.8):
Evaluates strictly:
1. Variant A (Base: logit_mid_price, log_time_left, candidate_spread, interaction)
2. Variant A + z (Adds normalized distance to strike)
3. Variant A + momentum (Adds ret_outsider_30s and ret_outsider_120s)
4. Variant B (Full: A + z + momentum)

Evaluated on identical development folds (for candidate selection) and external test fold,
broken down across time segments (T-10, T-5, T-2) and price bins ([0.1-0.25), [0.25-0.40), [0.40-0.50)).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polyflip.models.outsider_feature_sets import (
    MODEL_A1_FEATURES,
    MODEL_B1_FEATURES,
)
from polyflip.models.outsider_trainer import (
    train_outsider_model,
    OutsiderTrainResult,
)
from polyflip.models.outsider_baselines import evaluate_outsider_predictions

# Exact ablation feature variants (Item 2.8)
ABLATION_VARIANTS: dict[str, tuple[str, ...]] = {
    "Variant_A": MODEL_A1_FEATURES,
    "Variant_A_plus_z": MODEL_A1_FEATURES + ("z_outsider",),
    "Variant_A_plus_mom": MODEL_A1_FEATURES + ("ret_outsider_30s", "ret_outsider_120s"),
    "Variant_B_Full": MODEL_B1_FEATURES,
}

PRICE_BINS: list[tuple[str, float, float]] = [
    ("Deep_Outsider [0.05, 0.25)", 0.05, 0.25),
    ("Mid_Outsider  [0.25, 0.40)", 0.25, 0.40),
    ("Near_Even     [0.40, 0.50)", 0.40, 0.50),
]


def run_outsider_ablation(
    df: pd.DataFrame,
    fee_rate: float = 0.002,
    min_edge: float = 0.02,
) -> dict[str, Any]:
    """
    Executes sequential ablation on complete-B cohort.
    Splits into development folds (for candidate selection) and external test fold.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    folds = sorted(df["fold"].unique())
    if len(folds) < 2:
        dev_mask = np.ones(len(df), dtype=bool)
        test_mask = dev_mask
    else:
        # Last fold is external test fold, earlier folds are development
        test_fold = folds[-1]
        test_mask = (df["fold"] == test_fold).to_numpy()
        dev_mask = ~test_mask

    df_dev = df[dev_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    results: dict[str, Any] = {
        "dev_metrics": {},
        "test_metrics": {},
        "time_breakdowns": {},
        "price_breakdowns": {},
        "marginal_deltas": {},
        "summary_table": [],
    }

    # 1. Evaluate each variant on development folds (cross-validated OOF)
    dev_results: dict[str, OutsiderTrainResult] = {}
    for var_name, var_feats in ABLATION_VARIANTS.items():
        res = train_outsider_model(
            df_dev,
            custom_features=var_feats,
            fee_rate=fee_rate,
            min_edge=min_edge,
        )
        dev_results[var_name] = res
        results["dev_metrics"][var_name] = res.metrics

    # 2. Evaluate final models on external test fold
    test_results: dict[str, dict[str, Any]] = {}
    y_test = df_test["target"].to_numpy()
    ask_test = df_test["executable_ask"].to_numpy() if "executable_ask" in df_test.columns else None

    for var_name, res in dev_results.items():
        final_model = res.final_model
        feats = res.feature_names
        X_test = df_test[list(feats)]
        p_test = final_model.predict_proba(X_test)[:, 1]
        m_test = evaluate_outsider_predictions(
            y_true=y_test,
            p_win=p_test,
            executable_ask=ask_test,
            fee_rate=fee_rate,
            min_edge=min_edge,
        )
        test_results[var_name] = m_test
        results["test_metrics"][var_name] = m_test

    # 3. Time segment breakdowns on test fold
    time_points = sorted(df_test["target_time_point"].unique()) if "target_time_point" in df_test.columns else [10.0, 5.0, 2.0]
    for tp in time_points:
        sub_mask = (df_test["target_time_point"] == tp).to_numpy() if "target_time_point" in df_test.columns else np.ones(len(df_test), dtype=bool)
        if not np.any(sub_mask):
            continue
        tp_label = f"T_{int(tp)}m"
        results["time_breakdowns"][tp_label] = {}
        for var_name, res in dev_results.items():
            X_sub = df_test.loc[sub_mask, list(res.feature_names)]
            p_sub = res.final_model.predict_proba(X_sub)[:, 1]
            ask_sub = ask_test[sub_mask] if ask_test is not None else None
            m_sub = evaluate_outsider_predictions(y_test[sub_mask], p_sub, ask_sub, fee_rate, min_edge)
            results["time_breakdowns"][tp_label][var_name] = m_sub

    # 4. Price bin breakdowns on test fold
    mids_test = df_test["outsider_mid"].to_numpy() if "outsider_mid" in df_test.columns else df_test["mid_price"].to_numpy()
    for bin_label, p_min, p_max in PRICE_BINS:
        bin_mask = (mids_test >= p_min) & (mids_test < p_max)
        if not np.any(bin_mask):
            continue
        results["price_breakdowns"][bin_label] = {}
        for var_name, res in dev_results.items():
            X_bin = df_test.loc[bin_mask, list(res.feature_names)]
            p_bin = res.final_model.predict_proba(X_bin)[:, 1]
            ask_bin = ask_test[bin_mask] if ask_test is not None else None
            m_bin = evaluate_outsider_predictions(y_test[bin_mask], p_bin, ask_bin, fee_rate, min_edge)
            results["price_breakdowns"][bin_label][var_name] = m_bin

    # 5. Compute marginal deltas relative to Variant A (on external test)
    base_m = test_results["Variant_A"]
    for var_name in ["Variant_A_plus_z", "Variant_A_plus_mom", "Variant_B_Full"]:
        curr_m = test_results[var_name]
        results["marginal_deltas"][f"{var_name}_vs_A"] = {
            "delta_brier": round(float(curr_m["brier"] - base_m["brier"]), 6),
            "delta_log_loss": round(float(curr_m["log_loss"] - base_m["log_loss"]), 6),
            "delta_ece": round(float(curr_m["ece"] - base_m["ece"]), 6),
            "delta_pnl": round(float(curr_m["total_pnl"] - base_m["total_pnl"]), 4),
            "delta_expectancy": round(float(curr_m["expectancy"] - base_m["expectancy"]), 6),
        }

    # 6. Build summary table rows
    for var_name in ABLATION_VARIANTS:
        dev_m = results["dev_metrics"][var_name]
        test_m = results["test_metrics"][var_name]
        results["summary_table"].append({
            "variant": var_name,
            "n_features": len(ABLATION_VARIANTS[var_name]),
            "dev_brier": dev_m["brier"],
            "dev_log_loss": dev_m["log_loss"],
            "dev_ece": dev_m["ece"],
            "dev_pnl": dev_m["total_pnl"],
            "test_brier": test_m["brier"],
            "test_log_loss": test_m["log_loss"],
            "test_ece": test_m["ece"],
            "test_pnl": test_m["total_pnl"],
            "test_trades": test_m["n_trades"],
            "test_win_rate": test_m["win_rate"],
        })

    return results


def print_ablation_summary(results: dict[str, Any]) -> None:
    """Prints formatted ASCII ablation table to stdout."""
    print("\n" + "=" * 95)
    print("OUTSIDER FEATURE ABLATION STUDY (Item 2.8)")
    print("=" * 95)
    header = f"{'Variant':<20} | {'Feats':<5} | {'Dev Brier':<9} | {'Dev LogLoss':<11} | {'Test Brier':<10} | {'Test LogLoss':<12} | {'Test PnL':<8} | {'Trades':<6}"
    print(header)
    print("-" * 95)
    for r in results["summary_table"]:
        line = (
            f"{r['variant']:<20} | {r['n_features']:<5} | "
            f"{r['dev_brier']:<9.4f} | {r['dev_log_loss']:<11.4f} | "
            f"{r['test_brier']:<10.4f} | {r['test_log_loss']:<12.4f} | "
            f"{r['test_pnl']:<8.2f} | {r['test_trades']:<6}"
        )
        print(line)
    print("=" * 95)
    print("\nMarginal Deltas relative to Variant A (External Test):")
    for delta_name, d in results["marginal_deltas"].items():
        print(f"  [{delta_name}]")
        print(f"    d_Brier:      {d['delta_brier']:+.6f} {'(better)' if d['delta_brier'] < 0 else ''}")
        print(f"    d_LogLoss:    {d['delta_log_loss']:+.6f} {'(better)' if d['delta_log_loss'] < 0 else ''}")
        print(f"    d_ECE:        {d['delta_ece']:+.6f}")
        print(f"    d_PnL:        {d['delta_pnl']:+.4f}")
        print(f"    d_Expectancy: {d['delta_expectancy']:+.6f}")
    print("=" * 95 + "\n")


def generate_ablation_dataset(n_markets: int = 16) -> pd.DataFrame:
    """Generates synthetic multi-market decision rows for ablation benchmarks."""
    from polyflip.models.point_in_time_features import compute_outsider_model_features
    np.random.seed(42)
    rows = []
    for m in range(n_markets):
        m_id = f"market_{m:03d}"
        market_bias = np.random.uniform(-0.3, 0.3)
        for t_idx, tl in enumerate([10.0, 5.0, 2.0]):
            outsider_mid = float(np.clip(0.35 + 0.05 * market_bias + np.random.normal(0, 0.02), 0.15, 0.48))
            spread = 0.02
            k = 50000.0
            s = 50000.0 + 300.0 * market_bias + np.random.normal(0, 50.0)
            lag30 = s - np.random.normal(10.0 * market_bias, 15.0)
            lag120 = s - np.random.normal(30.0 * market_bias, 30.0)

            prob_win = 1.0 / (1.0 + np.exp(-(1.5 * (s - k) / 200.0 + 0.8 * (s - lag30) / 20.0)))
            y = int(np.random.rand() < prob_win)

            rows.append({
                "market_id": m_id,
                "fold": m % 4,
                "time_left_min": tl,
                "target_time_point": tl,
                "mid_price": outsider_mid,
                "outsider_mid": outsider_mid,
                "candidate_side": "UP" if m % 2 == 0 else "DOWN",
                "spread": spread,
                "candidate_spread": spread,
                "underlying_price": s,
                "strike_value": k,
                "underlying_lag_30s": lag30,
                "underlying_lag_120s": lag120,
                "sigma_1m": 0.0015,
                "executable_ask": outsider_mid + 0.01,
                "target": y,
                "y_candidate_win": y,
            })

    df = pd.DataFrame(rows)
    return compute_outsider_model_features(df)


if __name__ == "__main__":
    print("Running outsider ablation benchmark on synthetic integration dataset...")
    df_cohort = generate_ablation_dataset(n_markets=20)
    res = run_outsider_ablation(df_cohort)
    print_ablation_summary(res)
