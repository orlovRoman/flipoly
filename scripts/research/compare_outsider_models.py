"""
scripts/research/compare_outsider_models.py

Comprehensive model comparison and final configuration selection (Item 2.10):
Generates a unified comparison table for:
  - M0 (Market Price Baseline)
  - Mlegacy (Legacy Baseline)
  - Model A1 (Compact 4-feature Outsider LogReg)
  - Model B1 (A1 + z_outsider + ret_30s + ret_120s)
  - Model B + LGBM Veto
  - Model B + LGBM Input

Reports: unique markets, coverage, Brier, log loss, ECE, reliability across 0.05 bins,
net PnL, expectancy, payoff ratio, maximum drawdown, and paired delta PnL relative to Model A and M0.
Selects the simplest configuration that yields robust positive net expectancy.
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

from polyflip.models.probability_metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
)
from polyflip.crypto.edge import compute_net_ev_per_share
from polyflip.models.outsider_baselines import (
    MarketPriceBaseline,
    LegacyOutsiderBaseline,
)
from polyflip.models.outsider_trainer import train_outsider_model
from polyflip.trading.combined_voting import evaluate_lgbm_outsider_interaction
from polyflip.research.outsider_replay import OutsiderReplayEngine, ReplayPolicy
from polyflip.research.reporting_helpers import (
    compute_drawdown,
    compute_payoff_ratio,
    compute_clustered_uncertainty,
    compute_paired_cluster_delta,
    generate_price_bins_report,
)


def select_candidate_configuration(
    summary_rows: Sequence[dict[str, Any]],
    source_kind: str = "HISTORICAL",
) -> tuple[str, str | None]:
    """
    R1 & R8: Conclusive model configuration selection (Item 1.31).
    If source_kind is SYNTHETIC or DEMO, strictly returns ("DEMO_ONLY", None).
    If all models have net_pnl <= 0 or their expectancy CI crosses zero,
    returns ("EDGE_NOT_SUPPORTED", None) or ("INCONCLUSIVE", None).
    Only when a candidate achieves robust positive net expectancy and CI lower > 0
    is CANDIDATE_SELECTED declared.
    """
    if str(source_kind).upper() in ("SYNTHETIC", "DEMO", "DEMO_ONLY"):
        return "DEMO_ONLY", None

    candidates = [
        r for r in summary_rows
        if r.get("model") in ("MODEL_A1", "MODEL_B1", "MODEL_B_PLUS_VETO", "MODEL_B_PLUS_LGBM_INPUT")
    ]
    if not candidates:
        return "INCONCLUSIVE", None

    # Check if ANY candidate has strictly positive net_pnl AND positive lower CI
    viable = []
    for c in candidates:
        val_pnl = c.get("net_pnl")
        net_pnl = float(val_pnl) if val_pnl is not None and np.isfinite(float(val_pnl)) else 0.0
        val_ci = c.get("expectancy_ci_lower")
        ci_lower = float(val_ci) if val_ci is not None and np.isfinite(float(val_ci)) else -1.0
        if net_pnl > 0.0 and ci_lower > 0.0:
            viable.append(c)

    if not viable:
        return "EDGE_NOT_SUPPORTED", None

    # Select highest net PnL among viable
    best = max(viable, key=lambda x: float(x.get("net_pnl", 0.0)))
    return "CANDIDATE_SELECTED", best.get("model")


def compare_all_models(
    df: pd.DataFrame,
    fee_rate: float = 0.002,
    min_edge: float = 0.02,
    source_kind: str = "HISTORICAL",
) -> dict[str, Any]:
    """
    Evaluates all model configurations on identical complete cohort using OutsiderReplayEngine.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    from polyflip.models.point_in_time_features import compute_outsider_model_features

    df = df.copy()
    if "logit_mid_price" not in df.columns or "ret_outsider_30s" not in df.columns:
        df = compute_outsider_model_features(df)

    total_markets = int(df["market_id"].nunique())
    y_true = df["target"].to_numpy()
    asks = df["executable_ask"].to_numpy() if "executable_ask" in df.columns else np.full(len(df), 0.35)
    if "market_id" in df.columns and "decision_at" in df.columns:
        dates = pd.to_datetime(df["decision_at"], utc=True).dt.date.astype(str)
        clusters_full = (df["market_id"].astype(str) + "_" + dates).to_numpy()
    elif "market_id" in df.columns:
        clusters_full = df["market_id"].to_numpy()
    else:
        clusters_full = np.arange(len(df))

    # 1. Baseline M0
    m0 = MarketPriceBaseline()
    p_m0 = m0.predict_proba(df)[:, 1]

    # 2. Baseline Mlegacy (Authentic BTC_leaning@11)
    m_leg = LegacyOutsiderBaseline()
    p_leg = m_leg.predict_proba(df)[:, 1]

    # 3. Model A1 (Trained)
    res_a = train_outsider_model(df, feature_set="MODEL_A1", fee_rate=fee_rate, min_edge=min_edge)
    p_a = res_a.oof_predictions

    # 4. Model B1 (Trained)
    res_b = train_outsider_model(df, feature_set="MODEL_B1", fee_rate=fee_rate, min_edge=min_edge)
    p_b = res_b.oof_predictions

    # Helper to run unified research replay engine (Items 1.22, 1.23, 1.29)
    def _run_model_replay(probs: np.ndarray, veto_mode: bool = False) -> tuple[list[float], list[int], int, float, float, float]:
        decisions = []
        for i in range(len(df)):
            decisions.append({
                "market_id": df["market_id"].iloc[i] if "market_id" in df.columns else f"m_{i}",
                "decision_at": df["decision_at"].iloc[i] if "decision_at" in df.columns else pd.Timestamp.now(tz="UTC"),
                "time_left_min": df["time_left_min"].iloc[i] if "time_left_min" in df.columns else 5.0,
                "candidate_side": df["candidate_side"].iloc[i] if "candidate_side" in df.columns else "UP",
                "executable_ask": asks[i],
                "p_win": probs[i],
                "target": y_true[i],
                "lgbm_direction": df.get("lgbm_direction", pd.Series(["NONE"] * len(df))).iloc[i],
            })
        eng = OutsiderReplayEngine(policy=ReplayPolicy(
            max_positions_per_market=1,
            min_edge=min_edge,
            default_fee_rate=fee_rate,
            veto_mode=veto_mode,
        ))
        ledger = eng.run(decisions)
        executed = ledger.executed_trades
        pnls = [t["realized_pnl"] for t in executed]
        indices = [i for i, d in enumerate(decisions) if any(t["market_id"] == d["market_id"] and t["decision_at"] == d["decision_at"] for t in executed)]
        wins = sum(1 for t in executed if t.get("outcome", 0) == 1)
        tot_pnl = ledger.total_pnl
        payoff = compute_payoff_ratio(pnls)
        max_dd = compute_drawdown(pnls)
        return pnls, indices, wins, tot_pnl, payoff, max_dd

    models_data = [
        ("M0_MARKET", p_m0),
        ("MLEGACY", p_leg),
        ("MODEL_A1", p_a),
        ("MODEL_B1", p_b),
    ]

    report_rows = []
    pnl_dict = {}
    trades_dict = {}

    for name, p_vals in models_data:
        valid_mask = np.isfinite(y_true) & np.isfinite(p_vals)
        coverage = float(np.mean(valid_mask)) if len(y_true) > 0 else 0.0

        if valid_mask.sum() > 0:
            brier = float(brier_score(y_true[valid_mask], p_vals[valid_mask]))
            ll = float(log_loss_score(y_true[valid_mask], p_vals[valid_mask]))
            ece_val, diag = expected_calibration_error(y_true[valid_mask], p_vals[valid_mask], n_bins=20)
            ece = float(ece_val) if ece_val is not None else 0.0
        else:
            brier, ll, ece = 0.0, 0.0, 0.0

        pnls, indices, wins, total_pnl, payoff, max_dd = _run_model_replay(p_vals)
        n_trades = len(pnls)
        pnl_dict[name] = total_pnl
        trades_dict[name] = (pnls, indices)

        win_rate = float(wins / n_trades) if n_trades > 0 else 0.0
        expectancy = float(total_pnl / n_trades) if n_trades > 0 else 0.0

        trade_clusters = clusters_full[indices] if len(indices) > 0 else []
        unc = compute_clustered_uncertainty(pnls, trade_clusters)

        report_rows.append({
            "model": name,
            "prediction_id": f"pred_{name.lower()}",
            "unique_markets": total_markets,
            "coverage": round(coverage, 4),
            "brier": round(brier, 4),
            "log_loss": round(ll, 4),
            "ece": round(ece, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "net_pnl": round(total_pnl, 4),
            "expectancy": round(expectancy, 6),
            "expectancy_ci_lower": unc["ci_lower"],
            "expectancy_ci_upper": unc["ci_upper"],
            "payoff": round(payoff, 4),
            "max_dd": round(max_dd, 4),
        })

    # 5. Model B + LGBM Veto and Input (Items 1.28, 1.29)
    df_interaction = df.copy()
    df_interaction["p_b_win"] = p_b
    if "lgbm_direction" not in df_interaction.columns:
        df_interaction["lgbm_direction"] = "NONE"
    if "lgbm_oof_prob" not in df_interaction.columns:
        if "p_lgbm" in df_interaction.columns:
            df_interaction["lgbm_oof_prob"] = df_interaction["p_lgbm"]
        else:
            df_interaction["lgbm_oof_prob"] = 0.5

    lgbm_inter = evaluate_lgbm_outsider_interaction(df_interaction, min_edge=min_edge, fee_rate=fee_rate)
    in_res = lgbm_inter.get("b_plus_lgbm_input", {})
    p_meta = in_res.get("meta_probs", p_b)

    # Replay Model B with Veto
    v_pnls, v_indices, v_wins, v_tot_pnl, v_payoff, v_max_dd = _run_model_replay(p_b, veto_mode=True)
    v_clusters = clusters_full[v_indices] if len(v_indices) > 0 else []
    v_unc = compute_clustered_uncertainty(v_pnls, v_clusters)

    report_rows.append({
        "model": "MODEL_B_PLUS_VETO",
        "prediction_id": "pred_b_plus_veto",
        "unique_markets": total_markets,
        "coverage": round(float(np.mean(np.isfinite(p_b))), 4),
        "brier": round(float(brier_score(y_true[np.isfinite(p_b)], p_b[np.isfinite(p_b)])), 4) if np.any(np.isfinite(p_b)) else 0.0,
        "log_loss": round(float(log_loss_score(y_true[np.isfinite(p_b)], p_b[np.isfinite(p_b)])), 4) if np.any(np.isfinite(p_b)) else 0.0,
        "ece": round(float(expected_calibration_error(y_true[np.isfinite(p_b)], p_b[np.isfinite(p_b)], n_bins=20)[0] or 0.0), 4) if np.any(np.isfinite(p_b)) else 0.0,
        "n_trades": len(v_pnls),
        "win_rate": round(v_wins / len(v_pnls), 4) if v_pnls else 0.0,
        "net_pnl": round(v_tot_pnl, 4),
        "expectancy": round(v_tot_pnl / len(v_pnls), 6) if v_pnls else 0.0,
        "expectancy_ci_lower": v_unc["ci_lower"],
        "expectancy_ci_upper": v_unc["ci_upper"],
        "payoff": v_payoff,
        "max_dd": v_max_dd,
    })
    pnl_dict["MODEL_B_PLUS_VETO"] = v_tot_pnl
    trades_dict["MODEL_B_PLUS_VETO"] = (v_pnls, v_indices)

    # Replay Model B with Input (meta_probs)
    in_pnls, in_indices, in_wins, in_tot_pnl, in_payoff, in_max_dd = _run_model_replay(p_meta, veto_mode=False)
    in_clusters = clusters_full[in_indices] if len(in_indices) > 0 else []
    in_unc = compute_clustered_uncertainty(in_pnls, in_clusters)

    valid_meta = np.isfinite(y_true) & np.isfinite(p_meta)
    in_brier = float(brier_score(y_true[valid_meta], p_meta[valid_meta])) if np.any(valid_meta) else 0.0
    in_ll = float(log_loss_score(y_true[valid_meta], p_meta[valid_meta])) if np.any(valid_meta) else 0.0
    in_ece = float(expected_calibration_error(y_true[valid_meta], p_meta[valid_meta], n_bins=20)[0] or 0.0) if np.any(valid_meta) else 0.0

    report_rows.append({
        "model": "MODEL_B_PLUS_LGBM_INPUT",
        "prediction_id": "pred_meta_probs",
        "unique_markets": total_markets,
        "coverage": round(float(np.mean(valid_meta)), 4),
        "brier": round(in_brier, 4),
        "log_loss": round(in_ll, 4),
        "ece": round(in_ece, 4),
        "n_trades": len(in_pnls),
        "win_rate": round(in_wins / len(in_pnls), 4) if in_pnls else 0.0,
        "net_pnl": round(in_tot_pnl, 4),
        "expectancy": round(in_tot_pnl / len(in_pnls), 6) if in_pnls else 0.0,
        "expectancy_ci_lower": in_unc["ci_lower"],
        "expectancy_ci_upper": in_unc["ci_upper"],
        "payoff": in_payoff,
        "max_dd": in_max_dd,
    })
    pnl_dict["MODEL_B_PLUS_LGBM_INPUT"] = in_tot_pnl
    trades_dict["MODEL_B_PLUS_LGBM_INPUT"] = (in_pnls, in_indices)

    # Compute paired delta PnL relative to Model A1 and M0 with cluster-robust CI (Item 1.26)
    base_pnl_a = pnl_dict.get("MODEL_A1", 0.0)
    base_pnl_m0 = pnl_dict.get("M0_MARKET", 0.0)
    a_pnls, a_indices = trades_dict.get("MODEL_A1", ([], []))

    for r in report_rows:
        name = r["model"]
        m_pnls, m_indices = trades_dict.get(name, ([], []))
        paired_a = compute_paired_cluster_delta(
            pnls_model=m_pnls,
            indices_model=m_indices,
            pnls_ref=a_pnls,
            indices_ref=a_indices,
            cluster_ids_full=clusters_full,
        )
        r["delta_pnl_vs_A"] = round(r["net_pnl"] - base_pnl_a, 4)
        r["delta_pnl_vs_M0"] = round(r["net_pnl"] - base_pnl_m0, 4)
        r["delta_pnl_ci_lower_vs_A"] = paired_a["ci_lower"]
        r["delta_pnl_ci_upper_vs_A"] = paired_a["ci_upper"]

    # Item 1.31: Selection logic - strictly separate selection from confirmation
    status, winner = select_candidate_configuration(report_rows, source_kind=source_kind)
    if str(source_kind).upper() in ("SYNTHETIC", "DEMO", "DEMO_ONLY"):
        status = "DEMO_ONLY"
        selected_config = None
        rationale = "Synthetic demo run completed: no edge claims permitted on synthetic data"
    elif status == "CANDIDATE_SELECTED" and winner is not None:
        selected_config = winner
        rationale = f"Candidate {winner} confirmed with positive net expectancy and 95% CI > 0"
    else:
        selected_config = None
        rationale = "No candidate demonstrated statistically significant positive edge on outer validation"

    return {
        "status": status,
        "summary_table": report_rows,
        "selected_configuration": selected_config,
        "selection_rationale": rationale,
        "lgbm_interaction": lgbm_inter,
    }

    return {
        "status": status,
        "summary_table": report_rows,
        "selected_configuration": selected_config,
        "selection_rationale": rationale,
        "lgbm_interaction": lgbm_inter,
    }


def print_comparison_table(res: dict[str, Any]) -> None:
    """Prints formatted comparison table to stdout."""
    print("\n" + "=" * 136)
    print("STAGE 2 OUTSIDER MODELS BENCHMARK & COMPARISON TABLE (Item 2.10)")
    print("=" * 136)
    header = (
        f"{'Model':<24} | {'Brier':<7} | {'LogLoss':<8} | {'ECE':<6} | "
        f"{'Trades':<6} | {'WinRate':<7} | {'NetPnL':<8} | {'Payoff':<6} | {'MaxDD':<6} | "
        f"{'Exp [95% CI]':<22} | {'d_PnL vs A [95% CI]':<23}"
    )
    print(header)
    print("-" * 136)
    for r in res["summary_table"]:
        brier_s = f"{r['brier']:<7.4f}" if r.get('brier') is not None else "N/A    "
        ll_s = f"{r['log_loss']:<8.4f}" if r.get('log_loss') is not None else "N/A     "
        ece_s = f"{r['ece']:<6.4f}" if r.get('ece') is not None else "N/A   "
        wr_s = f"{r['win_rate']:<7.3f}" if r.get('win_rate') is not None else "N/A    "
        net_s = f"{r['net_pnl']:<8.2f}" if r.get('net_pnl') is not None else "N/A     "
        payoff_s = f"{r['payoff']:<6.2f}" if r.get('payoff') is not None else "N/A   "
        dd_s = f"{r['max_dd']:<6.2f}" if r.get('max_dd') is not None else "N/A   "
        
        if r.get('expectancy') is not None:
            exp_str = f"{r['expectancy']:+.4f} [{r.get('expectancy_ci_lower', 0.0):+.3f},{r.get('expectancy_ci_upper', 0.0):+.3f}]"
        else:
            exp_str = "N/A"
            
        if r.get('delta_pnl_vs_A') is not None:
            d_pnl_str = f"{r['delta_pnl_vs_A']:+6.2f} [{r.get('delta_pnl_ci_lower_vs_A', 0.0):+5.1f},{r.get('delta_pnl_ci_upper_vs_A', 0.0):+5.1f}]"
        else:
            d_pnl_str = "-"

        line = (
            f"{r['model']:<24} | {brier_s} | {ll_s} | {ece_s} | "
            f"{r['n_trades']:<6} | {wr_s} | {net_s} | {payoff_s} | {dd_s} | "
            f"{exp_str:<22} | {d_pnl_str:<23}"
        )
        print(line)
    print("=" * 136)
    print(f"\nFinal Recommended Configuration: {res['selected_configuration']}")
    print(f"Selection Rationale:             {res['selection_rationale']}")
    print("=" * 136 + "\n")


if __name__ == "__main__":
    from scripts.research.outsider_ablation import generate_ablation_dataset
    print("Running comprehensive model comparison benchmark...")
    df_sample = generate_ablation_dataset(n_markets=24)
    res = compare_all_models(df_sample)
    print_comparison_table(res)
